//! Shared head-major scratch for token-chunk recurrent prefill.
use super::super::capture::error;
use super::{
    ORDERED_BLOCK, ORDERED_DIM, ORDERED_WIDTHS,
    budget::{enabled, opted_in, ordered},
    kernels, prepared,
};
use crate::constants::GDN_PROJECTED_INPUTS;
use crate::tuning::recurrent::select;
use crate::{
    device::{CudaDevice, device_error},
    resident::{ActivationArena, ProgramWeights},
};
use cutile::prelude::*;
use infer_core::{Result, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

#[derive(Default)]
pub struct Workspace {
    buffers: BTreeMap<(usize, usize), Tensor<f32>>,
    lanes: usize,
    prepared: prepared::Buffers,
    wy_blocks: BTreeMap<TensorId, usize>,
    prefer_wy: bool,
}
impl Workspace {
    pub(crate) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        lanes: usize,
    ) -> Result<Self> {
        // Shared F32 inputs and independent 16-column blocks preserve the scalar norm, prediction and readout trees.
        // Captured 64/256-row fixtures and the complete checkpoint agree bit for bit.
        // Quantized projections use only the exact ordered geometry automatically.
        // Other shapes retain their existing scalar or explicit diagnostic policy.
        let opted_in = opted_in(weights);
        let legacy_shapes = enabled(weights, opted_in);
        if !legacy_shapes && !graph.nodes.iter().any(|node| ordered(node, weights, lanes)) {
            return Ok(Self {
                lanes,
                ..Self::default()
            });
        }
        let mut buffers = BTreeMap::new();
        let mut wy_blocks = BTreeMap::new();
        for node in &graph.nodes {
            if ordered(node, weights, lanes) {
                wy_blocks.insert(node.outputs[0], ORDERED_BLOCK);
            }
        }
        if opted_in {
            for node in &graph.nodes {
                if let TensorOp::Delta {
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                } = node.op
                    && key_dim == value_dim
                    && node.inputs[..GDN_PROJECTED_INPUTS]
                        .iter()
                        .all(|id| !weights.constants.contains_key(id))
                {
                    let shape = [key_heads, value_heads, key_dim, lanes];
                    let selected = if key_dim == ORDERED_DIM && ORDERED_WIDTHS.contains(&lanes) {
                        Some(ORDERED_BLOCK)
                    } else {
                        select(device, shape, weights.autotune, &weights.resident_tuning)
                    };
                    if let Some(block) = selected {
                        wy_blocks.insert(node.outputs[0], block);
                    }
                }
            }
        }
        for node in &graph.nodes {
            if !legacy_shapes && !ordered(node, weights, lanes) {
                continue;
            }
            if let TensorOp::Delta {
                value_heads,
                value_dim,
                ..
            } = node.op
                && !buffers.contains_key(&(value_heads, value_dim))
            {
                buffers.insert(
                    (value_heads, value_dim),
                    api::zeros::<f32>(&[value_heads, lanes, value_dim])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                );
            }
        }
        Ok(Self {
            buffers,
            prepared: prepared::allocate(device, graph, weights, lanes)?,
            lanes,
            wy_blocks,
            prefer_wy: opted_in || !legacy_shapes,
        })
    }
    pub(crate) fn record(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        weights: &ProgramWeights,
        states: &mut super::super::batch::States,
        metadata: &Tensor<i32>,
    ) -> StdResult<bool, DeviceError> {
        #[cfg(test)]
        if super::LEGACY_CAPTURE.get() {
            return Ok(false);
        }
        if super::super::conv_prefill::record(
            scope, node, arena, weights, states, metadata, self.lanes,
        )? {
            return Ok(true);
        }
        #[cfg(test)]
        if super::LEGACY_DELTA_CAPTURE.get() {
            return Ok(false);
        }
        if self.buffers.is_empty() {
            return Ok(false);
        }
        let TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = node.op
        else {
            return Ok(false);
        };
        if self.prefer_wy && !self.wy_blocks.contains_key(&node.outputs[0]) {
            return Ok(false);
        }
        if key_heads == 0
            || !value_heads.is_multiple_of(key_heads)
            || key_dim != value_dim
            || !key_dim.is_power_of_two()
        {
            return Err(error("chunk delta dimensions"));
        }
        if node.inputs[..GDN_PROJECTED_INPUTS]
            .iter()
            .any(|id| weights.constants.contains_key(id))
        {
            return Ok(false);
        }
        let state = states
            .get_mut(&node.states[0])
            .and_then(|s| s.first_mut())
            .ok_or_else(|| error("chunk delta state"))?;
        let scratch = self
            .buffers
            .get_mut(&(value_heads, value_dim))
            .ok_or_else(|| error("chunk delta scratch"))?;
        let slot = arena.slot(node.outputs[0]).map_err(error)?;
        let output = arena.buffers[slot]
            .take()
            .ok_or_else(|| error("chunk delta output"))?;
        let original: Vec<usize> = output
            .shape()
            .iter()
            .map(|&x| usize::try_from(x).map_err(error))
            .collect::<StdResult<_, _>>()?;
        let mut output = output.reshape(&[self.lanes, value_heads, value_dim])?;
        // The measured ordered path splits independent value columns without changing sums.
        // Other shapes use their existing qualified block policy.
        let block = if key_dim == ORDERED_DIM && ORDERED_WIDTHS.contains(&self.lanes) {
            ORDERED_BLOCK
        } else {
            self.wy_blocks
                .get(&node.outputs[0])
                .copied()
                .map_or_else(|| super::super::capture_state::value_block(value_dim), Ok)
                .map_err(|_| error("chunk delta value split"))?
        };
        record_delta(
            scope,
            node,
            arena,
            weights,
            state,
            scratch,
            self.prepared.get_mut(&(key_heads, value_heads, key_dim)),
            metadata,
            [key_heads, value_heads, key_dim, self.lanes, block],
            self.wy_blocks.contains_key(&node.outputs[0]),
        )?;
        scope.record(
            kernels::transpose((&mut output).partition([1, 1, value_dim]), &*scratch)
                .generics(vec![value_dim.to_string()]),
        )?;
        arena.buffers[slot] = Some(output.reshape(&original)?);
        Ok(true)
    }
}

const DELTA_GEOMETRY_RANK: usize = 5;

#[expect(
    clippy::too_many_arguments,
    reason = "Explicit graph state, IR inputs and capture geometry"
)]
fn record_delta(
    scope: &Scope,
    node: &TensorNode,
    arena: &ActivationArena,
    weights: &ProgramWeights,
    state: &mut Tensor<f32>,
    scratch: &mut Tensor<f32>,
    prepared: Option<&mut prepared::Inputs>,
    metadata: &Tensor<i32>,
    [key_heads, value_heads, key_dim, lanes, block]: [usize; DELTA_GEOMETRY_RANK],
    wy: bool,
) -> StdResult<(), DeviceError> {
    let qkv = arena.get(node.inputs[0]).map_err(error)?.view(&[
        lanes,
        2 * key_heads + value_heads,
        key_dim,
    ])?;
    let beta = arena
        .get(node.inputs[1])
        .map_err(error)?
        .view(&[lanes, value_heads])?;
    let alpha = arena
        .get(node.inputs[2])
        .map_err(error)?
        .view(&[lanes, value_heads])?;
    let decay = weights
        .constants
        .get(&node.inputs[3])
        .ok_or_else(|| error("chunk delta decay weight"))?;
    let bias = weights
        .constants
        .get(&node.inputs[4])
        .ok_or_else(|| error("chunk delta bias"))?;
    let generics = vec![
        key_heads.to_string(),
        value_heads.to_string(),
        key_dim.to_string(),
        lanes.to_string(),
        block.to_string(),
    ];
    if key_dim == ORDERED_DIM
        && [ORDERED_BLOCK, ORDERED_DIM].contains(&block)
        && ORDERED_WIDTHS.contains(&lanes)
    {
        #[cfg(test)]
        super::ORDERED_CAPTURE_COUNT.set(super::ORDERED_CAPTURE_COUNT.get() + 1);
        prepared
            .ok_or_else(|| error("ordered recurrent preparation"))?
            .record(
                scope,
                [
                    arena.get(node.inputs[0]).map_err(error)?,
                    arena.get(node.inputs[1]).map_err(error)?,
                    arena.get(node.inputs[2]).map_err(error)?,
                    decay,
                    bias,
                ],
                [state, scratch],
                metadata,
                (key_heads, value_heads, key_dim, lanes, block),
            )?;
    } else if wy {
        scope.record(
            kernels::delta_wy(
                state.partition([1, key_dim, block]),
                scratch.partition([1, lanes, block]),
                &qkv,
                &beta,
                &alpha,
                decay,
                bias,
                metadata,
            )
            .generics(generics),
        )?;
    } else {
        scope.record(
            kernels::delta(
                state.partition([1, key_dim, block]),
                scratch.partition([1, lanes, block]),
                &qkv,
                &beta,
                &alpha,
                decay,
                bias,
                metadata,
            )
            .generics(generics),
        )?;
    }
    Ok(())
}
