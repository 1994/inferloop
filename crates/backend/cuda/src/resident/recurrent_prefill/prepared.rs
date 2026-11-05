//! Per-token/head F32 normalization and gating shared across independent state columns.
use super::super::buffers::Operand;
use super::super::{buffers::buffer, capture::error};
use super::{ORDERED_DIM, ORDERED_WIDTHS, chunk, oxide};
use crate::constants::F32_BYTES;
use crate::constants::GDN_PROJECTED_INPUTS;
use crate::{device::CudaDevice, resident::ProgramWeights};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    Buffer, GateBuffers, PreparedDeltaBuffers, Recording, Recurrent, RecurrentShape,
};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorOp};
use std::collections::{BTreeMap, BTreeSet};
use std::result::Result as StdResult;

const INPUT_BINDINGS: usize = 5;
type Geometry = (usize, usize, usize);
pub(super) type Buffers = BTreeMap<Geometry, Inputs>;

pub(super) struct Inputs {
    normalized: Allocation<f32>,
    betas: Allocation<f32>,
    decays: Allocation<f32>,
    /// The pinned SIMT build records the recurrence; prepared on the program's families.
    simt: bool,
    chunk: Option<chunk::Scratch>,
}

fn geometries(graph: &DataflowGraph, weights: &ProgramWeights, lanes: usize) -> BTreeSet<Geometry> {
    let mut shapes = BTreeSet::new();
    if !ORDERED_WIDTHS.contains(&lanes) {
        return shapes;
    }
    for node in &graph.nodes {
        if let TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = node.op
            && key_dim == ORDERED_DIM
            && value_dim == key_dim
            && key_heads > 0
            && value_heads > 0
            && value_heads.is_multiple_of(key_heads)
            && node.inputs[..GDN_PROJECTED_INPUTS]
                .iter()
                .all(|id| !weights.constants.contains_key(id))
        {
            shapes.insert((key_heads, value_heads, key_dim));
        }
    }
    shapes
}

pub(super) fn allocate(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    lanes: usize,
) -> Result<Buffers> {
    geometries(graph, weights, lanes)
        .into_iter()
        .map(|shape| {
            let mut inputs = Inputs::new(device, shape, lanes)?;
            inputs.simt = oxide::select(
                device,
                weights.kernels.recurrent.as_ref(),
                RecurrentShape {
                    key_heads: shape.0,
                    value_heads: shape.1,
                    dim: shape.2,
                    lanes,
                },
                weights.autotune,
                &weights.resident_tuning,
            )?;
            Ok((shape, inputs))
        })
        .collect()
}

pub(super) fn required_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    lanes: usize,
) -> Result<u64> {
    geometries(graph, weights, lanes)
        .into_iter()
        .try_fold(0u64, |total, (kh, vh, dim)| {
            let chunked = chunk::enabled(dim, lanes);
            let extra = if chunked {
                chunk::Scratch::bytes((kh, vh, dim), lanes)
            } else {
                0
            };
            let normalized_rows = if chunked { 1 } else { lanes };
            let bytes = kh
                .checked_mul(2)
                .and_then(|x| x.checked_mul(dim))
                .and_then(|x| x.checked_mul(normalized_rows))
                .and_then(|x| {
                    vh.checked_mul(2 * lanes)
                        .and_then(|gates| x.checked_add(gates))
                })
                .and_then(|x| x.checked_mul(F32_BYTES))
                .and_then(|x| x.checked_add(extra))
                .and_then(|x| u64::try_from(x).ok())
                .ok_or_else(|| Error::invalid("CUDA recurrent preparation size overflow"))?;
            total
                .checked_add(bytes)
                .ok_or_else(|| Error::invalid("CUDA recurrent preparation size overflow"))
        })
}

impl Inputs {
    pub(super) fn new(device: &CudaDevice, (kh, vh, dim): Geometry, lanes: usize) -> Result<Self> {
        let chunked = chunk::enabled(dim, lanes);
        // The chunked rule normalizes into its own BF16 rows; the F32 rows only feed the
        // token-serial kernels.
        let normalized_rows = if chunked { 1 } else { lanes };
        let runtime = device.runtime();
        Ok(Self {
            simt: false,
            chunk: chunked
                .then(|| chunk::Scratch::new(device, (kh, vh, dim), lanes))
                .transpose()?,
            normalized: runtime.zeros::<f32>(normalized_rows * 2 * kh * dim)?,
            betas: runtime.zeros::<f32>(lanes * vh)?,
            decays: runtime.zeros::<f32>(lanes * vh)?,
        })
    }

    pub(super) fn record<S: Operand<Marker = f32>, O: Operand<Marker = f32>>(
        &self,
        scope: &Scope,
        recurrent: &dyn Recurrent,
        [qkv_storage, beta, alpha, a_log, bias]: [Buffer<'_, f32>; INPUT_BINDINGS],
        (state, output): (&S, &O),
        metadata: &Allocation<i32>,
        (kh, vh, dim, lanes, block): (usize, usize, usize, usize, usize),
    ) -> StdResult<(), DeviceError> {
        let recording = Recording::scope(scope);
        let shape = RecurrentShape {
            key_heads: kh,
            value_heads: vh,
            dim,
            lanes,
        };
        if self.chunk.is_none() {
            recurrent
                .normalize(&recording, shape, buffer(&self.normalized)?, qkv_storage)
                .map_err(error)?;
        }
        recurrent
            .gates(
                &recording,
                shape,
                GateBuffers {
                    betas: buffer(&self.betas)?,
                    decays: buffer(&self.decays)?,
                    beta,
                    alpha,
                    decay: a_log,
                    bias,
                },
            )
            .map_err(error)?;
        if let Some(scratch) = &self.chunk {
            return chunk::record(
                scope,
                recurrent,
                scratch,
                shape,
                [
                    qkv_storage,
                    buffer(&self.betas)?,
                    buffer(&self.decays)?,
                    buffer(state)?,
                    buffer(output)?,
                ],
                buffer(metadata)?,
            );
        }
        if self.simt {
            #[cfg(test)]
            oxide::recorded();
            return recurrent
                .delta_prepared_simt(
                    &recording,
                    shape,
                    PreparedDeltaBuffers {
                        state: buffer(state)?,
                        output: buffer(output)?,
                        qkv: qkv_storage,
                        normalized: buffer(&self.normalized)?,
                        betas: buffer(&self.betas)?,
                        decays: buffer(&self.decays)?,
                        metadata: buffer(metadata)?,
                    },
                )
                .map_err(error);
        }
        recurrent
            .delta_prepared(
                &recording,
                shape,
                block,
                PreparedDeltaBuffers {
                    state: buffer(state)?,
                    output: buffer(output)?,
                    qkv: qkv_storage,
                    normalized: buffer(&self.normalized)?,
                    betas: buffer(&self.betas)?,
                    decays: buffer(&self.decays)?,
                    metadata: buffer(metadata)?,
                },
            )
            .map_err(error)?;
        Ok(())
    }
}
