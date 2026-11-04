//! Contiguous pool state families with the same per-slot views used by homes and fallback graphs.
use super::slot_recurrent::{LANES, ValueDevice, covers, fused_block, value_split};
use super::{ActivationArena, buffers::buffer, capture::error};
use super::{batch::States, program::ProgramWeights};
use crate::constants::METADATA_FIELDS;
use crate::{
    constants::F32_BYTES,
    device::{CudaDevice, device_error},
};
use cutile::prelude::*;
use infer_core::{Error, OpId, Result};
use infer_cuda_kernels::contract::{Kernels, PoolDeltaBuffers, PoolDeltaPlan, Recording};
use infer_cuda_kernels::tile::recurrent::pack_parameters;
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::result::Result as StdResult;
use std::sync::Arc;

const RECURRENT_STATE_RANK: usize = 3;

/// Views drop before their owning allocations; the pool drops its captures before this object.
pub(super) struct SlotStates {
    lanes: Vec<States>,
    pub(super) roots: States,
}

impl SlotStates {
    pub(super) fn allocate(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, usize),
        budget: u64,
    ) -> Result<Self> {
        let (width, capacity) = geometry;
        let plan = super::program::state_shapes(graph, capacity, weights)?;
        let mut bytes = 0_u64;
        for (_, shapes) in &plan {
            for shape in shapes {
                let elements = shape
                    .iter()
                    .try_fold(width, |n, &dim| n.checked_mul(dim))
                    .ok_or_else(|| Error::invalid("pooled state overflow"))?;
                bytes = bytes
                    .checked_add(
                        (elements as u64)
                            .checked_mul(F32_BYTES as u64)
                            .ok_or_else(|| Error::invalid("pooled state byte overflow"))?,
                    )
                    .ok_or_else(|| Error::invalid("pooled state budget overflow"))?;
            }
        }
        if bytes > budget {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "pooled state exceeds available budget",
            ));
        }
        let mut result = Self {
            lanes: (0..width).map(|_| States::new()).collect(),
            roots: States::new(),
        };
        for (id, shapes) in plan {
            let mut owners = Vec::new();
            for shape in shapes {
                let size = shape.iter().product::<usize>();
                let memory = Allocation::adopt(Arc::new(
                    api::zeros::<f32>(&[width * size])
                        .sync_on(&device.stream)
                        .map_err(device_error)?,
                ))?;
                let rows = super::arena::rows(&memory, width)?;
                for (row, lane) in rows.into_iter().zip(result.lanes.iter_mut()) {
                    lane.entry(id)
                        .or_default()
                        .push(row.reshape(&shape).map_err(device_error)?);
                }
                let root_shape = if shape.len() == RECURRENT_STATE_RANK {
                    vec![width, shape[0] * shape[1], shape[2]]
                } else {
                    vec![width, size]
                };
                owners.push(super::arena::whole(&memory, &root_shape)?);
            }
            result.roots.insert(id, owners);
        }
        Ok(result)
    }

    pub(super) fn parts(&mut self) -> (&mut [States], &mut States) {
        (&mut self.lanes, &mut self.roots)
    }

    pub(super) const fn roots(&self) -> &States {
        &self.roots
    }
}

impl Deref for SlotStates {
    type Target = [States];
    fn deref(&self) -> &Self::Target {
        &self.lanes
    }
}
impl DerefMut for SlotStates {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.lanes
    }
}

pub(super) type RecurrentParameters = BTreeMap<OpId, Tensor<f32>>;

/// Full owners used only while constructing the shared verification launch.
pub(super) struct VerifyBatch<'a> {
    pub kernels: &'a Kernels,
    pub roots: &'a States,
    pub parameters: &'a RecurrentParameters,
    pub metadata: &'a Tensor<i32>,
    /// Candidate lanes per slot.
    pub lanes: usize,
}

const DELTA_PARAMETER_VECTORS: u64 = 2;
const PARAMETER_ALIGNMENT_BYTES: u64 = 256;

/// Conservative shared-parameter quote, including allocator alignment for every delta node.
pub(super) fn parameter_bytes(nodes: &[TensorNode]) -> Result<u64> {
    let counts = if nodes.iter().any(covers) {
        PARAMETER_ALIGNMENT_BYTES
    } else {
        0
    };
    nodes
        .iter()
        .filter(|node| covers(node))
        .try_fold(counts, |sum, node| {
            let TensorOp::Delta { value_heads, .. } = node.op else {
                return Ok(sum);
            };
            let bytes = (value_heads as u64)
                .checked_mul(DELTA_PARAMETER_VECTORS * F32_BYTES as u64)
                .and_then(|n| n.checked_add(PARAMETER_ALIGNMENT_BYTES - 1))
                .map(|n| n / PARAMETER_ALIGNMENT_BYTES * PARAMETER_ALIGNMENT_BYTES)
                .ok_or_else(|| Error::invalid("pooled parameter byte overflow"))?;
            sum.checked_add(bytes)
                .ok_or_else(|| Error::invalid("pooled parameter budget overflow"))
        })
}

/// Prepack immutable constants without changing their bits or allocating per replay.
pub(super) fn prepare_parameters(
    device: &CudaDevice,
    nodes: &[TensorNode],
    weights: &ProgramWeights,
) -> Result<RecurrentParameters> {
    let mut result = RecurrentParameters::new();
    for node in nodes.iter().filter(|node| covers(node)) {
        let TensorOp::Delta { value_heads, .. } = node.op else {
            continue;
        };
        let (Some(rate), Some(bias)) = (
            weights.constants.get(&node.inputs[3]),
            weights.constants.get(&node.inputs[4]),
        ) else {
            continue;
        };
        if rate.size() != value_heads || bias.size() != value_heads {
            return Err(Error::invalid("pooled delta parameter geometry"));
        }
        let mut parameters = api::zeros::<f32>(&[2, value_heads])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        pack_parameters(&device.stream, &mut parameters, rate, bias)?;
        result.insert(node.id, parameters);
    }
    Ok(result)
}

/// Use the slot axis when its immutable constants and contiguous state family are available.
pub(super) fn record_batch(
    scope: &Scope,
    node: &TensorNode,
    arena: &ActivationArena,
    batch: &VerifyBatch<'_>,
    hardware: ValueDevice,
) -> StdResult<bool, DeviceError> {
    let TensorOp::Delta {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    } = node.op
    else {
        return Ok(false);
    };
    let Some(parameters) = batch.parameters.get(&node.id) else {
        return Ok(false);
    };
    let roots = batch
        .roots
        .get(&node.states[0])
        .ok_or_else(|| error("pooled delta owner"))?;
    let [state] = roots.as_slice() else {
        return Err(error("pooled delta state layout"));
    };
    let lanes = batch.lanes;
    let slots = arena.lanes() / lanes;
    let split = value_split(hardware, value_heads, value_dim, slots).map_err(error)?;
    let block = fused_block(value_dim, split);
    let index = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena
        .buffer(index)
        .and_then(|output| {
            output.require(
                "pooled delta output",
                slots * lanes * value_heads * value_dim,
            )
        })
        .map_err(error)?;
    batch
        .kernels
        .recurrent_pool
        .delta_verify_pool(
            &Recording::scope(scope),
            &PoolDeltaPlan {
                key_heads,
                value_heads,
                dim: key_dim,
                block,
                slots,
                lanes,
                fields: METADATA_FIELDS,
                fused: lanes == LANES,
            },
            output,
            PoolDeltaBuffers {
                state: buffer(state)?,
                qkv: buffer(arena.get(node.inputs[0]).map_err(error)?)?,
                beta: buffer(arena.get(node.inputs[1]).map_err(error)?)?,
                alpha: buffer(arena.get(node.inputs[2]).map_err(error)?)?,
                parameters: buffer(parameters)?,
                metadata: buffer(batch.metadata)?,
            },
        )
        .map_err(error)?;
    Ok(true)
}
