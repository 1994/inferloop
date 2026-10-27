//! Contiguous pool state families with the same per-slot views used by homes and fallback graphs.
use super::slot_recurrent::{LANES, ValueDevice, covers, fused, fused_block, value_split};
use super::{ActivationArena, capture::error};
use super::{batch::States, program::ProgramWeights};
use crate::constants::METADATA_FIELDS;
use crate::{
    constants::F32_BYTES,
    device::{CudaDevice, device_error},
};
use cutile::prelude::*;
use infer_core::{Error, OpId, Result};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::result::Result as StdResult;

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
                let mut root = api::zeros::<f32>(&[width * size])
                    .sync_on(&device.stream)
                    .map_err(device_error)?;
                for (slot, lane) in result.lanes.iter_mut().enumerate() {
                    let (row, owner) = super::arena::row_tensor(root, width, slot)?;
                    root = Arc::try_unwrap(owner)
                        .map_err(|_| Error::invariant("unique pooled state owner"))?;
                    lane.entry(id)
                        .or_default()
                        .push(row.reshape(&shape).map_err(device_error)?);
                }
                let root_shape = if shape.len() == RECURRENT_STATE_RANK {
                    vec![width, shape[0] * shape[1], shape[2]]
                } else {
                    vec![width, size]
                };
                owners.push(root.reshape(&root_shape).map_err(device_error)?);
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
        fused::pack_parameters(
            (&mut parameters).partition([2, 1]),
            &rate.view(&[value_heads]).map_err(device_error)?,
            &bias.view(&[value_heads]).map_err(device_error)?,
        )
        .generics(vec![value_heads.to_string()])
        .sync_on(&device.stream)
        .map_err(device_error)?;
        result.insert(node.id, parameters);
    }
    Ok(result)
}

/// Use the slot axis when its immutable constants and contiguous state family are available.
pub(super) fn record_batch(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
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
    let output = arena.buffers[index]
        .take()
        .ok_or_else(|| error("pooled delta output"))?;
    let output_size = output.size();
    let mut output = output.reshape(&[slots, lanes, value_heads * value_dim])?;
    let recorded = (|| {
        let state = state.view(&[slots * value_heads, key_dim, value_dim])?;
        let parameters = parameters.view(&[parameters.size()])?;
        let mut generics = vec![
            key_heads.to_string(),
            value_heads.to_string(),
            key_dim.to_string(),
            block.to_string(),
            METADATA_FIELDS.to_string(),
        ];
        if lanes == LANES {
            scope.record(
                fused::delta_verify_batch(
                    (&mut output).partition([1, LANES, block]),
                    &state,
                    arena.get(node.inputs[0]).map_err(error)?,
                    arena.get(node.inputs[1]).map_err(error)?,
                    arena.get(node.inputs[2]).map_err(error)?,
                    &parameters,
                    batch.metadata,
                )
                .generics(generics),
            )?;
        } else {
            generics.push(lanes.to_string());
            scope.record(
                fused::delta_verify_lanes(
                    (&mut output).partition([1, lanes, block]),
                    &state,
                    arena.get(node.inputs[0]).map_err(error)?,
                    arena.get(node.inputs[1]).map_err(error)?,
                    arena.get(node.inputs[2]).map_err(error)?,
                    &parameters,
                    batch.metadata,
                )
                .generics(generics),
            )?;
        }
        Ok(true)
    })();
    arena.buffers[index] = Some(output.reshape(&[output_size])?);
    recorded
}
