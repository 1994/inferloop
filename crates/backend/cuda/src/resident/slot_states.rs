//! Contiguous pool state families with the same per-slot views used by homes and fallback graphs.
use super::shaped::Shaped;
use super::slot_recurrent::{LANES, ValueDevice, covers, fused_block, value_split};
use super::{ActivationArena, buffers::buffer, capture::error};
use super::{batch::States, program::ProgramWeights};
use crate::constants::METADATA_FIELDS;
use crate::{constants::F32_BYTES, device::CudaDevice};
use infer_core::{Error, OpId, Result, TensorId};
use infer_cuda_kernels::contract::{Kernels, PoolDeltaBuffers, PoolDeltaPlan, Recording};
use infer_cuda_kernels::ops::pack_parameters;
use infer_cuda_kernels::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

const RECURRENT_STATE_RANK: usize = 3;

/// Every view of a family holds its allocation, so views stay valid on their own.
pub(super) struct SlotStates {
    lanes: Vec<States>,
    pub(super) roots: States,
    families: BTreeMap<TensorId, Vec<Family>>,
}

/// One pooled state tensor: `width` concatenated lanes of `lane` shape, viewed whole as `root`.
struct Family {
    memory: Allocation<f32>,
    lane: Vec<usize>,
    root: Vec<usize>,
}

impl Family {
    fn lane_elements(&self) -> usize {
        self.lane.iter().product()
    }
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
            families: BTreeMap::new(),
        };
        for (id, shapes) in plan {
            let mut owners = Vec::new();
            let mut families = Vec::new();
            for shape in shapes {
                let size = shape.iter().product::<usize>();
                let memory = device.tile().zeros::<f32>(width * size)?;
                let rows = memory.rows(width)?;
                for (row, lane) in rows.into_iter().zip(result.lanes.iter_mut()) {
                    lane.entry(id)
                        .or_default()
                        .push(Shaped::new(row, shape.clone())?);
                }
                let root_shape = if shape.len() == RECURRENT_STATE_RANK {
                    vec![width, shape[0] * shape[1], shape[2]]
                } else {
                    vec![width, size]
                };
                owners.push(Shaped::new(memory.clone(), root_shape.clone())?);
                families.push(Family {
                    memory,
                    lane: shape,
                    root: root_shape,
                });
            }
            result.roots.insert(id, owners);
            result.families.insert(id, families);
        }
        Ok(result)
    }

    pub(super) fn parts(&mut self) -> (&mut [States], &mut States) {
        (&mut self.lanes, &mut self.roots)
    }

    /// The first `slots` lanes of every root family, for a graph narrower than the pool.
    /// # Errors
    /// Rejects a slot count outside `1..=width`.
    pub(super) fn prefix(&self, slots: usize) -> Result<States> {
        if slots == 0 || slots > self.lanes.len() {
            return Err(Error::invalid("pooled state prefix"));
        }
        let mut roots = States::new();
        for (id, families) in &self.families {
            let mut views = Vec::with_capacity(families.len());
            for family in families {
                let mut shape = family.root.clone();
                shape[0] = slots;
                views.push(Shaped::new(
                    family.memory.slice(0..slots * family.lane_elements())?,
                    shape,
                )?);
            }
            roots.insert(*id, views);
        }
        Ok(roots)
    }

    /// Another view of lane `slot` of every family, shaped as the lane's own states.
    /// # Errors
    /// Rejects a slot past the pool.
    pub(super) fn lane(&self, slot: usize) -> Result<States> {
        if slot >= self.lanes.len() {
            return Err(Error::invalid("slot view index"));
        }
        let mut lane = States::new();
        for (id, families) in &self.families {
            let mut views = Vec::with_capacity(families.len());
            for family in families {
                let size = family.lane_elements();
                views.push(Shaped::new(
                    family.memory.slice(slot * size..(slot + 1) * size)?,
                    family.lane.clone(),
                )?);
            }
            lane.insert(*id, views);
        }
        Ok(lane)
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

pub(super) type RecurrentParameters = BTreeMap<OpId, Allocation<f32>>;

/// Full owners used only while constructing the shared verification launch.
pub(super) struct VerifyBatch<'a> {
    pub kernels: &'a Kernels,
    pub roots: &'a States,
    pub parameters: &'a RecurrentParameters,
    pub metadata: &'a Allocation<i32>,
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
        if rate.len() != value_heads || bias.len() != value_heads {
            return Err(Error::invalid("pooled delta parameter geometry"));
        }
        let parameters = device.runtime().zeros(2 * value_heads)?;
        pack_parameters(
            &device.runtime(),
            parameters.operand(),
            rate.operand(),
            bias.operand(),
        )?;
        result.insert(node.id, parameters);
    }
    Ok(result)
}

/// Use the slot axis when its immutable constants and contiguous state family are available.
pub(super) fn record_batch(
    scope: &Recording<'_>,
    node: &TensorNode,
    arena: &ActivationArena,
    batch: &VerifyBatch<'_>,
    hardware: ValueDevice,
) -> Result<bool> {
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
            scope,
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
                state: buffer(state)?.into(),
                qkv: arena.activation(node.inputs[0]).map_err(error)?,
                beta: arena.activation(node.inputs[1]).map_err(error)?,
                alpha: arena.activation(node.inputs[2]).map_err(error)?,
                parameters: buffer(parameters)?,
                metadata: buffer(batch.metadata)?,
            },
        )
        .map_err(error)?;
    Ok(true)
}
