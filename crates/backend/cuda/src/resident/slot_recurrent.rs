//! Snapshot-free recurrent verification for the pooled speculation graph.
//!
//! The verification lanes of one slot are scored in one launch per recurrent node that reads the
//! slot's state and leaves it untouched; once acceptance is known, one commit launch applies the
//! accepted prefix. Rolling back therefore needs no per-lane state snapshot, which was two full
//! recurrent state copies (about 300 MB on the 27B) per slot.
pub use infer_cuda_kernels::tile::slot_recurrent as fused;

use super::{ActivationArena, ProgramWeights, batch::States, capture::error};
use crate::constants::{CONV_KERNEL_SIZE, CONV_KERNEL_TILE, GDN_PROJECTED_INPUTS};
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::{Error, TensorId};
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Candidate lanes of one slot that the fused kernels cover.
pub(super) const LANES: usize = 3;

/// One slot's verification lanes in the batched arena and their per-lane metadata.
pub(super) struct SlotLanes<'a> {
    pub lanes: [usize; LANES],
    pub metadata: [&'a Tensor<i32>; LANES],
}

/// One node input's candidate rows for a slot.
type StagedInputs = Vec<[Tensor<f32>; LANES]>;

/// Commit views precede their owning batched buffers in drop order. Verification copies
/// each input once for all slots; commits keep the same per-slot/candidate row layout.
#[derive(Default)]
pub(super) struct Staging {
    entries: BTreeMap<(usize, usize), StagedInputs>,
    roots: BTreeMap<(usize, usize), Tensor<f32>>,
}

impl Staging {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn root(&self, index: usize, input: usize) -> Option<&Tensor<f32>> {
        self.roots.get(&(index, input))
    }

    pub(super) fn get(&self, key: &(usize, usize)) -> Option<&StagedInputs> {
        self.entries.get(key)
    }

    /// Preserve every slot's commit input before the arena reuses its allocation.
    pub(super) fn record_node(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &ActivationArena,
        index: usize,
    ) -> Result<(), DeviceError> {
        for (input, &id) in commit_inputs(node).iter().enumerate() {
            let source = arena.get(id).map_err(error)?;
            let target = self
                .roots
                .get_mut(&(index, input))
                .ok_or_else(|| error("batched recurrent staging input"))?;
            if source.size() != target.size() {
                return Err(error("batched recurrent staging geometry"));
            }
            scope.record(api::memcpy(target, source))?;
        }
        Ok(())
    }
}

/// Activations a covered node's commit re-reads; weights are read in place.
pub(super) fn commit_inputs(node: &TensorNode) -> &[TensorId] {
    match node.op {
        TensorOp::Conv { .. } => &node.inputs[..1],
        _ => &node.inputs[..GDN_PROJECTED_INPUTS],
    }
}

/// Allocate the staging rows of every covered node for `slots` slots.
/// # Errors
/// Rejects unbatched activations and failed device allocation.
pub(super) fn allocate_staging(
    device: &CudaDevice,
    nodes: &[TensorNode],
    arena: &ActivationArena,
    (slots, verify): (usize, usize),
) -> infer_core::Result<Staging> {
    let lanes = slots
        .checked_mul(verify)
        .ok_or_else(|| Error::invalid("batched recurrent staging overflow"))?;
    if lanes != arena.lanes() {
        return Err(Error::invalid("batched recurrent staging lanes"));
    }
    let mut staging = Staging::new();
    for (index, node) in nodes.iter().enumerate().filter(|(_, node)| covers(node)) {
        for (input, &id) in commit_inputs(node).iter().enumerate() {
            let mut root = api::zeros::<f32>(&[arena.get(id)?.size()])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            // Per-slot rows feed the three-lane commit graphs; wider pools commit in one batch.
            for slot in (0..slots).filter(|_| verify == LANES) {
                let base = slot * LANES;
                let (first, owner) = staging_row(root, lanes, base)?;
                let (second, owner) = staging_row(owner, lanes, base + 1)?;
                let (third, owner) = staging_row(owner, lanes, base + 2)?;
                root = owner;
                staging
                    .entries
                    .entry((slot, index))
                    .or_default()
                    .push([first, second, third]);
            }
            staging.roots.insert((index, input), root);
        }
    }
    Ok(staging)
}

/// Borrow a row while returning the unique parent for its persistent staging anchor.
fn staging_row(
    root: Tensor<f32>,
    lanes: usize,
    lane: usize,
) -> infer_core::Result<(Tensor<f32>, Tensor<f32>)> {
    let (row, owner) = super::arena::row_tensor(root, lanes, lane)?;
    let root =
        Arc::try_unwrap(owner).map_err(|_| Error::invariant("unique recurrent staging owner"))?;
    Ok((row, root))
}

/// Value-dimension block of the fused kernels: one CTA per value head per split. Every
/// reduction is over the key dimension, so the split changes no summation order.
pub(super) const fn fused_block(value_dim: usize, split: usize) -> usize {
    if split > 1 && value_dim.is_multiple_of(split) {
        value_dim / split
    } else {
        value_dim
    }
}

/// Device facts used by pooled recurrent launch geometry.
#[derive(Debug, Clone, Copy)]
pub(super) struct ValueDevice {
    multiprocessors: u32,
    warp_size: u32,
}
impl ValueDevice {
    pub(super) fn new(device: &CudaDevice) -> infer_core::Result<Self> {
        use cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE as WARP;
        Ok(Self {
            multiprocessors: device.profile()?.multiprocessors,
            warp_size: device.attribute(WARP)?,
        })
    }
}

/// Value partitioning nearest to one SM wave over the pooled head grid.
///
/// Bound each contiguous value row to one warp. Carrying several warps' columns
/// through all key rows increases the fused three-lane update's register pressure,
/// even when the wider tile produces fewer CTAs.
/// Within that bound, choose the legal grid nearest to `SM / (slots * value_heads)`;
/// at a tie, prefer covering the full wave.
pub(super) fn value_split(
    hardware: ValueDevice,
    value_heads: usize,
    value_dim: usize,
    slots: usize,
) -> infer_core::Result<usize> {
    if hardware.multiprocessors == 0
        || !hardware.warp_size.is_power_of_two()
        || value_heads == 0
        || slots == 0
        || !value_dim.is_power_of_two()
    {
        return Err(Error::invalid("recurrent wave geometry"));
    }
    let wave =
        usize::try_from(hardware.multiprocessors).map_err(|e| Error::invalid(e.to_string()))?;
    let warp = usize::try_from(hardware.warp_size).map_err(|e| Error::invalid(e.to_string()))?;
    let minimum = value_dim / value_dim.min(warp);
    let heads = value_heads
        .checked_mul(slots)
        .ok_or_else(|| Error::invalid("recurrent head grid overflow"))?;
    if heads >= wave {
        return Ok(minimum);
    }
    let needed = wave.div_ceil(heads);
    if needed >= value_dim {
        return Ok(value_dim);
    }
    let upper = needed
        .checked_next_power_of_two()
        .ok_or_else(|| Error::invalid("recurrent wave split overflow"))?;
    let lower = upper / 2;
    let upper_grid = heads
        .checked_mul(upper)
        .ok_or_else(|| Error::invalid("recurrent wave grid overflow"))?;
    let lower_grid = heads
        .checked_mul(lower)
        .ok_or_else(|| Error::invalid("recurrent wave grid overflow"))?;
    let selected = if upper_grid.abs_diff(wave) <= lower_grid.abs_diff(wave) {
        upper
    } else {
        lower
    };
    Ok(selected.max(minimum))
}

/// Whether `node` is a recurrent state writer the fused verify/commit pair covers.
pub(super) const fn covers(node: &TensorNode) -> bool {
    node.outputs.len() == 1
        && node.states.len() == 1
        && match node.op {
            TensorOp::Conv { kernel, .. } => kernel == CONV_KERNEL_SIZE,
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                key_dim == value_dim
                    && key_dim.is_power_of_two()
                    && key_heads > 0
                    && value_heads.is_multiple_of(key_heads)
            }
            _ => false,
        }
}

/// A weight constant, or one lane's row of a batched activation.
fn lane_input<'a>(
    weights: &'a ProgramWeights,
    arena: &'a ActivationArena,
    id: TensorId,
    lane: usize,
) -> Result<TensorView<'a, f32>, DeviceError> {
    weights.constants.get(&id).map_or_else(
        || arena.row(id, lane).map_err(error),
        |constant| constant.view(&[constant.size()]).map_err(error),
    )
}

fn rows<'a>(
    weights: &'a ProgramWeights,
    arena: &'a ActivationArena,
    id: TensorId,
    slot: &SlotLanes<'_>,
) -> Result<[TensorView<'a, f32>; LANES], DeviceError> {
    let [first, second, third] = slot.lanes;
    Ok([
        lane_input(weights, arena, id, first)?,
        lane_input(weights, arena, id, second)?,
        lane_input(weights, arena, id, third)?,
    ])
}

/// Record the outputs of one slot's verification lanes for a covered recurrent node, reading
/// but never writing the slot's state.
#[expect(
    unsafe_code,
    reason = "Audited row alias: the three output rows are disjoint lanes of one arena slot, \
              which outlives the captured graph"
)]
pub(super) fn record_verify(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    states: &States,
    slot: &SlotLanes<'_>,
    weights: &ProgramWeights,
    hardware: ValueDevice,
) -> Result<(), DeviceError> {
    let index = arena.slot(node.outputs[0]).map_err(error)?;
    let buffer = arena.buffers[index]
        .take()
        .ok_or_else(|| error("missing slot verify output"))?;
    let lanes = arena.lanes();
    let recorded = (|| {
        let state = states
            .get(&node.states[0])
            .ok_or_else(|| error("slot verify state"))?;
        let [first, second, third] = slot.lanes;
        // SAFETY: see the expectation above.
        let o0 = unsafe { super::arena::row_view(&buffer, lanes, first) }.map_err(error)?;
        // SAFETY: see the expectation above.
        let o1 = unsafe { super::arena::row_view(&buffer, lanes, second) }.map_err(error)?;
        // SAFETY: see the expectation above.
        let o2 = unsafe { super::arena::row_view(&buffer, lanes, third) }.map_err(error)?;
        let [m0, m1, m2] = slot.metadata;
        match node.op {
            TensorOp::Conv { channels, .. } => {
                let [h0, h1, h2] = state.as_slice() else {
                    return Err(error("convolution state layout"));
                };
                let (mut o0, mut o1, mut o2) = (o0, o1, o2);
                let x = rows(weights, arena, node.inputs[0], slot)?;
                let weight = lane_input(weights, arena, node.inputs[1], slot.lanes[0])?;
                let w = weight.view(&[channels, CONV_KERNEL_SIZE])?;
                let tile = CONV_KERNEL_TILE;
                scope.record(fused::conv_verify3(
                    (&mut o0).partition([tile]),
                    (&mut o1).partition([tile]),
                    (&mut o2).partition([tile]),
                    &h0.view(&[h0.size()])?,
                    &h1.view(&[h1.size()])?,
                    &h2.view(&[h2.size()])?,
                    &x[0],
                    &x[1],
                    &x[2],
                    &w,
                    m0,
                    m1,
                    m2,
                ))?;
            }
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                let [state] = state.as_slice() else {
                    return Err(error("delta state layout"));
                };
                let split =
                    value_split(hardware, value_heads, value_dim, lanes / LANES).map_err(error)?;
                let block = fused_block(value_dim, split);
                let shape = [value_heads, 1, value_dim];
                let (mut o0, mut o1, mut o2) = (
                    o0.reshape(&shape)?,
                    o1.reshape(&shape)?,
                    o2.reshape(&shape)?,
                );
                let qkv = rows(weights, arena, node.inputs[0], slot)?;
                let beta = rows(weights, arena, node.inputs[1], slot)?;
                let alpha = rows(weights, arena, node.inputs[2], slot)?;
                scope.record(
                    fused::delta_verify3(
                        (&mut o0).partition([1, 1, block]),
                        (&mut o1).partition([1, 1, block]),
                        (&mut o2).partition([1, 1, block]),
                        &state.view(&[value_heads, key_dim, value_dim])?,
                        &qkv[0],
                        &qkv[1],
                        &qkv[2],
                        &beta[0],
                        &beta[1],
                        &beta[2],
                        &alpha[0],
                        &alpha[1],
                        &alpha[2],
                        &lane_input(weights, arena, node.inputs[3], slot.lanes[0])?,
                        &lane_input(weights, arena, node.inputs[4], slot.lanes[0])?,
                        m0,
                        m1,
                        m2,
                    )
                    .generics(vec![
                        key_heads.to_string(),
                        value_heads.to_string(),
                        key_dim.to_string(),
                        block.to_string(),
                    ]),
                )?;
            }
            _ => return Err(error("slot verify recurrent node")),
        }
        Ok(())
    })();
    arena.buffers[index] = Some(buffer);
    recorded
}

/// Record the commit of one slot's first `count` verification lanes for a covered node.
#[expect(
    clippy::too_many_arguments,
    reason = "The captured commit node binds scope, geometry, state, lanes and staging explicitly"
)]
pub(super) fn record_commit(
    scope: &Scope,
    node: &TensorNode,
    arena: &ActivationArena,
    states: &mut States,
    slot: &SlotLanes<'_>,
    weights: &ProgramWeights,
    hardware: ValueDevice,
    (staged, count): (&[[Tensor<f32>; LANES]], usize),
) -> Result<(), DeviceError> {
    let count = i32::try_from(count).map_err(error)?;
    let state = states
        .get_mut(&node.states[0])
        .ok_or_else(|| error("slot commit state"))?;
    let [m0, m1, m2] = slot.metadata;
    match node.op {
        TensorOp::Conv { .. } => {
            let [h0, h1, h2] = state.as_mut_slice() else {
                return Err(error("convolution state layout"));
            };
            let [x] = staged else {
                return Err(error("convolution staging layout"));
            };
            let tile = CONV_KERNEL_TILE;
            scope.record(fused::conv_commit3(
                h0.partition([tile]),
                h1.partition([tile]),
                h2.partition([tile]),
                &x[0],
                &x[1],
                &x[2],
                m0,
                m1,
                m2,
                count,
            ))?;
        }
        TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } => {
            let [state] = state.as_mut_slice() else {
                return Err(error("delta state layout"));
            };
            let split = value_split(hardware, value_heads, value_dim, arena.lanes() / LANES)
                .map_err(error)?;
            let block = fused_block(value_dim, split);
            let [qkv, beta, alpha] = staged else {
                return Err(error("delta staging layout"));
            };
            scope.record(
                fused::delta_commit3(
                    state.partition([1, key_dim, block]),
                    &qkv[0],
                    &qkv[1],
                    &qkv[2],
                    &beta[0],
                    &beta[1],
                    &beta[2],
                    &alpha[0],
                    &alpha[1],
                    &alpha[2],
                    &lane_input(weights, arena, node.inputs[3], slot.lanes[0])?,
                    &lane_input(weights, arena, node.inputs[4], slot.lanes[0])?,
                    m0,
                    m1,
                    m2,
                    count,
                )
                .generics(vec![
                    key_heads.to_string(),
                    value_heads.to_string(),
                    key_dim.to_string(),
                    block.to_string(),
                ]),
            )?;
        }
        _ => return Err(error("slot commit recurrent node")),
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_recurrent.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/slot_staging.rs"]
mod staging_tests;
