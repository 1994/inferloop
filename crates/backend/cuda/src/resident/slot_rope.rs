//! One rotary launch across verification rows, using each row's original position.
use super::{ActivationArena, capture::error, program::ProgramWeights};
use crate::constants::METADATA_FIELDS;
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    metadata: &Tensor<i32>,
) -> Result<bool, DeviceError> {
    let TensorOp::Rope {
        heads,
        head_dim,
        rotary_dim,
        ..
    } = node.op
    else {
        return Ok(false);
    };
    if weights.constants.contains_key(&node.inputs[0])
        || metadata.size() != arena.lanes() * METADATA_FIELDS
    {
        return Ok(false);
    }
    let half = rotary_dim / 2;
    if heads == 0 || !half.is_power_of_two() || !head_dim.is_multiple_of(half) {
        return Err(error("unsupported pooled rotary dimensions"));
    }
    let frequencies = weights
        .rope_frequencies
        .get(&node.id)
        .ok_or_else(|| error("missing pooled rotary frequencies"))?;
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("pooled rotary output"))?;
    let restore = output
        .shape()
        .iter()
        .map(|&dim| usize::try_from(dim).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = output.reshape(&[arena.lanes() * heads, head_dim])?;
    scope.record(
        kernels::rope(
            (&mut output).partition([1, half]),
            &arena
                .get(node.inputs[0])
                .map_err(error)?
                .view(&[arena.lanes() * heads, head_dim])?,
            frequencies,
            metadata,
        )
        .generics(
            [head_dim, half, heads, METADATA_FIELDS]
                .iter()
                .map(ToString::to_string)
                .collect(),
        ),
    )?;
    arena.buffers[slot] = Some(output.reshape(&restore)?);
    Ok(true)
}

pub use infer_cuda_kernels::tile::rope::slot_rope as kernels;

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_rope.rs"]
mod tests;
