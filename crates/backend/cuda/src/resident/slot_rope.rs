//! One rotary launch across verification rows, using each row's original position.
use super::{ActivationArena, buffers::buffer, capture::error, program::ProgramWeights};
use crate::constants::METADATA_FIELDS;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{Recording, RopeBuffers, RopeMode, RopePlan};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{TensorNode, TensorOp};

pub(super) fn record(
    scope: &Recording<'_>,
    node: &TensorNode,
    arena: &ActivationArena,
    weights: &ProgramWeights,
    metadata: &Allocation<i32>,
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
        || metadata.len() != arena.lanes() * METADATA_FIELDS
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
    let output = arena
        .buffer(slot)
        .and_then(|output| output.require("pooled rotary output", arena.lanes() * heads * head_dim))
        .map_err(error)?;
    let input = arena.get(node.inputs[0]).map_err(error)?;
    let plan = RopePlan {
        mode: RopeMode::Pooled {
            fields: METADATA_FIELDS,
        },
        rows: arena.lanes() * heads,
        heads,
        head_dim,
        half,
    };
    weights
        .kernels
        .rope
        .rope(
            scope,
            &plan,
            RopeBuffers {
                out: output,
                x: buffer(&input)?,
                frequencies: buffer(frequencies)?,
                axes: None,
                metadata: buffer(metadata)?,
            },
        )
        .map_err(error)?;
    Ok(true)
}

#[cfg(test)]
use infer_cuda_kernels::tile::rope::slot_rope as kernels;

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_rope.rs"]
mod tests;
