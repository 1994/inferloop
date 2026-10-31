//! Parallel causal convolution across a prompt chunk, followed by one state commit.
pub use infer_cuda_kernels::tile::recurrent::conv_prefill as kernels;

use super::{ActivationArena, ProgramWeights, batch::States, capture::error};
use crate::constants::CONV_KERNEL_SIZE;
use crate::constants::CONV_KERNEL_TILE;
use cutile::prelude::*;
use infer_ir::{TensorNode, TensorOp};

/// Rows per block of the row-sliding convolution kernel (fixed in the kernel).
const CONV_ROWS: usize = 16;

pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    states: &mut States,
    metadata: &Tensor<i32>,
    lanes: usize,
) -> Result<bool, DeviceError> {
    #[cfg(test)]
    if super::recurrent_prefill::LEGACY_CAPTURE.get() {
        return Ok(false);
    }
    let TensorOp::Conv {
        channels,
        kernel: CONV_KERNEL_SIZE,
    } = node.op
    else {
        return Ok(false);
    };
    if weights.constants.contains_key(&node.inputs[0]) {
        return Ok(false);
    }
    let state = states
        .get_mut(&node.states[0])
        .ok_or_else(|| error("prefill convolution state"))?;
    let [h0, h1, h2] = state.as_mut_slice() else {
        return Err(error("prefill convolution layout"));
    };
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prefill convolution output"))?;
    let original: Vec<usize> = output
        .shape()
        .iter()
        .map(|&n| usize::try_from(n).map_err(error))
        .collect::<Result<_, _>>()?;
    let mut output = output.reshape(&[lanes, channels])?;
    let input = arena
        .get(node.inputs[0])
        .map_err(error)?
        .view(&[lanes, channels])
        .map_err(error)?;
    let weight = weights
        .constants
        .get(&node.inputs[1])
        .ok_or_else(|| error("prefill convolution weight"))?
        .view(&[channels, CONV_KERNEL_SIZE])
        .map_err(error)?;
    if lanes.is_multiple_of(CONV_ROWS) {
        scope.record(kernels::forward_rows(
            (&mut output).partition([CONV_ROWS, CONV_KERNEL_TILE]),
            &*h0,
            &*h1,
            &*h2,
            &input,
            &weight,
            metadata,
        ))?;
    } else {
        scope.record(kernels::forward(
            (&mut output).partition([1, CONV_KERNEL_TILE]),
            &*h0,
            &*h1,
            &*h2,
            &input,
            &weight,
            metadata,
        ))?;
    }
    scope.record(kernels::commit(
        h0.partition([CONV_KERNEL_TILE]),
        h1.partition([CONV_KERNEL_TILE]),
        h2.partition([CONV_KERNEL_TILE]),
        &input,
        metadata,
    ))?;
    arena.buffers[slot] = Some(output.reshape(&original)?);
    Ok(true)
}

#[cfg(test)]
#[path = "../../tests/unit/resident_conv_prefill.rs"]
mod tests;
