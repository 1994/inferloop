//! Parallel causal convolution across a prompt chunk, followed by one state commit.
#[cfg(test)]
pub use infer_cuda_kernels::tile::recurrent::conv_prefill as kernels;

use super::{ActivationArena, ProgramWeights, batch::States, buffers::buffer, capture::error};
use crate::constants::CONV_KERNEL_SIZE;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{ConvBuffers, ConvKernel, Recording};
use infer_ir::{TensorNode, TensorOp};

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
    let output = output.reshape(&[lanes, channels])?;
    weights
        .kernels
        .recurrent
        .conv(
            &Recording::scope(scope),
            ConvKernel::Prefill { lanes },
            channels,
            ConvBuffers {
                output: buffer(&output)?,
                history: [buffer(&*h0)?, buffer(&*h1)?, buffer(&*h2)?],
                input: buffer(arena.get(node.inputs[0]).map_err(error)?)?,
                weight: buffer(
                    weights
                        .constants
                        .get(&node.inputs[1])
                        .ok_or_else(|| error("prefill convolution weight"))?
                        .as_ref(),
                )?,
                metadata: buffer(metadata)?,
            },
        )
        .map_err(error)?;
    arena.buffers[slot] = Some(output.reshape(&original)?);
    Ok(true)
}

#[cfg(test)]
#[path = "../../tests/unit/resident_conv_prefill.rs"]
mod tests;
