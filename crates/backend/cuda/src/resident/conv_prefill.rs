//! Parallel causal convolution across a prompt chunk, followed by one state commit.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::recurrent::conv_prefill as kernels;

use super::{ActivationArena, ProgramWeights, batch::States, buffers::buffer, capture::error};
use infer_core::Result;
use infer_cuda_kernels::contract::{ConvBuffers, ConvKernel, Recording};
use infer_cuda_kernels::runtime::Allocation;
use infer_ir::TensorNode;

pub(super) fn record(
    scope: &Recording<'_>,
    node: &TensorNode,
    arena: &ActivationArena,
    weights: &ProgramWeights,
    states: &mut States,
    metadata: &Allocation<i32>,
    (lanes, channels): (usize, usize),
) -> Result<()> {
    let state = states
        .get_mut(&node.states[0])
        .ok_or_else(|| error("prefill convolution state"))?;
    let [h0, h1, h2] = state.as_mut_slice() else {
        return Err(error("prefill convolution layout"));
    };
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena
        .buffer(slot)
        .and_then(|output| output.require("prefill convolution output", lanes * channels))
        .map_err(error)?;
    weights
        .kernels
        .recurrent
        .conv(
            scope,
            ConvKernel::Prefill { lanes },
            channels,
            ConvBuffers {
                output,
                history: [buffer(&*h0)?, buffer(&*h1)?, buffer(&*h2)?],
                input: arena.activation(node.inputs[0]).map_err(error)?,
                weight: buffer(
                    weights
                        .constants
                        .get(&node.inputs[1])
                        .ok_or_else(|| error("prefill convolution weight"))?,
                )?,
                metadata: buffer(metadata)?,
            },
        )
        .map_err(error)
}

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_conv_prefill.rs"]
mod tests;
