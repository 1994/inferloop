//! Prompt graphs project only the last active lane onto the vocabulary.
//!
//! A prompt chunk reads back one row of logits, so its arena keeps one row (see
//! [`super::ActivationArena::new_prompt`]) instead of `lanes x vocabulary`, which is most of a
//! wide prompt arena (2 GB at 2048 lanes for a 248K vocabulary).
use super::{ActivationArena, ProgramWeights, capture::error};
use cutile::prelude::*;
use infer_ir::TensorNode;

pub use infer_cuda_kernels::tile::prompt_logits as kernels;

/// Columns one `last_row` block copies.
const ROW_TILE: usize = 256;

/// Whether a prompt graph of these weights can keep a one-row logits slot: the vocabulary
/// projection must take the quantized single-row GEMM, and its input row must tile evenly.
pub fn supported(graph: &infer_ir::DataflowGraph, weights: &ProgramWeights) -> bool {
    let Some(logits) = graph.logits else {
        return false;
    };
    graph
        .nodes
        .iter()
        .find(|node| node.outputs.first() == Some(&logits))
        .is_some_and(|node| {
            node.op == infer_ir::TensorOp::Linear
                && weights.activation_quantization(node.inputs[1]).is_some()
                && graph
                    .tensors
                    .iter()
                    .find(|tensor| tensor.id == node.inputs[0])
                    .and_then(|tensor| tensor.elements().ok())
                    .is_some_and(|hidden| hidden.is_multiple_of(ROW_TILE))
        })
}

/// Prompt arena bytes at `lanes` rows: one logits row when [`supported`], every row otherwise.
/// # Errors
/// Rejects invalid graphs or overflowing sizes.
pub fn required_bytes(
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
    lanes: usize,
) -> infer_core::Result<usize> {
    if supported(graph, weights) {
        ActivationArena::required_prompt_bytes(graph, lanes)
    } else {
        ActivationArena::required_bytes(graph, lanes)
    }
}

/// Record the final-lane vocabulary projection of a prompt graph into the one-row logits slot.
pub(super) fn record(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    weights: &ProgramWeights,
    nvfp4: &mut super::nvfp4_gemm::Workspace,
    (row, metadata): (&mut Tensor<f32>, &Tensor<i32>),
) -> Result<(), DeviceError> {
    let quantization = weights
        .activation_quantization(node.inputs[1])
        .ok_or_else(|| error("prompt logits quantization"))?;
    let weight = weights
        .projections
        .get(&node.inputs[1])
        .ok_or_else(|| error("prompt logits projection"))?;
    let width = arena.lanes();
    let hidden = row.size();
    {
        let input = arena.get(node.inputs[0]).map_err(error)?;
        scope.record(
            kernels::last_row(
                (&mut *row).partition([1, ROW_TILE]),
                &input.view(&[width, hidden]).map_err(error)?,
                metadata,
            )
            .generics(vec![hidden.to_string()]),
        )?;
    }
    let slot = arena.slot(node.outputs[0]).map_err(error)?;
    let output = arena.buffers[slot]
        .take()
        .ok_or_else(|| error("prompt logits output"))?;
    let vocabulary = output.size();
    let mut matrix = output.reshape(&[1, vocabulary])?;
    let pointer = row.device_pointer().cu_deviceptr();
    let recorded = super::nvfp4_gemm::record_with_input_pointer(
        scope,
        nvfp4,
        weight,
        Some(quantization),
        &row.view(&[1, hidden]).map_err(error)?,
        &mut matrix,
        hidden,
        pointer,
    )?;
    arena.buffers[slot] = Some(matrix.reshape(&[vocabulary])?);
    if recorded {
        Ok(())
    } else {
        Err(error("prompt logits quantized projection"))
    }
}
