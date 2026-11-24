//! Pre-capture upper bound for shared recurrent scratch, shared by admission and capture.
use super::prepared;
use crate::constants::F32_BYTES;
use crate::constants::GDN_PROJECTED_INPUTS;
use crate::resident::ProgramWeights;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeSet;

/// Chunk experiments are explicitly configured by the diagnostic model loader.
/// A production process environment cannot switch the recurrent arithmetic.
pub(super) const fn opted_in(weights: &ProgramWeights) -> bool {
    weights.chunked_recurrent
}
pub(super) fn enabled(weights: &ProgramWeights, opted_in: bool) -> bool {
    // MoE retains scalar execution after its chunk kernel failed sanitizer parity.
    weights.expert_banks.is_empty()
        && (opted_in || (weights.input_scales.is_empty() && weights.fp8_inputs.is_empty()))
}
/// Only this exact F32 lowering is automatic for quantized recurrent projections.
/// Other geometries retain their existing scalar or explicit diagnostic policy.
pub(super) fn ordered(node: &TensorNode, weights: &ProgramWeights, lanes: usize) -> bool {
    let TensorOp::Delta {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        ..
    } = node.op
    else {
        return false;
    };
    weights.expert_banks.is_empty()
        && super::ORDERED_WIDTHS.contains(&lanes)
        && key_dim == super::ORDERED_DIM
        && value_dim == key_dim
        && key_heads > 0
        && value_heads > 0
        && value_heads.is_multiple_of(key_heads)
        && node
            .inputs
            .get(..GDN_PROJECTED_INPUTS)
            .is_some_and(|inputs| inputs.iter().all(|id| !weights.constants.contains_key(id)))
}

/// Sequential layers share output scratch and ordered Q/K/gate preparation buffers.
/// Each captured prompt width owns a workspace, including explicitly opted-in
/// quantized execution. Verification and slot decode do not retain this scratch.
pub fn required_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    lanes: usize,
) -> Result<u64> {
    let opted_in = opted_in(weights);
    let legacy_shapes = enabled(weights, opted_in);
    if !legacy_shapes && !graph.nodes.iter().any(|node| ordered(node, weights, lanes)) {
        return Ok(0);
    }
    let mut outputs = BTreeSet::new();
    let mut bytes = 0u64;
    for node in &graph.nodes {
        if !legacy_shapes && !ordered(node, weights, lanes) {
            continue;
        }
        let TensorOp::Delta {
            value_heads,
            value_dim,
            ..
        } = node.op
        else {
            continue;
        };
        let geometry = (value_heads, value_dim);
        if outputs.insert(geometry) {
            let storage = value_heads
                .checked_mul(value_dim)
                .and_then(|n| n.checked_mul(lanes))
                .and_then(|n| n.checked_mul(F32_BYTES))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| Error::invalid("CUDA recurrent scratch size overflow"))?;
            bytes = bytes
                .checked_add(storage)
                .ok_or_else(|| Error::invalid("CUDA recurrent scratch size overflow"))?;
        }
    }
    bytes
        .checked_add(prepared::required_bytes(graph, weights, lanes)?)
        .ok_or_else(|| Error::invalid("CUDA recurrent scratch size overflow"))
}
