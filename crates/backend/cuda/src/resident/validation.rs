use super::ProgramWeights;
use crate::constants::MAX_HIDDEN_SIZE;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorOp, TensorStorage};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// `GatedNorm` binds activation, gate and norm weight inputs.
const GATED_NORM_INPUTS: usize = 3;
/// Delta-rule mixer binds qkv, beta, alpha, log-decay and bias inputs.
const DELTA_INPUTS: usize = 5;
/// Attention binds query, key and value inputs.
const ATTENTION_INPUTS: usize = 3;

/// Refuse at load what the kernels cannot record, so a request never finds it first.
///
/// cuTile partitions tensors into power-of-two tiles, so these geometries need a general cell
/// rather than a different call.
pub fn require_kernel_geometry(graph: &DataflowGraph) -> Result<()> {
    graph
        .nodes
        .iter()
        .try_for_each(|node| check_kernel_geometry(&node.op))
}

fn check_kernel_geometry(op: &TensorOp) -> Result<()> {
    match *op {
        TensorOp::Rope {
            head_dim,
            rotary_dim,
            ..
        } if !(rotary_dim / 2).is_power_of_two() || !head_dim.is_multiple_of(rotary_dim / 2) => {
            Err(Error::unsupported(format!(
                "CUDA rotary kernels need a power-of-two half dimension dividing head_dim; \
                 got rotary_dim {rotary_dim}, head_dim {head_dim}"
            )))
        }
        TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            ..
        } if kv_heads == 0
            || !query_heads.is_multiple_of(kv_heads)
            || !head_dim.is_power_of_two() =>
        {
            Err(Error::unsupported(format!(
                "CUDA attention kernels need a power-of-two head_dim and heads divisible by \
                 kv_heads; got head_dim {head_dim}, heads {query_heads}/{kv_heads}"
            )))
        }
        _ => Ok(()),
    }
}

pub(super) fn validate(graph: &DataflowGraph, weights: &ProgramWeights) -> Result<()> {
    graph.validate()?;
    let specs: BTreeMap<_, _> = graph.tensors.iter().map(|spec| (spec.id, spec)).collect();
    for node in &graph.nodes {
        let (inputs, outputs, states) = match &node.op {
            TensorOp::Embedding
            | TensorOp::Silu
            | TensorOp::Sigmoid
            | TensorOp::Rope { .. }
            | TensorOp::MoeRoute { .. } => (1, 1, 0),
            TensorOp::Linear
            | TensorOp::Norm { .. }
            | TensorOp::Add
            | TensorOp::Multiply
            | TensorOp::Scale
            | TensorOp::MoeCombine { .. } => (2, 1, 0),
            TensorOp::ExpertLinear { experts, .. } => (2 + experts, 1, 0),
            TensorOp::Split { widths, .. } => (1, widths.len(), 0),
            TensorOp::GatedNorm { .. } => (GATED_NORM_INPUTS, 1, 0),
            TensorOp::Conv { .. } => (2, 1, 1),
            TensorOp::Delta { .. } => (DELTA_INPUTS, 1, 1),
            TensorOp::Attention { .. } => (ATTENTION_INPUTS, 1, 1),
        };
        if node.inputs.len() != inputs
            || node.outputs.len() != outputs
            || node.states.len() != states
        {
            return Err(Error::invalid("resident operation binding count"));
        }
        let output = specs[&node.outputs[0]].elements()?;
        let input = specs[&node.inputs[0]].elements()?;
        super::moe::validate_node(node, &specs, weights, input, output)?;
        if node.op == TensorOp::Linear {
            weights
                .projections
                .get(&node.inputs[1])
                .ok_or_else(|| Error::invalid("missing projection"))?
                .validate(output, input)?;
        }
        if let TensorOp::Rope {
            heads,
            head_dim,
            rotary_dim,
            theta,
        } = node.op
            && (heads.checked_mul(head_dim) != Some(output)
                || input != output
                || rotary_dim == 0
                || rotary_dim > head_dim
                || !theta.is_finite()
                || theta <= 0.0)
        {
            return Err(Error::invalid("resident rotary geometry"));
        }
        if let TensorOp::Norm {
            head_dim,
            epsilon,
            offset,
        } = node.op
        {
            validate_norm(head_dim, input, output, epsilon, offset)?;
        }
        if let TensorOp::GatedNorm { head_dim, epsilon } = node.op {
            validate_norm(head_dim, input, output, epsilon, 0.0)?;
        }
        if let TensorOp::Attention {
            kv_heads, head_dim, ..
        } = node.op
        {
            let state = specs[&node.states[0]];
            if state.shape.len() != 2
                || kv_heads.checked_mul(head_dim) != state.shape.get(1).copied()
            {
                return Err(Error::invalid("resident KV geometry"));
            }
        }
    }
    let bank_ids: BTreeSet<_> = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, TensorOp::ExpertLinear { .. }))
        .flat_map(|node| node.inputs[2..].iter().copied())
        .collect();
    for spec in &graph.tensors {
        if matches!(spec.storage, TensorStorage::Weight { .. })
            && !bank_ids.contains(&spec.id)
            && !weights.constants.contains_key(&spec.id)
            && !weights.projections.contains_key(&spec.id)
            && !weights.embeddings.contains_key(&spec.id)
        {
            return Err(Error::invalid("missing resident weight binding"));
        }
    }
    Ok(())
}

fn validate_norm(
    dimension: usize,
    input: usize,
    output: usize,
    epsilon: f32,
    offset: f32,
) -> Result<()> {
    if dimension == 0
        || dimension > MAX_HIDDEN_SIZE
        || input != output
        || !input.is_multiple_of(dimension)
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !offset.is_finite()
    {
        return Err(Error::invalid("resident normalization geometry"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/resident_validation.rs"]
mod tests;
