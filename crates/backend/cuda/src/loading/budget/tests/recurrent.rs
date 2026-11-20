use super::fixture_weights;
use crate::resident::ProgramGeometry;
use crate::resident::recurrent_prefill::required_bytes;
use infer_core::{OpId, Result, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};

fn graph() -> DataflowGraph {
    DataflowGraph {
        nodes: vec![TensorNode {
            id: OpId::ONE,
            layer: None,
            op: TensorOp::Delta {
                key_heads: 16,
                value_heads: 48,
                key_dim: 128,
                value_dim: 128,
            },
            inputs: vec![TensorId::ONE; 3],
            outputs: vec![TensorId::ONE],
            states: vec![],
        }],
        ..DataflowGraph::default()
    }
}

// The chunked path retains one F32 normalization row, two F32 gates, BF16
// Q/K/U/W/interaction matrices, and F32 cumulative decay for each geometry.
const fn preparation_bytes(key_heads: u64, lanes: u64) -> u64 {
    let normalized = 2 * key_heads * 128 * 4;
    let gates = 2 * 48 * lanes * 4;
    let queries_and_keys = 2 * key_heads * lanes * 128 * 2;
    let updates_and_weights = 2 * 48 * lanes * 128 * 2;
    let interactions = 48 * lanes * 64 * 2;
    let cumulative_decay = 48 * lanes * 4;
    normalized + gates + queries_and_keys + updates_and_weights + interactions + cumulative_decay
}

#[test]
fn quantized_recurrent_admission_counts_retained_prompt_graphs_and_shares_layers() -> Result<()> {
    let mut weights = fixture_weights();
    weights.input_scales.insert(TensorId::ONE, 1.0);
    weights.chunked_recurrent = true;
    weights.prefill_width = 256;
    weights.narrow_prefill_width = 64;
    let mut graph = graph();
    // Even scalar fallback nodes share output storage by value geometry.
    graph.nodes[0].op = TensorOp::Delta {
        key_heads: 0,
        value_heads: 48,
        key_dim: 128,
        value_dim: 128,
    };
    let mut later = graph.nodes[0].clone();
    later.id = OpId::new(2)?;
    later.op = TensorOp::Delta {
        key_heads: 16,
        value_heads: 48,
        key_dim: 128,
        value_dim: 128,
    };
    graph.nodes.push(later.clone());
    later.id = OpId::new(3)?;
    graph.nodes.push(later);
    let quote = |private| {
        ProgramGeometry::new(&weights, private)
            .prompts
            .into_iter()
            .filter(|width| *width > 0)
            .try_fold(0, |total, width| {
                Ok::<_, infer_core::Error>(total + required_bytes(&graph, &weights, width)?)
            })
    };
    // Output storage is shared by value geometry; preparation is shared by full geometry.
    let wide = 48 * 128 * 256 * 4 + preparation_bytes(16, 256);
    let narrow = 48 * 128 * 64 * 4 + preparation_bytes(16, 64);
    assert_eq!(quote(false)?, narrow);
    assert_eq!(quote(true)?, wide + narrow);
    weights.fp8_inputs.insert(TensorId::ONE);
    assert_eq!(required_bytes(&graph, &weights, 256)?, wide);
    Ok(())
}

#[test]
fn recurrent_admission_counts_all_prompt_scratch_widths() -> Result<()> {
    let mut weights = fixture_weights();
    weights.chunked_recurrent = true;
    assert_eq!(required_bytes(&graph(), &weights, 16)?, 48 * 128 * 16 * 4);
    for lanes in [64, 256, 512, 1024, 2048] {
        assert_eq!(
            required_bytes(&graph(), &weights, lanes)?,
            48 * 128 * lanes as u64 * 4 + preparation_bytes(16, lanes as u64)
        );
    }
    Ok(())
}

#[test]
fn recurrent_admission_rejects_scratch_overflow_before_allocation() {
    let mut weights = fixture_weights();
    weights.chunked_recurrent = true;
    let mut graph = graph();
    graph.nodes[0].op = TensorOp::Delta {
        key_heads: 1,
        value_heads: usize::MAX,
        key_dim: 128,
        value_dim: 128,
    };
    assert!(required_bytes(&graph, &weights, 256).is_err());
}

#[test]
fn ordered_recurrent_preparation_counts_distinct_key_geometries() -> Result<()> {
    let mut weights = fixture_weights();
    weights.chunked_recurrent = true;
    let mut graph = graph();
    let mut node = graph.nodes[0].clone();
    node.id = OpId::new(2)?;
    node.op = TensorOp::Delta {
        key_heads: 8,
        value_heads: 48,
        key_dim: 128,
        value_dim: 128,
    };
    graph.nodes.push(node.clone());
    node.id = OpId::new(3)?;
    graph.nodes.push(node);
    let output = 48 * 128 * 64 * 4;
    let first_inputs = preparation_bytes(16, 64);
    let second_inputs = preparation_bytes(8, 64);
    assert_eq!(
        required_bytes(&graph, &weights, 64)?,
        output + first_inputs + second_inputs
    );
    Ok(())
}

#[test]
fn quantized_default_prices_only_exact_ordered_recurrent_shapes() -> Result<()> {
    let mut weights = fixture_weights();
    weights.input_scales.insert(TensorId::ONE, 1.0);
    assert!(!weights.chunked_recurrent);
    let graph = graph();
    for width in [64, 256, 512, 1024, 2048] {
        let expected = 48 * 128 * width as u64 * 4 + preparation_bytes(16, width as u64);
        assert_eq!(required_bytes(&graph, &weights, width)?, expected);
    }
    assert_eq!(required_bytes(&graph, &weights, 32)?, 0);
    let mut unsupported = graph.clone();
    unsupported.nodes[0].op = TensorOp::Delta {
        key_heads: 1,
        value_heads: 2,
        key_dim: 32,
        value_dim: 32,
    };
    assert_eq!(required_bytes(&unsupported, &weights, 64)?, 0);
    let mut mixed = graph;
    let mut later = unsupported.nodes[0].clone();
    later.id = OpId::new(2)?;
    mixed.nodes.push(later);
    assert_eq!(
        required_bytes(&mixed, &weights, 64)?,
        48 * 128 * 64 * 4 + preparation_bytes(16, 64)
    );
    Ok(())
}
