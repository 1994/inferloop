use super::{gdn_ba_nodes, gdn_ba_ordered, gdn_ba_weight_compatible};
use infer_core::{OpId, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};

fn linear(layer: Option<usize>, input: u64, weight: u64, output: u64) -> TensorNode {
    TensorNode {
        id: OpId::new(output).unwrap(),
        layer,
        op: TensorOp::Linear,
        inputs: vec![
            TensorId::new(input).unwrap(),
            TensorId::new(weight).unwrap(),
        ],
        outputs: vec![TensorId::new(output).unwrap()],
        states: Vec::new(),
    }
}

fn delta(layer: Option<usize>, beta: u64, alpha: u64, output: u64) -> TensorNode {
    TensorNode {
        id: OpId::new(output).unwrap(),
        layer,
        op: TensorOp::Delta {
            key_heads: 16,
            value_heads: 32,
            key_dim: 128,
            value_dim: 128,
        },
        inputs: [1, beta, alpha, 4, 5]
            .into_iter()
            .map(|id| TensorId::new(id).unwrap())
            .collect(),
        outputs: vec![TensorId::new(output).unwrap()],
        states: Vec::new(),
    }
}

fn make_graph(nodes: Vec<TensorNode>) -> DataflowGraph {
    DataflowGraph {
        tensors: Vec::new(),
        nodes,
        hidden: None,
        logits: None,
        lifetimes: Vec::new(),
        scratch_elements: 0,
    }
}

#[test]
fn gdn_ba_pair_is_structural_beta_alpha_of_one_delta() {
    let first = linear(Some(3), 1, 10, 2);
    let second = linear(Some(3), 1, 11, 3);
    assert!(gdn_ba_nodes(&first, &second));

    let other_layer = linear(Some(4), 1, 11, 3);
    assert!(!gdn_ba_nodes(&first, &other_layer));
    let other_input = linear(Some(3), 9, 11, 3);
    assert!(!gdn_ba_nodes(&first, &other_input));
    let mut not_linear = second.clone();
    not_linear.op = TensorOp::Silu;
    assert!(!gdn_ba_nodes(&first, &not_linear));
    let mut second_output = second;
    second_output.outputs.push(TensorId::new(12).unwrap());
    assert!(!gdn_ba_nodes(&first, &second_output));
}

#[test]
fn gdn_ba_order_requires_beta_then_alpha_producers() {
    let beta = linear(Some(3), 1, 10, 2);
    let alpha = linear(Some(3), 1, 11, 3);
    let node = delta(Some(3), 2, 3, 4);
    let graph = make_graph(vec![beta.clone(), alpha.clone(), node.clone()]);
    assert!(gdn_ba_ordered(&graph, 0, 1, 2));

    // Swapping the Delta's beta/alpha inputs must not pair the same adjacent linears.
    let swapped = delta(Some(3), 3, 2, 4);
    let graph = make_graph(vec![beta.clone(), alpha.clone(), swapped]);
    assert!(!gdn_ba_ordered(&graph, 0, 1, 2));

    // Adjacent same-shaped linears with no Delta consumer are not a GDN b/a pair.
    let graph = make_graph(vec![beta.clone(), alpha.clone(), linear(Some(3), 2, 12, 4)]);
    assert!(!gdn_ba_ordered(&graph, 0, 1, 2));

    // A second tensor between the producers and the Delta is not accepted.
    let graph = make_graph(vec![beta, linear(Some(3), 8, 9, 6), alpha, node]);
    assert!(!gdn_ba_ordered(&graph, 0, 2, 3));
}

#[test]
fn gdn_ba_weight_compatibility_is_shape_and_tile_bounded() {
    assert!(gdn_ba_weight_compatible(&[48, 5120], &[48, 5120]));
    assert!(!gdn_ba_weight_compatible(&[48, 5120], &[48, 4096]));
    assert!(!gdn_ba_weight_compatible(&[48, 5120], &[64, 5120]));
    assert!(!gdn_ba_weight_compatible(&[0, 5120], &[0, 5120]));
    assert!(!gdn_ba_weight_compatible(&[65, 5120], &[65, 5120]));
}
