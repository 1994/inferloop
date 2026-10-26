use super::{gdn_ba_nodes, gdn_ba_ordered};
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

fn make_delta(layer: Option<usize>, beta: u64, alpha: u64, output: u64) -> TensorNode {
    TensorNode {
        id: OpId::new(output).unwrap(),
        layer,
        op: TensorOp::Delta {
            key_heads: 16,
            value_heads: 32,
            key_dim: 128,
            value_dim: 128,
        },
        inputs: vec![
            TensorId::new(9).unwrap(),
            TensorId::new(beta).unwrap(),
            TensorId::new(alpha).unwrap(),
        ],
        outputs: vec![TensorId::new(output).unwrap()],
        states: Vec::new(),
    }
}

fn graph(beta: TensorNode, alpha: TensorNode, delta: TensorNode) -> DataflowGraph {
    DataflowGraph {
        nodes: vec![beta, alpha, delta],
        ..DataflowGraph::default()
    }
}

#[test]
fn structural_delta_role_requires_beta_then_alpha_for_one_delta() {
    let beta = linear(Some(3), 1, 10, 2);
    let alpha = linear(Some(3), 1, 11, 3);
    let delta = make_delta(Some(3), 2, 3, 4);
    assert!(gdn_ba_nodes(&beta, &alpha));
    assert!(gdn_ba_ordered(
        &graph(beta.clone(), alpha.clone(), delta.clone()),
        0,
        1,
        2
    ));

    let swapped = graph(beta.clone(), alpha.clone(), make_delta(Some(3), 3, 2, 4));
    assert!(!gdn_ba_ordered(&swapped, 0, 1, 2));
    let wrong_layer = graph(beta.clone(), alpha.clone(), make_delta(Some(4), 2, 3, 4));
    assert!(!gdn_ba_ordered(&wrong_layer, 0, 1, 2));
    let mut extra_output = alpha;
    extra_output.outputs.push(TensorId::new(12).unwrap());
    assert!(!gdn_ba_ordered(&graph(beta, extra_output, delta), 0, 1, 2));
}

#[test]
fn role_selector_accepts_alternate_beta_alpha_cardinalities_and_rejects_decoys() {
    use super::gdn_ba_roles_by;
    use crate::resident::prefill_projection::GdnRole;
    use std::collections::BTreeMap;
    let beta = linear(Some(3), 1, 10, 2);
    let alpha = linear(Some(3), 1, 11, 3);
    let delta = make_delta(Some(3), 2, 3, 4);
    for rows in [32, 40, 48, 64] {
        let flow = graph(beta.clone(), alpha.clone(), delta.clone());
        let shapes = BTreeMap::from([
            (TensorId::new(10).unwrap(), (rows, 5120)),
            (TensorId::new(11).unwrap(), (rows, 5120)),
        ]);
        let roles = gdn_ba_roles_by(&flow, |id| shapes.get(&id).copied());
        assert_eq!(roles.get(&0), Some(&GdnRole::Beta));
        assert_eq!(roles.get(&1), Some(&GdnRole::Alpha));
    }
    // Same-shaped adjacent linears with no Delta consumer are not roles.
    let decoy = graph(beta.clone(), alpha.clone(), linear(Some(3), 1, 12, 4));
    let shapes = BTreeMap::from([
        (TensorId::new(10).unwrap(), (48, 5120)),
        (TensorId::new(11).unwrap(), (48, 5120)),
    ]);
    assert!(gdn_ba_roles_by(&decoy, |id| shapes.get(&id).copied()).is_empty());
    // Missing or mismatched weights are ambiguous/invalid.
    let flow = graph(beta.clone(), alpha.clone(), delta);
    assert!(gdn_ba_roles_by(&flow, |_| None).is_empty());
    let mismatch = BTreeMap::from([
        (TensorId::new(10).unwrap(), (48, 5120)),
        (TensorId::new(11).unwrap(), (40, 5120)),
    ]);
    assert!(gdn_ba_roles_by(&flow, |id| mismatch.get(&id).copied()).is_empty());
    // Swapped semantic inputs are rejected.
    let swapped = graph(beta, alpha, make_delta(Some(3), 3, 2, 4));
    assert!(gdn_ba_roles_by(&swapped, |id| shapes.get(&id).copied()).is_empty());
}

#[test]
fn role_selector_ignores_one_input_producer_without_indexing_panic() {
    use super::gdn_ba_roles_by;
    use crate::resident::prefill_projection::GdnRole;
    let mut beta = linear(Some(3), 1, 10, 2);
    beta.inputs.pop();
    let alpha = linear(Some(3), 1, 11, 3);
    let flow = graph(beta, alpha, make_delta(Some(3), 2, 3, 4));
    let roles = gdn_ba_roles_by(&flow, |_| Some((48, 5120)));
    assert!(roles.is_empty());
    assert!(!roles.values().any(|role| *role == GdnRole::Beta));
}
