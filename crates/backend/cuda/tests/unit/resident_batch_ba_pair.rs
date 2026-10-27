use crate::resident::prefill_projection::DensePlan;
use infer_core::{OpId, TensorId};
use infer_ir::{DType, DataflowGraph, StateKind, TensorNode, TensorOp, TensorSpec, TensorStorage};

fn id(value: u64) -> TensorId {
    TensorId::new(value).unwrap()
}

fn fixture(heads: usize, columns: usize) -> DataflowGraph {
    let mut graph = DataflowGraph::default();
    for (index, shape, storage) in [
        (1, vec![columns], TensorStorage::Activation),
        (
            2,
            vec![heads, columns],
            TensorStorage::Weight {
                slot: "unrelated-name-left".into(),
            },
        ),
        (3, vec![heads], TensorStorage::Activation),
        (
            4,
            vec![heads, columns],
            TensorStorage::Weight {
                slot: "different-name-right".into(),
            },
        ),
        (5, vec![heads], TensorStorage::Activation),
        (6, vec![(2 + heads) * 4], TensorStorage::Activation),
        (
            7,
            vec![heads],
            TensorStorage::Weight {
                slot: "rate".into(),
            },
        ),
        (
            8,
            vec![heads],
            TensorStorage::Weight {
                slot: "bias".into(),
            },
        ),
        (
            9,
            vec![heads * 4, 4],
            TensorStorage::State {
                layer: 3,
                kind: StateKind::LinearAttention,
            },
        ),
        (10, vec![heads * 4], TensorStorage::Activation),
    ] {
        graph.tensors.push(TensorSpec {
            id: id(index),
            shape,
            dtype: DType::F32,
            storage,
        });
    }
    for (index, op, inputs, outputs, states) in [
        (1, TensorOp::Linear, vec![1, 2], vec![3], vec![]),
        (2, TensorOp::Linear, vec![1, 4], vec![5], vec![]),
        (
            3,
            TensorOp::Delta {
                key_heads: 1,
                value_heads: heads,
                key_dim: 4,
                value_dim: 4,
            },
            vec![6, 3, 5, 7, 8],
            vec![10],
            vec![9],
        ),
    ] {
        graph.nodes.push(TensorNode {
            id: OpId::new(index).unwrap(),
            layer: Some(3),
            op,
            inputs: inputs.into_iter().map(id).collect(),
            outputs: outputs.into_iter().map(id).collect(),
            states: states.into_iter().map(id).collect(),
        });
    }
    graph
}

#[test]
fn dense_siblings_are_identified_across_model_dimensions() {
    for (heads, columns) in [(16, 96), (48, 5120), (64, 128), (128, 96)] {
        let plan = DensePlan::new(&fixture(heads, columns), None);
        assert_eq!(plan.narrow(0), heads <= 64);
        assert_eq!(plan.narrow(1), heads <= 64);
        assert_eq!(plan.pair(0), Some(1));
        assert_eq!(plan.pair(1), None);
    }
}

#[test]
fn same_input_dense_siblings_do_not_require_a_specific_consumer() {
    let mut graph = fixture(48, 5120);
    graph.nodes.truncate(2);
    let plan = DensePlan::new(&graph, None);
    assert!(plan.narrow(0) && plan.narrow(1));
    assert_eq!(plan.pair(0), Some(1));
}

#[test]
fn pairing_requires_adjacent_nodes_and_a_common_activation() {
    let mut graph = fixture(16, 96);
    graph.tensors.push(TensorSpec {
        id: id(11),
        shape: vec![96],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    graph.nodes[1].inputs[0] = id(11);
    let plan = DensePlan::new(&graph, None);
    assert!(plan.narrow(0) && plan.narrow(1));
    assert_eq!(plan.pair(0), None);
    let mut graph = fixture(16, 96);
    let mut intervening = graph.nodes[0].clone();
    intervening.id = OpId::new(4).unwrap();
    intervening.op = TensorOp::Silu;
    intervening.inputs = vec![id(6)];
    intervening.outputs = vec![id(11)];
    graph.nodes.insert(1, intervening);
    let plan = DensePlan::new(&graph, None);
    assert!(plan.narrow(0) && plan.narrow(2));
    assert_eq!(plan.pair(0), None);
}

#[test]
fn invalid_dense_geometry_and_stateful_nodes_are_rejected() {
    for variant in 0..6 {
        let mut graph = fixture(16, 96);
        match variant {
            0 => graph.nodes[1].op = TensorOp::Silu,
            1 => graph.nodes[1].states.push(id(9)),
            2 => graph.nodes[1].inputs.push(id(7)),
            3 => {
                graph
                    .tensors
                    .iter_mut()
                    .find(|s| s.id == id(4))
                    .unwrap()
                    .shape = vec![17, 96];
            }
            4 => {
                graph
                    .tensors
                    .iter_mut()
                    .find(|s| s.id == id(5))
                    .unwrap()
                    .shape = vec![17];
            }
            _ => {
                graph
                    .tensors
                    .iter_mut()
                    .find(|s| s.id == id(1))
                    .unwrap()
                    .shape = vec![usize::MAX, 2];
            }
        }
        let plan = DensePlan::new(&graph, None);
        assert!(!plan.narrow(1), "variant {variant}");
        assert_eq!(plan.pair(0), None);
    }
}

#[test]
fn reverse_control_order_preserves_graph_recording_order() {
    let mut graph = fixture(16, 96);
    graph.nodes.swap(0, 1);
    let plan = DensePlan::new(&graph, None);
    assert!(plan.narrow(0) && plan.narrow(1));
    assert_eq!(plan.pair(0), Some(1));
}

#[test]
fn pairing_rejects_shared_outputs_and_cross_layer_siblings() {
    let mut graph = fixture(16, 96);
    graph.nodes[1].outputs[0] = graph.nodes[0].outputs[0];
    let plan = DensePlan::new(&graph, None);
    assert_eq!(plan.pair(0), None);
    let mut graph = fixture(16, 96);
    graph.nodes[1].layer = Some(4);
    let plan = DensePlan::new(&graph, None);
    assert!(plan.narrow(0) && plan.narrow(1));
    assert_eq!(plan.pair(0), None);
}

#[test]
fn single_row_logits_keep_their_separate_readout_path() {
    let mut graph = fixture(16, 96);
    graph.logits = Some(id(5));
    let full = DensePlan::new(&graph, None);
    assert_eq!(full.pair(0), Some(1));
    let last_row = DensePlan::new(&graph, graph.logits);
    assert!(last_row.narrow(0));
    assert!(!last_row.narrow(1));
    assert_eq!(last_row.pair(0), None);
}

#[path = "resident_gdn_projection_gpu.rs"]
mod gpu;
