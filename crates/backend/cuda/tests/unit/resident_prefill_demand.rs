use super::*;
use infer_core::{OpId, Result, TensorId};
use infer_ir::{TensorNode, TensorOp};
use std::panic::catch_unwind;

fn node(id: u64, inputs: &[u64], outputs: &[u64], state: bool) -> Result<TensorNode> {
    Ok(TensorNode {
        id: OpId::new(id)?,
        layer: None,
        op: if state {
            TensorOp::Conv {
                channels: 1,
                kernel: 4,
            }
        } else {
            TensorOp::Linear
        },
        inputs: inputs
            .iter()
            .copied()
            .map(TensorId::new)
            .collect::<Result<_>>()?,
        outputs: outputs
            .iter()
            .copied()
            .map(TensorId::new)
            .collect::<Result<_>>()?,
        states: if state {
            vec![TensorId::new(200 + id)?]
        } else {
            vec![]
        },
    })
}
#[test]
fn every_state_effect_and_its_dependency_closure_is_retained() -> Result<()> {
    let graph = DataflowGraph {
        nodes: vec![
            node(1, &[100], &[1], false)?,
            node(2, &[1, 101], &[2], false)?,
            node(3, &[2, 102], &[3], true)?,
            node(4, &[3, 103], &[4], false)?,
            node(5, &[4, 104], &[5], false)?,
            node(6, &[5, 105], &[6], false)?,
            node(7, &[1, 106], &[7], true)?,
        ],
        hidden: Some(TensorId::new(5)?),
        logits: Some(TensorId::new(6)?),
        features: None,
        ..DataflowGraph::default()
    };
    let plan = StateDemand::new(&graph);
    assert_eq!(plan.retained, [true, true, true, false, false, false, true]);
    assert!(
        plan.retains(graph.nodes.len()),
        "unknown node conservatively retained"
    );
    let mut graph = graph;
    graph.nodes[6].inputs.push(TensorId::new(5)?);
    assert_eq!(
        StateDemand::new(&graph).retained,
        [true, true, true, true, true, false, true]
    );
    Ok(())
}
#[test]
fn multi_output_and_stateful_dead_results_keep_all_required_inputs() -> Result<()> {
    let graph = DataflowGraph {
        nodes: vec![
            node(1, &[100], &[1], false)?,
            node(2, &[1, 101], &[2, 3], false)?,
            node(3, &[3, 102], &[4], true)?,
            node(4, &[2, 103], &[5], false)?,
        ],
        ..DataflowGraph::default()
    };
    assert_eq!(StateDemand::new(&graph).retained, [true, true, true, false]);
    Ok(())
}
#[test]
fn internal_control_is_scoped_and_restored_after_unwinding() {
    assert!(!enabled());
    with_state_prefill(true, || {
        assert!(enabled());
        with_state_prefill(false, || assert!(!enabled()));
        assert!(enabled());
        let result = catch_unwind(|| with_state_prefill(false, || panic!("restore")));
        assert!(result.is_err());
        assert!(enabled());
    });
    assert!(!enabled());
}
