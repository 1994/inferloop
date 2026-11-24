//! The planner and permission state must reject reuse when a value contract changes.
use super::*;
use infer_core::{OpId, Result};
use infer_ir::{DType, TensorNode, TensorSpec, TensorStorage};

fn graph() -> Result<DataflowGraph> {
    let mut graph = DataflowGraph::default();
    for (id, size) in [(1, 32), (2, 64), (3, 64), (4, 64), (5, 64)] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape: vec![size],
            dtype: DType::F32,
            storage: TensorStorage::Activation,
        });
    }
    for (id, op, inputs, output) in [
        (1, TensorOp::Linear, vec![1, 10], 2),
        (2, TensorOp::Linear, vec![1, 11], 3),
        (3, TensorOp::Silu, vec![2], 4),
        (4, TensorOp::Multiply, vec![4, 3], 5),
    ] {
        graph.nodes.push(TensorNode {
            id: OpId::new(id)?,
            layer: None,
            op,
            inputs: inputs
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<_>>()?,
            outputs: vec![TensorId::new(output)?],
            states: vec![],
        });
    }
    Ok(graph)
}

fn plan(graph: &DataflowGraph, kinds: [Option<QuantKind>; 2]) -> QuantPairPlan {
    QuantPairPlan::from_contracts(
        graph,
        |_| false,
        |id| match id.get() {
            10 => kinds[0],
            11 | 12 => kinds[1],
            _ => None,
        },
    )
}

#[test]
fn only_matching_adjacent_projection_contracts_are_certified() -> Result<()> {
    let graph = graph()?;
    for kind in [QuantKind::Fp8Token, QuantKind::Fp4(1.0_f32.to_bits())] {
        let plan = plan(&graph, [Some(kind); 2]);
        assert!(matches!(plan.role(0), Some(QuantPairRole::First(_))));
        assert!(matches!(plan.role(1), Some(QuantPairRole::Second { .. })));
        assert_eq!(plan.role(2), None);
        assert_eq!(plan.role(4), None);
    }
    for kinds in [
        [None, None],
        [Some(QuantKind::Fp8Token), None],
        [
            Some(QuantKind::Fp8Token),
            Some(QuantKind::Fp4(1.0_f32.to_bits())),
        ],
        [
            Some(QuantKind::Fp4(1.0_f32.to_bits())),
            Some(QuantKind::Fp4(2.0_f32.to_bits())),
        ],
    ] {
        assert!(plan(&graph, kinds).roles.iter().all(Option::is_none));
    }
    let constant = QuantPairPlan::from_contracts(&graph, |_| true, |_| Some(QuantKind::Fp8Token));
    assert!(constant.roles.iter().all(Option::is_none));
    Ok(())
}

#[test]
fn aliases_different_values_and_intervening_work_are_rejected() -> Result<()> {
    for case in 0..8 {
        let mut graph = graph()?;
        match case {
            0 => graph.nodes[1].inputs[0] = TensorId::new(9)?,
            1 => graph.nodes[0].outputs[0] = TensorId::new(1)?,
            2 => graph.nodes[1].outputs[0] = TensorId::new(1)?,
            3 => graph.nodes[1].outputs[0] = graph.nodes[0].outputs[0],
            4 => graph.nodes[1].op = TensorOp::Add,
            5 => graph.nodes[0].states.push(TensorId::new(9)?),
            6 => graph.tensors[0].shape = vec![0],
            _ => graph.nodes[1].outputs.push(TensorId::new(9)?),
        }
        assert!(
            plan(&graph, [Some(QuantKind::Fp8Token); 2])
                .roles
                .iter()
                .all(Option::is_none)
        );
    }
    Ok(())
}

fn key() -> QuantKey {
    QuantKey {
        input_pointer: 4096,
        rows: 4,
        columns: 32,
        shape: [4, 32],
        strides: [32, 1],
        kind: QuantKind::Fp8Token,
        codes_pointer: 8192,
        scales_pointer: 12288,
    }
}
fn prime(reuse: &mut CaptureQuantReuse, plan: &QuantPairPlan) {
    reuse.enter(0, plan.role(0));
    let (skip, ticket) = reuse.prepare(Some(key()));
    assert!(!skip);
    assert!(ticket.is_some());
    reuse.finish(ticket, true);
}

#[test]
fn each_capture_and_replay_retains_its_first_quantizer() -> Result<()> {
    let plan = plan(&graph()?, [Some(QuantKind::Fp8Token); 2]);
    let mut reuse = CaptureQuantReuse::default();
    reuse.set_enabled(true);
    for _ in 0..3 {
        reuse.begin();
        prime(&mut reuse, &plan);
        reuse.enter(1, plan.role(1));
        assert!(reuse.prepare(Some(key())).0);
        assert_eq!(
            reuse.stats(),
            QuantReuseStats {
                recorded: 1,
                reused: 1
            }
        );
        assert!(
            !reuse.prepare(Some(key())).0,
            "permit must be consumed once"
        );
        reuse.end();
        reuse.enter(1, plan.role(1));
        assert!(!reuse.prepare(Some(key())).0, "outside capture");
    }
    Ok(())
}

#[test]
fn every_operand_and_buffer_field_must_match() -> Result<()> {
    let plan = plan(&graph()?, [Some(QuantKind::Fp8Token); 2]);
    for field in 0..8 {
        let mut reuse = CaptureQuantReuse::default();
        reuse.set_enabled(true);
        reuse.begin();
        prime(&mut reuse, &plan);
        let mut changed = key();
        match field {
            0 => changed.input_pointer += 4,
            1 => changed.rows += 1,
            2 => changed.columns += 1,
            3 => changed.shape[0] += 1,
            4 => changed.strides[0] += 1,
            5 => changed.kind = QuantKind::Fp4(1.0_f32.to_bits()),
            6 => changed.codes_pointer += 1,
            _ => changed.scales_pointer += 1,
        }
        reuse.enter(1, plan.role(1));
        assert!(!reuse.prepare(Some(changed)).0);
    }
    Ok(())
}

#[test]
fn skips_errors_epochs_and_unknown_views_clear_permissions() -> Result<()> {
    let plan = plan(&graph()?, [Some(QuantKind::Fp8Token); 2]);
    for invalidation in 0..7 {
        let mut reuse = CaptureQuantReuse::default();
        reuse.set_enabled(true);
        reuse.begin();
        prime(&mut reuse, &plan);
        match invalidation {
            0 => reuse.begin(),
            1 => reuse.end(),
            2 => reuse.enter(2, None),
            3 => reuse.clear(),
            4 => reuse.finish(None, false),
            5 => reuse.set_enabled(false),
            _ => {
                reuse.enter(1, plan.role(1));
                reuse.prepare(None);
            }
        }
        reuse.enter(1, plan.role(1));
        assert!(!reuse.prepare(Some(key())).0);
    }
    // Entering a node without consuming its permission cannot prime its successor.
    let mut reuse = CaptureQuantReuse::default();
    reuse.set_enabled(true);
    reuse.begin();
    prime(&mut reuse, &plan);
    reuse.enter(0, plan.role(0));
    reuse.enter(1, plan.role(1));
    assert!(!reuse.prepare(Some(key())).0);
    Ok(())
}

fn triple_graph() -> Result<DataflowGraph> {
    let mut graph = graph()?;
    graph.tensors[2].shape = vec![128];
    graph.tensors.push(TensorSpec {
        id: TensorId::new(6)?,
        shape: vec![12],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    graph.nodes.insert(
        2,
        TensorNode {
            id: OpId::new(5)?,
            layer: None,
            op: TensorOp::Linear,
            inputs: vec![TensorId::new(1)?, TensorId::new(12)?],
            outputs: vec![TensorId::new(6)?],
            states: vec![],
        },
    );
    Ok(graph)
}

#[test]
fn unequal_output_widths_and_three_consumers_share_one_quantizer() -> Result<()> {
    let graph = triple_graph()?;
    let plan = plan(&graph, [Some(QuantKind::Fp8Token); 2]);
    assert!(matches!(plan.role(0), Some(QuantPairRole::First(_))));
    assert!(matches!(plan.role(1), Some(QuantPairRole::Second { .. })));
    assert!(matches!(plan.role(2), Some(QuantPairRole::Second { .. })));
    assert_eq!(plan.role(3), None);
    let mut reuse = CaptureQuantReuse::default();
    reuse.set_enabled(true);
    reuse.begin();
    prime(&mut reuse, &plan);
    for index in 1..=2 {
        reuse.enter(index, plan.role(index));
        let (skip, ticket) = reuse.prepare(Some(key()));
        assert!(skip);
        assert_eq!(ticket.is_some(), index == 1);
        reuse.finish(ticket, true);
    }
    assert_eq!(
        reuse.stats(),
        QuantReuseStats {
            recorded: 1,
            reused: 2
        }
    );
    reuse.enter(2, plan.role(2));
    assert!(
        !reuse.prepare(Some(key())).0,
        "last consumer cannot be repeated"
    );
    Ok(())
}

#[test]
fn later_consumers_require_every_previous_success_and_unchanged_buffers() -> Result<()> {
    let plan = plan(&triple_graph()?, [Some(QuantKind::Fp8Token); 2]);
    for failure in 0..5 {
        let mut reuse = CaptureQuantReuse::default();
        reuse.set_enabled(true);
        reuse.begin();
        prime(&mut reuse, &plan);
        if failure != 0 {
            reuse.enter(1, plan.role(1));
            let mut second_key = key();
            if failure == 1 {
                second_key.codes_pointer += 1;
            }
            let (skip, ticket) = reuse.prepare(Some(second_key));
            assert_eq!(skip, failure != 1);
            if failure != 2 {
                reuse.finish(ticket, failure != 3);
            }
        }
        let mut third_key = key();
        if failure == 4 {
            third_key.scales_pointer += 1;
        }
        reuse.enter(2, plan.role(2));
        assert!(!reuse.prepare(Some(third_key)).0, "failure {failure}");
    }
    Ok(())
}
