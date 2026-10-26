use super::{gdn_ba_nodes, gdn_ba_ordered, gdn_ba_weight_compatible};
use infer_core::{OpId, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::error::Error;

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

fn graph(beta: TensorNode, alpha: TensorNode, delta: TensorNode) -> DataflowGraph {
    DataflowGraph {
        nodes: vec![beta, alpha, delta],
        ..DataflowGraph::default()
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

#[test]
fn gdn_ba_independent_pair_checks_cover_structural_order() {
    let first = linear(Some(3), 1, 10, 2);
    let second = linear(Some(3), 1, 11, 3);
    assert!(gdn_ba_nodes(&first, &second));
    let node = delta(Some(3), 2, 3, 4);
    let graph = make_graph(vec![first.clone(), second.clone(), node.clone()]);
    assert!(gdn_ba_ordered(&graph, 0, 1, 2));

    let swapped = delta(Some(3), 3, 2, 4);
    let graph = make_graph(vec![first.clone(), second.clone(), swapped]);
    assert!(!gdn_ba_ordered(&graph, 0, 1, 2));

    let graph = make_graph(vec![
        first.clone(),
        second.clone(),
        linear(Some(3), 2, 12, 4),
    ]);
    assert!(!gdn_ba_ordered(&graph, 0, 1, 2));

    let graph = make_graph(vec![first, linear(Some(3), 8, 9, 6), second, node]);
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

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn gdn_ba_pair_matches_separate_dense_projections_including_verify_width()
-> Result<(), Box<dyn Error + Send + Sync>> {
    use crate::device::CudaDevice;
    use crate::resident::prefill_gemm::gemm;
    use cutile::half::bf16;
    use cutile::prelude::*;

    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for rows in [48, 64] {
        let columns = 5120;
        let outputs = 48;
        let input = device.upload(
            (0..rows * columns)
                .map(|i| (f32::from(u16::try_from(i % 211).unwrap()) - 105.0) / 37.0)
                .collect::<Vec<_>>(),
            &[rows, columns],
        )?;
        let weight = |seed: usize| {
            device.upload(
                (0..outputs * columns)
                    .map(|i| {
                        bf16::from_f32(
                            f32::from(u16::try_from((i + seed) % 97).unwrap()) / 31.0 - 1.5,
                        )
                    })
                    .collect::<Vec<_>>(),
                &[outputs, columns],
            )
        };
        let weight_b = weight(0)?;
        let weight_a = weight(17)?;
        let mut out_b = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
        let mut out_a = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
        let mut pair_b = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
        let mut pair_a = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                gemm::dense((&mut out_b).partition([32, 64]), &input, &weight_b)
                    .generics(vec![columns.to_string()]),
            )?;
            scope.record(
                gemm::dense((&mut out_a).partition([32, 64]), &input, &weight_a)
                    .generics(vec![columns.to_string()]),
            )?;
            scope.record(
                gemm::dense_pair(
                    (&mut pair_b).partition([32, 64]),
                    (&mut pair_a).partition([32, 64]),
                    &input,
                    &weight_b,
                    &weight_a,
                )
                .generics(vec![columns.to_string()]),
            )?;
            Ok(())
        })?;
        graph.launch().sync_on(&device.stream)?;
        assert_eq!(
            device.read_borrowed(&out_b)?,
            device.read_borrowed(&pair_b)?,
            "b output rows={rows}"
        );
        assert_eq!(
            device.read_borrowed(&out_a)?,
            device.read_borrowed(&pair_a)?,
            "a output rows={rows}"
        );
    }
    Ok(())
}
