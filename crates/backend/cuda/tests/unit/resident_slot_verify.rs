use super::*;
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::{
    mlp::ProjectionWeight,
    resident::{DeviceProgram, ProgramWeights, PromptScratchRegistry, slot_batch::SlotPool},
};
use cutile::half::bf16;
use cutile::jit_cache::enable_default;
use infer_core::{OpId, TensorId};
use infer_ir::{DType, DataflowGraph, TensorNode, TensorSpec};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

fn fixture(device: &CudaDevice) -> Result<(DataflowGraph, ProgramWeights)> {
    let mut graph = DataflowGraph::default();
    for (index, shape, storage) in [
        (
            1,
            vec![32, 32],
            TensorStorage::Weight {
                slot: "embedding".into(),
            },
        ),
        (2, vec![32], TensorStorage::Activation),
        (
            3,
            vec![32, 4],
            TensorStorage::Weight {
                slot: "conv".into(),
            },
        ),
        (4, vec![32], TensorStorage::Activation),
        (
            5,
            vec![32, 4],
            TensorStorage::State {
                layer: 0,
                kind: StateKind::Conv,
            },
        ),
        (
            6,
            vec![32, 32],
            TensorStorage::Weight {
                slot: "projection".into(),
            },
        ),
        (7, vec![32], TensorStorage::Activation),
    ] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(index)?,
            shape,
            dtype: DType::F32,
            storage,
        });
    }
    for (index, op, inputs, outputs, states) in [
        (1, TensorOp::Embedding, vec![1], vec![2], vec![]),
        (
            2,
            TensorOp::Conv {
                channels: 32,
                kernel: 4,
            },
            vec![2, 3],
            vec![4],
            vec![5],
        ),
        (3, TensorOp::Linear, vec![4, 6], vec![7], vec![]),
    ] {
        let ids = |values: Vec<u64>| {
            values
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<Vec<_>>>()
        };
        graph.nodes.push(TensorNode {
            id: OpId::new(index)?,
            layer: None,
            op,
            inputs: ids(inputs)?,
            outputs: ids(outputs)?,
            states: ids(states)?,
        });
    }
    graph.hidden = Some(TensorId::new(4)?);
    graph.logits = Some(TensorId::new(7)?);
    let dense = (0..1024_u16)
        .map(|i| bf16::from_f32(f32::from(i % 29) / 29.0 - 0.5))
        .collect();
    let embedding = device.upload(dense, &[32, 32])?;
    let weights = ProgramWeights {
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        chunked_recurrent: false,
        batch_width: 3,
        prefill_width: 3,
        narrow_prefill_width: 0,
        attention_profile: crate::AttentionProfile::Canonical,
        fusion: None,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        embeddings: BTreeMap::from([(TensorId::ONE, Arc::clone(&embedding))]),
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        projections: BTreeMap::from([(TensorId::new(6)?, ProjectionWeight::Dense(embedding))]),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::from([(TensorId::new(3)?, device.upload(vec![0.25; 128], &[32, 4])?)]),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    };
    Ok((graph, weights))
}

fn add_rotary(
    device: &CudaDevice,
    graph: &mut DataflowGraph,
    weights: &mut ProgramWeights,
) -> Result<()> {
    let output = TensorId::new(8)?;
    let id = OpId::new(8)?;
    graph.tensors.push(TensorSpec {
        id: output,
        shape: vec![32],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    graph.nodes[1].inputs[0] = output;
    graph.nodes.insert(
        1,
        TensorNode {
            id,
            layer: None,
            op: TensorOp::Rope {
                heads: 4,
                head_dim: 8,
                rotary_dim: 4,
                theta: 10000.0,
            },
            inputs: vec![TensorId::new(2)?],
            outputs: vec![output],
            states: vec![],
        },
    );
    weights
        .rope_frequencies
        .insert(id, device.upload(vec![1.0, 0.01], &[2])?);
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly inside tools/bench/safe-run.sh"]
fn pooled_verify_preserves_independent_prefixes_and_reuse() -> Result<()> {
    for (width, verify) in [(2, 2), (2, 3), (3, 3), (4, 3), (8, 3), (16, 3), (2, 5)] {
        check_pool_width(width, verify, 32)?;
    }
    // Long-capacity pooled split-KV must match unsplit private graphs, including inactive lanes.
    check_pool_width(4, 3, 1024)?;
    Ok(())
}

fn check_pool_width(width: usize, verify: usize, capacity: usize) -> Result<()> {
    check_pool_width_rotary(width, verify, capacity, false)
}

/// A stateless decoder needs cursor commits, without any recurrent restore graph.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn pooled_verify_without_recurrent_state_matches_serial() -> Result<()> {
    enable_default().map_err(device_error)?;
    let device = CudaDevice::new(0)?;
    let (mut graph, mut weights) = fixture(&device)?;
    graph
        .nodes
        .retain(|node| !matches!(node.op, TensorOp::Conv { .. }));
    graph
        .tensors
        .retain(|tensor| ![3, 4, 5].contains(&tensor.id.get()));
    graph.nodes[1].inputs[0] = TensorId::new(2)?;
    graph.hidden = Some(TensorId::new(2)?);
    weights.batch_width = 8;
    weights.prefill_width = 1;
    for width in [1, 2, 4] {
        let mut pool = SlotPool::new(&device, &graph, &weights, width, 32, 32, (32, 8), None)?;
        let mut serial = DeviceProgram::new(&device, &graph, &weights, 32, 32, 32)?;
        for (position, accepted) in [(0, 2), (3, 0), (4, 1)] {
            let tokens = [1, 2, 3];
            let expected = serial.step_batch(&tokens, position)?;
            let lanes: Vec<_> = tokens
                .iter()
                .enumerate()
                .map(|(offset, &token)| (0, offset, token, position + offset, position + offset))
                .collect();
            let actual = pool.run_verify(&device, &lanes)?;
            compare(&actual[..tokens.len()], &expected);
            pool.commit_verify(&device, 0, accepted)?;
            serial.commit_batch(accepted + 1)?;
            assert_eq!(pool.cursors()[0], serial.position());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn pooled_rotary_preserves_row_consumers_prefixes_and_reuse() -> Result<()> {
    for width in [2, 5, 16] {
        check_pool_width_rotary(width, 3, 32, true)?;
    }
    Ok(())
}

fn check_pool_width_rotary(
    width: usize,
    verify: usize,
    capacity: usize,
    rotary: bool,
) -> Result<()> {
    enable_default().map_err(device_error)?;
    let device = CudaDevice::new(0)?;
    let (mut graph, mut weights) = fixture(&device)?;
    if rotary {
        add_rotary(&device, &mut graph, &mut weights)?;
    }
    weights.batch_width = verify;
    weights.prefill_width = 1;
    let mut pool = SlotPool::new(
        &device,
        &graph,
        &weights,
        width,
        capacity,
        32,
        (32, verify),
        None,
    )?;
    assert!(pool.test_batched_commit());
    let mut serial = (0..width)
        .map(|_| DeviceProgram::new(&device, &graph, &weights, 32, 32, 32))
        .collect::<Result<Vec<_>>>()?;
    // Different accepted prefixes, an inactive slot, reversed task order, and a short
    // chain exercise row groups beyond the first four rows and recurrent rollback.
    for mut plans in [
        vec![
            (0, vec![1, 2, 3], 0),
            (1, vec![7, 8, 9], 1),
            (2, vec![12, 13, 14], 2),
        ],
        vec![(2, vec![15, 16, 17], 0), (0, vec![4, 5], 1)],
        vec![(1, vec![10], 0), (2, vec![18, 19, 20], 2)],
    ] {
        plans.retain(|(slot, _, _)| *slot < width);
        for (_, tokens, accepted) in &mut plans {
            tokens.truncate(verify);
            *accepted = (*accepted).min(tokens.len() - 1);
        }
        if width >= 4 {
            plans.push((width - 1, vec![25, 26, 27], 1));
        }
        let mut lanes = Vec::new();
        let mut expected = Vec::new();
        for (slot, tokens, _) in &plans {
            let position = serial[*slot].position();
            for (offset, &token) in tokens.iter().enumerate() {
                lanes.push((*slot, offset, token, position + offset, position + offset));
            }
            expected.push(serial[*slot].step_batch(tokens, position)?);
        }
        let actual = pool.run_verify(&device, &lanes)?;
        for ((slot, tokens, accepted), reference) in plans.into_iter().zip(expected) {
            compare(&actual[slot * verify..][..tokens.len()], &reference);
            pool.commit_verify(&device, slot, accepted)?;
            serial[slot].commit_batch(accepted + 1)?;
            assert_eq!(pool.cursors()[slot], serial[slot].position());
        }
    }
    serial[1].reset()?;
    serial[1].step(21, 0, 0, None, false)?;
    pool.bind(1, serial[1].states(), serial[1].fp8_states(), 1, &device)?;
    let count = verify.min(3);
    let tokens = &[22, 23, 24][..count];
    let expected = serial[1].step_batch(tokens, 1)?;
    let actual = pool.run_verify(
        &device,
        &[(1, 0, 22, 1, 1), (1, 1, 23, 2, 2), (1, 2, 24, 3, 3)][..count],
    )?;
    compare(&actual[verify..verify + count], &expected);
    pool.commit_verify(&device, 1, count - 1)?;
    Ok(())
}

fn compare(actual: &[(Vec<f32>, Vec<f32>)], expected: &[(Vec<f32>, Vec<f32>)]) {
    for (a, b) in actual.iter().zip(expected) {
        for (x, y) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
            assert!((x - y).abs() < 0.00002, "{x} != {y}");
        }
    }
}

#[test]
fn verify_lanes_rejects_holes_duplicates_and_invalid_positions() {
    let cursors = [2, 5, 7];
    assert!(verify_lanes(&[(1, 1, 3, 6, 6)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 1, 2, 2); 2], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(usize::MAX, 0, 1, 2, 2)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 1, 3, 3)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 32, 2, 2)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(0, 0, 1, 2, 1)], &cursors, 3, 16, 32).is_err());
    assert!(verify_lanes(&[(2, 0, 1, 7, 7)], &cursors, 3, 7, 32).is_err());
    assert!(verify_lanes(&[(2, 0, 1, 7, 7), (0, 0, 2, 2, 2)], &cursors, 3, 16, 32).is_ok());
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn wide_prefill_preserves_tail_and_scalar_continuation() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (graph, mut weights) = fixture(&device)?;
    weights.batch_width = 0;
    weights.prefill_width = 1;
    let mut reference = DeviceProgram::new(&device, &graph, &weights, 256, 32, 32)?;
    weights.prefill_width = 128;
    let mut candidate = DeviceProgram::new(&device, &graph, &weights, 256, 32, 32)?;
    let mut position = 0;
    for count in [117, 11] {
        let tokens = (position..position + count)
            .map(|i| u32::try_from(i % 32).unwrap())
            .collect::<Vec<_>>();
        let actual = candidate.prefill_batch(&tokens, position, true)?;
        for (offset, token) in tokens.iter().enumerate() {
            let expected =
                reference.step(*token, position + offset, position + offset, None, true)?;
            compare(
                &[(actual[offset].0.clone(), Vec::new())],
                &[(expected.0.clone(), Vec::new())],
            );
            if offset + 1 == count {
                // Prompt projection uses BF16 inputs; compare its declared arithmetic
                // with an independent dot product instead of the F32 scalar GEMV.
                for col in 0..32 {
                    let expected: f32 = expected
                        .0
                        .iter()
                        .enumerate()
                        .map(|(row, &x)| {
                            let index = u16::try_from((col * 32 + row) % 29).unwrap();
                            bf16::from_f32(x).to_f32()
                                * bf16::from_f32(f32::from(index) / 29.0 - 0.5).to_f32()
                        })
                        .sum();
                    assert!((actual[offset].1[col] - expected).abs() < 0.00002);
                }
            }
        }
        position += count;
    }
    compare(
        &[candidate.step(7, position, position, None, true)?],
        &[reference.step(7, position, position, None, true)?],
    );
    Ok(())
}

#[test]
fn commit_batch_rejects_duplicate_and_non_pending_slots() {
    let pending = [3, 0, 2];
    assert!(validate_commits(&pending, &[(0, 2), (2, 1)]).is_ok());
    assert!(validate_commits(&pending, &[]).is_ok());
    for batch in [
        vec![(0, 0), (0, 1)],
        vec![(1, 0)],
        vec![(3, 0)],
        vec![(0, 3)],
    ] {
        assert!(validate_commits(&pending, &batch).is_err());
    }
}

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn split_decode_preserves_outputs_cursors_and_inactive_slots() -> Result<()> {
    use crate::device::asynchronous::block_on;
    enable_default().map_err(device_error)?;
    let device = CudaDevice::new(0)?;
    let (graph, mut weights) = fixture(&device)?;
    weights.prefill_width = 1;
    let mut serial = (0..3)
        .map(|_| DeviceProgram::new(&device, &graph, &weights, 512, 32, 32))
        .collect::<Result<Vec<_>>>()?;
    let mut candidate = SlotPool::new(&device, &graph, &weights, 3, 512, 32, (32, 1), None)?;
    for round in 0..128 {
        let lanes: Vec<_> = [2, 0, 1]
            .into_iter()
            .filter(|slot| (round + slot) % 3 != 0)
            .map(|slot| {
                (
                    slot,
                    u32::try_from((round + slot) % 32).unwrap(),
                    serial[slot].position(),
                    serial[slot].position(),
                )
            })
            .collect();
        let greedy = round % 2 == 0;
        let mut expected = Vec::new();
        for &(slot, token, position, _) in &lanes {
            expected.push(serial[slot].step(token, position, position, None, true)?.1);
        }
        let actual = block_on(candidate.run_lanes_readout_async(&device, &lanes, greedy))?;
        for (index, (_, logits)) in actual.rows.iter().enumerate() {
            if greedy {
                let token = expected[index]
                    .iter()
                    .enumerate()
                    .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
                    .unwrap()
                    .0;
                assert_eq!(
                    actual.tokens.as_ref().unwrap()[index],
                    Some(u32::try_from(token).unwrap())
                );
                assert_eq!(logits.as_slice(), &[0.0_f32; 0]);
            } else {
                assert_eq!(logits.len(), expected[index].len());
                assert!(logits.iter().all(|value| value.is_finite()));
                compare(
                    &[(Vec::new(), logits.clone())],
                    &[(Vec::new(), expected[index].clone())],
                );
            }
        }
        for (slot, state) in serial.iter().enumerate() {
            assert_eq!(candidate.cursors()[slot], state.position());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA/safe-run; complete private/pool state at extended protocol widths"]
fn pooled_verify_covers_extended_concurrency_and_last_slot() -> Result<()> {
    for width in [17, 32, 64] {
        check_pool_width_rotary(width, 3, 32, true)?;
    }
    Ok(())
}
