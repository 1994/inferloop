use super::*;
use crate::mlp::ProjectionWeight;
use crate::resident::{DeviceProgram, FusionWeights, PromptScratchRegistry};
use crate::strategy::default_tiling;
use crate::tuning::resident_gemm::ResidentTuningSession;
use cutile::half::bf16;
use infer_core::OpId;
use infer_cuda_kernels::tile::kernels as tile_kernels;
use infer_ir::{DType, TensorNode, TensorOp, TensorSpec};
use std::collections::{BTreeMap, BTreeSet};

fn fixture(device: &CudaDevice) -> Result<(DataflowGraph, ProgramWeights)> {
    let mut graph = DataflowGraph::default();
    for (id, shape, storage) in [
        (
            1,
            vec![32, 32],
            TensorStorage::Weight {
                slot: "embedding".into(),
            },
        ),
        (2, vec![32], TensorStorage::Activation),
        (3, vec![32], TensorStorage::Activation),
        (
            4,
            vec![1, 32],
            TensorStorage::State {
                layer: 0,
                kind: StateKind::AttentionKv,
            },
        ),
        (5, vec![32], TensorStorage::Activation),
    ] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape,
            dtype: DType::F32,
            storage,
        });
    }
    for (id, op, inputs, output, states) in [
        (1, TensorOp::Embedding, vec![1], 2, vec![]),
        (
            2,
            TensorOp::Attention {
                query_heads: 1,
                kv_heads: 1,
                head_dim: 32,
                window: None,
            },
            vec![2, 2, 2],
            3,
            vec![4],
        ),
        (3, TensorOp::Add, vec![3, 3], 5, vec![]),
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
            states: states
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<_>>()?,
        });
    }
    graph.hidden = Some(TensorId::new(3)?);
    graph.logits = Some(TensorId::new(5)?);
    let values = |n| {
        (0..n)
            .map(|i| bf16::from_f32(f32::from(u16::try_from(i % 31).unwrap()) / 31.0 - 0.5))
            .collect()
    };
    let weights = ProgramWeights {
        kernels: Arc::new(tile_kernels(0)),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        chunked_recurrent: false,
        batch_width: 0,
        prefill_width: 1,
        narrow_prefill_width: 0,
        attention_profile: crate::AttentionProfile::Canonical,
        fusion: Some(FusionWeights {
            projection: ProjectionWeight::Dense(device.upload(values(32 * 64), &[32, 64])?),
            tiling: default_tiling()?,
            norms: device.upload(vec![1.0; 64], &[2, 32])?,
            epsilon: 1e-6,
            offset: 0.0,
        }),
        embeddings: BTreeMap::from([(TensorId::ONE, device.upload(values(32 * 32), &[32, 32])?)]),
        projections: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    };
    Ok((graph, weights))
}
fn external(slot: usize, step: usize) -> Vec<f32> {
    (0..32)
        .map(|i| f32::from(u16::try_from((i + slot * 7 + step * 3) % 29).unwrap()) / 17.0 - 0.5)
        .collect()
}
fn close(a: &(Vec<f32>, Vec<f32>), b: &(Vec<f32>, Vec<f32>)) {
    assert_eq!(a.0.len(), b.0.len());
    assert_eq!(a.1.len(), b.1.len());
    for (a, b) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
        assert!((a - b).abs() < 0.00002, "{a} != {b}");
    }
}
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn external_hidden_batch_preserves_private_kv_and_rollback() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (graph, weights) = fixture(&device)?;
    let mut pool = SlotPool::new(&device, &graph, &weights, 4, 64, 32, (32, 0), None)?;
    let mut references = (0..4)
        .map(|_| DeviceProgram::new(&device, &graph, &weights, 128, 32, 32))
        .collect::<Result<Vec<_>>>()?;
    for (slot, reference) in references.iter_mut().enumerate() {
        for position in 0..slot + 2 {
            reference.step(
                u32::try_from(position).unwrap(),
                position + 1,
                position,
                Some(&external(slot, position)),
                true,
            )?;
        }
        pool.bind(
            slot,
            reference.states(),
            reference.fp8_states(),
            reference.position(),
            &device,
        )?;
    }
    for active in [vec![0, 1, 2, 3], vec![2, 0], vec![1], vec![3, 1, 0]] {
        let hidden: Vec<_> = active
            .iter()
            .map(|&slot| external(slot, references[slot].position()))
            .collect();
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let p = references[slot].position();
                (slot, u32::try_from(p % 32).unwrap(), p + 1, p)
            })
            .collect();
        let actual = pool.run_external(
            &device,
            &lanes,
            &hidden.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )?;
        for ((&slot, h), actual) in active.iter().zip(&hidden).zip(&actual) {
            let p = references[slot].position();
            let expected =
                references[slot].step(u32::try_from(p % 32).unwrap(), p + 1, p, Some(h), true)?;
            close(actual, &expected);
        }
    }
    // Different rejection depths share one catch-up replay. Inactive slots must
    // retain their timeline, and rewound rows must overwrite the rejected suffix.
    for (slot, reference) in references.iter_mut().enumerate() {
        let position = reference.position() - slot.min(2);
        pool.rewind_external(slot, position)?;
        reference.rewind_attention(position)?;
    }
    assert!(pool.rewind_external(0, 65).is_err());
    assert!(pool.rewind_external(4, 0).is_err());
    for active in [vec![0, 1, 2, 3], vec![1, 3]] {
        let hidden: Vec<_> = active.iter().map(|&slot| external(slot, 37)).collect();
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let p = references[slot].position();
                (slot, 23, p + 1, p)
            })
            .collect();
        let actual = pool.run_external(
            &device,
            &lanes,
            &hidden.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )?;
        for ((&slot, h), actual) in active.iter().zip(&hidden).zip(&actual) {
            let p = references[slot].position();
            close(actual, &references[slot].step(23, p + 1, p, Some(h), true)?);
        }
    }
    check_state_only(&device, &mut pool, &mut references)?;
    for (slot, reference) in references.iter_mut().enumerate() {
        let mut restored = DeviceProgram::new(&device, &graph, &weights, 128, 32, 32)?;
        pool.export_attention(slot, &mut restored, &device)?;
        let p = reference.position() - 1;
        reference.rewind_attention(p)?;
        restored.rewind_attention(p)?;
        let h = external(slot, 19);
        close(
            &reference.step(17, p + 1, p, Some(&h), true)?,
            &restored.step(17, p + 1, p, Some(&h), true)?,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn accepted_draft_tokens_refresh_kv_with_target_hidden_rows() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (graph, weights) = fixture(&device)?;
    let mut pool = SlotPool::new(&device, &graph, &weights, 2, 64, 32, (32, 0), None)?;
    let mut reference = DeviceProgram::new(&device, &graph, &weights, 128, 32, 32)?;
    for position in 0..3 {
        reference.step(
            2,
            position + 1,
            position,
            Some(&external(0, position)),
            false,
        )?;
    }
    pool.bind(0, reference.states(), reference.fp8_states(), 3, &device)?;
    let submitted_hidden = external(0, 3);
    let draft = pool.run_external(&device, &[(0, 5, 4, 3)], &[&submitted_hidden])?;
    reference.step(5, 4, 3, Some(&submitted_hidden), false)?;
    // Simulate two accepted proposals. The second draft step used its own hidden
    // row, whereas the committed draft context must use the target's row.
    pool.run_external(&device, &[(0, 7, 5, 4)], &[&draft[0].0])?;
    pool.rewind_external(0, 4)?;
    for (offset, token) in [7, 9].into_iter().enumerate() {
        let target_hidden = external(1, offset + 21);
        pool.run_external_readout(
            &device,
            &[(0, token, 5 + offset, 4 + offset)],
            &[&target_hidden],
            false,
        )?;
        reference.step(token, 5 + offset, 4 + offset, Some(&target_hidden), false)?;
    }
    let next = external(1, 30);
    let actual = pool.run_external(&device, &[(0, 11, 7, 6)], &[&next])?;
    close(&actual[0], &reference.step(11, 7, 6, Some(&next), true)?);
    Ok(())
}

fn check_state_only(
    device: &CudaDevice,
    pool: &mut SlotPool,
    references: &mut [DeviceProgram],
) -> Result<()> {
    // State-only catch-up omits the vocabulary head and host readout. Its next
    // ordinary decode must still match independently executed private programs.
    for read_logits in [false, true] {
        let active = [3, 0, 2];
        let hidden: Vec<_> = active.iter().map(|&slot| external(slot, 43)).collect();
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let p = references[slot].position();
                (slot, 11, p + 1, p)
            })
            .collect();
        let actual = pool.run_external_readout(
            device,
            &lanes,
            &hidden.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            read_logits,
        )?;
        for ((&slot, h), actual) in active.iter().zip(&hidden).zip(&actual) {
            let p = references[slot].position();
            let expected = references[slot].step(11, p + 1, p, Some(h), true)?;
            if read_logits {
                close(actual, &expected);
            } else {
                assert!(actual.0.is_empty() && actual.1.is_empty());
            }
        }
    }
    Ok(())
}

/// Non-identity block tables must scatter a private prefix, isolate live slots, and
/// overwrite a released slot's old data before its blocks serve another request.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn paged_kv_scatter_survives_exhaustion_and_block_reuse() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (graph, mut weights) = fixture(&device)?;
    weights.fusion = None;
    weights.kv_scales.insert(TensorId::new(4)?, [0.01, 0.02]);
    let mut pool = SlotPool::new(&device, &graph, &weights, 3, 96, 32, (32, 0), Some(160))?;
    assert_eq!(pool.kv_pages(), (5, 5));
    let mut references = (0..3)
        .map(|_| DeviceProgram::new(&device, &graph, &weights, 96, 32, 32))
        .collect::<Result<Vec<_>>>()?;
    for (slot, count) in [(0, 33), (1, 65)] {
        assert!(pool.take_slot(slot));
        assert!(pool.assign(slot, count + 1, &device)?);
        for position in 0..count {
            references[slot].step(
                u32::try_from((position + slot * 7) % 32).unwrap(),
                position,
                position,
                None,
                false,
            )?;
        }
        pool.bind(
            slot,
            references[slot].states(),
            references[slot].fp8_states(),
            count,
            &device,
        )?;
    }
    assert!(pool.take_slot(2));
    assert_eq!(pool.kv_pages(), (5, 0));
    assert!(!pool.assign(2, 1, &device)?);
    assert_eq!(pool.kv_pages(), (5, 0));
    assert!(pool.assign(0, pool.capacity() + 1, &device).is_err());
    let table = device.read_borrowed(&pool.tables[1])?;
    assert_ne!(&table[..2], &[0, 1], "slot 1 must read through its table");
    for active in [vec![1, 0], vec![0], vec![1]] {
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let position = references[slot].position();
                (
                    slot,
                    u32::try_from((position + slot * 7) % 32).unwrap(),
                    position,
                    position,
                )
            })
            .collect();
        let actual = pool.run_lanes(&device, &lanes)?;
        for ((&slot, &(.., token, position, _)), actual) in active.iter().zip(&lanes).zip(&actual) {
            let expected = references[slot].step(token, position, position, None, true)?;
            close(actual, &(Vec::new(), expected.1));
        }
    }
    pool.release(0);
    assert_eq!(pool.kv_pages(), (5, 2));
    assert!(pool.assign(2, 34, &device)?);
    assert_eq!(pool.kv_pages(), (5, 0));
    for position in 0..33 {
        references[2].step(
            u32::try_from((position * 3 + 5) % 32).unwrap(),
            position,
            position,
            None,
            false,
        )?;
    }
    pool.bind(
        2,
        references[2].states(),
        references[2].fp8_states(),
        33,
        &device,
    )?;
    let actual = pool.run_lanes(&device, &[(2, 17, 33, 33), (1, 19, 67, 67)])?;
    let expected = references[2].step(17, 33, 33, None, true)?;
    close(&actual[0], &(Vec::new(), expected.1));
    let expected = references[1].step(19, 67, 67, None, true)?;
    close(&actual[1], &(Vec::new(), expected.1));
    pool.release(1);
    pool.release(2);
    assert_eq!(pool.paging.as_ref().unwrap().pages.free_blocks(), 5);
    assert_eq!(pool.free_slots(), 3);
    Ok(())
}

fn paged_attention_reuse_fixture(device: &CudaDevice) -> Result<(DataflowGraph, ProgramWeights)> {
    let (mut graph, mut weights) = fixture(device)?;
    // Make the attention output temporary: the final Add reuses its physical arena slot.
    // Keeping owned shared row aliases would panic when that slot is written again.
    graph.tensors.push(TensorSpec {
        id: TensorId::new(6)?,
        shape: vec![32],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    graph.nodes.push(TensorNode {
        id: OpId::new(4)?,
        layer: None,
        op: TensorOp::Add,
        inputs: vec![TensorId::new(5)?, TensorId::new(5)?],
        outputs: vec![TensorId::new(6)?],
        states: Vec::new(),
    });
    graph.hidden = Some(TensorId::new(5)?);
    graph.logits = Some(TensorId::new(6)?);
    weights.fusion = None;
    weights.kv_scales.insert(TensorId::new(4)?, [0.01, 0.02]);
    graph.plan_lifetimes()?;
    let slot = |id| {
        graph
            .lifetimes
            .iter()
            .find(|life| life.tensor == id)
            .unwrap()
            .slot
    };
    assert_eq!(slot(TensorId::new(3)?), slot(TensorId::new(6)?));
    Ok((graph, weights))
}

/// Three candidates read one paged cache while preserving each candidate's causal prefix.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn paged_verify_attention_matches_scalar_causality_and_reuse() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (graph, weights) = paged_attention_reuse_fixture(&device)?;
    let mut pool = SlotPool::new(&device, &graph, &weights, 4, 128, 32, (32, 3), Some(384))?;
    let mut serial = (0..4)
        .map(|_| DeviceProgram::new(&device, &graph, &weights, 128, 32, 32))
        .collect::<Result<Vec<_>>>()?;
    for (slot, prefix) in [17, 33, 65, 97].into_iter().enumerate() {
        assert!(pool.take_slot(slot));
        assert!(pool.assign(slot, prefix + 16, &device)?);
        for position in 0..prefix {
            serial[slot].step(
                u32::try_from((position * 3 + slot * 7) % 32).unwrap(),
                position,
                position,
                None,
                false,
            )?;
        }
        pool.bind(
            slot,
            serial[slot].states(),
            serial[slot].fp8_states(),
            prefix,
            &device,
        )?;
    }
    assert_ne!(&device.read_borrowed(&pool.tables[1])?[..2], &[0, 1]);
    for plans in [
        vec![
            (3, vec![3, 7, 11], 2),
            (1, vec![9, 13, 17], 0),
            (0, vec![1, 5], 1),
        ],
        vec![
            (2, vec![19, 23, 27], 1),
            (3, vec![4], 0),
            (1, vec![10, 14], 1),
        ],
        vec![
            (0, vec![6, 12, 18], 0),
            (2, vec![20], 0),
            (3, vec![8, 16, 24], 2),
        ],
    ] {
        let mut lanes = Vec::new();
        let mut expected = Vec::new();
        for (slot, tokens, accepted) in &plans {
            let start = serial[*slot].position();
            let mut output = Vec::new();
            for (offset, &token) in tokens.iter().enumerate() {
                let position = start + offset;
                lanes.push((*slot, offset, token, position, position));
                output.push(serial[*slot].step(token, position, position, None, true)?);
            }
            serial[*slot].rewind_attention(start + accepted + 1)?;
            expected.push(output);
        }
        let actual = pool.run_verify(&device, &lanes)?;
        for ((slot, tokens, accepted), expected) in plans.into_iter().zip(expected) {
            for (row, expected) in expected.iter().enumerate() {
                close(&actual[slot * 3 + row], expected);
            }
            assert_eq!(tokens.len(), expected.len());
            pool.commit_verify(&device, slot, accepted)?;
            assert_eq!(pool.cursors()[slot], serial[slot].position());
        }
    }
    pool.release(1);
    serial[1].reset()?;
    serial[1].step(29, 0, 0, None, false)?;
    assert!(pool.take_slot(1));
    assert!(pool.assign(1, 4, &device)?);
    pool.bind(1, serial[1].states(), serial[1].fp8_states(), 1, &device)?;
    let actual = pool.run_verify(&device, &[(1, 0, 2, 1, 1), (1, 1, 28, 2, 2)])?;
    close(&actual[3], &serial[1].step(2, 1, 1, None, true)?);
    close(&actual[4], &serial[1].step(28, 2, 2, None, true)?);
    pool.commit_verify(&device, 1, 1)?;
    Ok(())
}
