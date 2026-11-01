use super::super::ProgramGeometry;
use super::*;
use crate::resident::BatchOutput;
use crate::resident::PromptScratchRegistry;
use crate::resident::batch::BatchReadout;
use crate::resident::metadata::identity_table;
use crate::strategy::default_tiling;
use crate::test_support::fixture_graph;
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::{mlp::ProjectionWeight, resident::FusionWeights};
use cutile::half::bf16;
use infer_core::TensorId;
use infer_cuda_kernels::tile::kernels as tile_kernels;
use infer_ir::{DType, Sampling, StateKind, TensorSpec, TensorStorage};
use std::collections::{BTreeMap, BTreeSet};

fn fixture_weights(device: &CudaDevice, width: usize) -> Result<ProgramWeights> {
    let embedding = TensorId::ONE;
    let embedding_values = (0..width * width)
        .map(|i| bf16::from_f32(f32::from(i16::try_from(i * 7 % 31).unwrap() - 15) / 16.0))
        .collect();
    let fusion_values = (0..width * 2 * width)
        .map(|i| bf16::from_f32(f32::from(i16::try_from(i * 13 % 29).unwrap() - 14) / 32.0))
        .collect();
    Ok(ProgramWeights {
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
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion: Some(FusionWeights {
            projection: ProjectionWeight::Dense(device.upload(fusion_values, &[width, 2 * width])?),
            tiling: default_tiling()?,
            norms: device.upload(vec![0.0; 2 * width], &[2, width])?,
            epsilon: 1e-6,
            offset: 1.0,
        }),
        fp8_inputs: BTreeSet::default(),
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        embeddings: BTreeMap::from([(
            embedding,
            device.upload(embedding_values, &[width, width])?,
        )]),
    })
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn worker_copy_moves_committed_kv_rows_and_whole_recurrent_state() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let hidden = 48;
    let kv = TensorId::new(4)?;
    let recurrent = TensorId::new(5)?;
    let mut graph = fixture_graph(hidden)?;
    graph.tensors.extend([
        TensorSpec {
            id: kv,
            shape: vec![2, 24],
            dtype: DType::F32,
            storage: TensorStorage::State {
                layer: 0,
                kind: StateKind::AttentionKv,
            },
        },
        TensorSpec {
            id: recurrent,
            shape: vec![2, 3],
            dtype: DType::F32,
            storage: TensorStorage::State {
                layer: 0,
                kind: StateKind::LinearAttention,
            },
        },
    ]);
    let mut weights = fixture_weights(&device, hidden)?;
    weights.fusion = None;
    let mut source = DeviceProgram::new(&device, &graph, &weights, 64, hidden, hidden)?;
    let mut target = DeviceProgram::new(&device, &graph, &weights, 64, hidden, hidden)?;
    let (heads, capacity, dim) = (3, source.capacity, 8);
    let elements = heads * capacity * dim;
    let data: Vec<_> = (0..elements)
        .map(|i| f32::from(i16::try_from(i).unwrap()))
        .collect();
    let source_kv = api::copy_host_vec_to_device(&Arc::new(data.clone()))
        .sync_on(&device.stream)
        .map_err(device_error)?
        .reshape(&[heads, capacity, dim])
        .map_err(device_error)?;
    source.states.insert(kv, vec![source_kv]);
    source.states.insert(
        recurrent,
        vec![
            api::copy_host_vec_to_device(&Arc::new(vec![5.0_f32; 6]))
                .sync_on(&device.stream)
                .map_err(device_error)?
                .reshape(&[2, 3])
                .map_err(device_error)?,
        ],
    );
    for covered in [0, 17, capacity] {
        let destination = api::copy_host_vec_to_device(&Arc::new(vec![-1.0_f32; elements]))
            .sync_on(&device.stream)
            .map_err(device_error)?
            .reshape(&[heads, capacity, dim])
            .map_err(device_error)?;
        target.states.insert(kv, vec![destination]);
        target.copy_state_from(&source, covered)?;
        let actual = device.read_borrowed(&target.states[&kv][0])?;
        for head in 0..heads {
            for row in 0..capacity {
                for column in 0..dim {
                    let index = (head * capacity + row) * dim + column;
                    assert_eq!(
                        actual[index],
                        if row < covered { data[index] } else { -1.0 }
                    );
                }
            }
        }
        assert_eq!(
            device.read_borrowed(&target.states[&recurrent][0])?,
            vec![5.0; 6]
        );
        assert_eq!(target.position(), covered);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn draft_device_hidden_reuse_matches_host_roundtrips_bitwise() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let weights = fixture_weights(&device, width)?;
    let mut host = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    let mut resident = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    assert!(resident.has_device_greedy());
    let first: Vec<_> = (0..width)
        .map(|i| f32::from(i16::try_from(i).unwrap() - 24) / 16.0)
        .collect();
    let mut previous = first.clone();
    let mut token = 7;
    for position in 0..32 {
        let (next_hidden, expected_logits) =
            host.step(token, position, position, Some(&previous), true)?;
        let expected =
            infer_workloads::sample(&expected_logits, &Sampling::default(), 0, position + 1)?;
        let actual = resident.step_greedy(
            token,
            position,
            position,
            (position == 0).then_some(first.as_slice()),
        )?;
        assert_eq!(actual, expected, "position={position}");
        let actual_hidden = device.read(&resident.hidden)?;
        let actual_logits = device.read(&resident.logits)?;
        assert_eq!(
            actual_hidden
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            next_hidden.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "hidden position={position}"
        );
        assert_eq!(
            actual_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "logits position={position}"
        );
        assert_eq!(resident.next_position, host.next_position);
        previous = next_hidden;
        token = expected;
    }
    Ok(())
}

fn assert_selected_rows(expected: &[(Vec<f32>, Vec<f32>)], actual: &BatchReadout) -> Result<()> {
    assert_eq!(expected.len(), actual.rows.len());
    let tokens = actual
        .tokens
        .as_ref()
        .ok_or_else(|| Error::invariant("fixture selected tokens"))?;
    for (row, ((hidden, logits), (actual_hidden, actual_logits))) in
        expected.iter().zip(&actual.rows).enumerate()
    {
        assert_eq!(actual_logits.as_slice(), &[] as &[f32]);
        assert_eq!(
            hidden.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            actual_hidden
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let selected = infer_workloads::sample(logits, &Sampling::default(), 0, row)?;
        assert_eq!(tokens[row], Some(selected));
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn target_device_readout_matches_host_for_continuation_and_partial_verification() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    weights.fusion = None;
    weights.batch_width = 3;
    let mut host = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    let mut selected = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    let mut token = 7;
    for position in 0..16 {
        let expected = host.step(token, position, position, None, true)?;
        let actual = selected.step_selected(token, position, true)?;
        assert_selected_rows(&[expected], &actual)?;
        assert_eq!(host.position(), selected.position());
        token = actual.tokens.as_ref().unwrap()[0].unwrap();
    }
    host.reset()?;
    selected.reset()?;
    let mut position = 0;
    for round in 0..12 {
        let chain = [7, 13, 31];
        let expected = host.step_batch(&chain, position)?;
        let actual = selected.step_batch_readout(&chain, position, true)?;
        assert_selected_rows(&expected, &actual)?;
        let committed = round % 3 + 1;
        host.commit_batch(committed)?;
        selected.commit_batch(committed)?;
        assert_eq!(host.position(), selected.position());
        position += committed;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn pooled_device_readout_matches_host_for_inactive_slots_and_partial_commits() -> Result<()> {
    use crate::resident::slot_batch::SlotPool;
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    weights.fusion = None;
    let mut host = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 1), None)?;
    let mut selected = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 1), None)?;
    for position in 0..8 {
        let lanes = [(3, 31, position, position), (1, 13, position, position)];
        let expected = host.run_lanes(&device, &lanes)?;
        let actual = selected.run_lanes_readout(&device, &lanes, true)?;
        assert_selected_rows(&expected, &actual)?;
    }
    let mut host = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 3), None)?;
    let mut selected = SlotPool::new(&device, &graph, &weights, 4, 64, width, (width, 3), None)?;
    let mut positions = [0; 4];
    for round in 0..8 {
        let lanes: Vec<_> = [3, 1]
            .into_iter()
            .flat_map(|slot| {
                (0..3).map(move |lane| {
                    (
                        slot,
                        lane,
                        [7, 13, 31][lane],
                        positions[slot] + lane,
                        positions[slot] + lane,
                    )
                })
            })
            .collect();
        let expected = host.run_verify(&device, &lanes)?;
        let actual = selected.run_verify_readout(&device, &lanes, true)?;
        assert_eq!(actual.rows.len(), 12);
        let tokens = actual.tokens.as_ref().unwrap();
        for slot in 0..4 {
            for lane in 0..3 {
                let row = slot * 3 + lane;
                if [1, 3].contains(&slot) {
                    assert_eq!(
                        tokens[row],
                        Some(infer_workloads::sample(
                            &expected[row].1,
                            &Sampling::default(),
                            0,
                            row
                        )?)
                    );
                    assert_eq!(actual.rows[row].1.as_slice(), &[] as &[f32]);
                    assert_eq!(
                        actual.rows[row]
                            .0
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected[row]
                            .0
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                } else {
                    assert_eq!(tokens[row], None);
                    assert!(actual.rows[row].0.is_empty() && actual.rows[row].1.is_empty());
                }
            }
        }
        for slot in [1, 3] {
            let accepted = (round + slot) % 3;
            host.commit_verify(&device, slot, accepted)?;
            selected.commit_verify(&device, slot, accepted)?;
            positions[slot] += accepted + 1;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn fresh_pooled_prompt_and_retained_wide_prompt_match_for_continuation() -> Result<()> {
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    weights.fusion = None;
    weights.batch_width = 3;
    weights.prefill_width = 64;
    weights.narrow_prefill_width = 32;
    let mut retained = DeviceProgram::new(&device, &graph, &weights, 256, width, width)?;
    let mut pooled = DeviceProgram::new_pooled(&device, &graph, &weights, 256, width, width)?;
    assert_eq!(retained.prefill_width(), 64);
    assert_eq!(pooled.prefill_width(), 32);
    assert!(pooled.batch.is_none() && pooled.prompts.is_empty());
    retained.forfeit_verification();
    assert!(retained.batch.is_none());
    assert_eq!(retained.prefill_width(), 64);
    assert!(retained.reclaimable_bytes > pooled.reclaimable_bytes);
    let prompt: Vec<_> = (0..63)
        .map(|i| u32::try_from(i * 7 % width).expect("fixture token fits"))
        .collect();
    let expected = retained.prefill_batch(&prompt, 0, true)?;
    let mut actual = pooled.prefill_batch(&prompt[..32], 0, false)?;
    actual.extend(pooled.prefill_batch(&prompt[32..], 32, true)?);
    let bits = |rows: &BatchOutput| {
        rows.iter()
            .map(|(h, l)| {
                (
                    h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    l.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&expected), bits(&actual));
    for position in 63..79 {
        let token = u32::try_from(position % width).expect("fixture token fits");
        let expected = retained.step(token, position, position, None, true)?;
        let actual = pooled.step(token, position, position, None, true)?;
        assert_eq!(
            bits(&vec![expected]),
            bits(&vec![actual]),
            "position={position}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn slot_program_external_rows_follow_captured_geometry() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    let mut reference = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    weights.prefill_width = 2048;
    // This graph has no recurrent state or KV, so the slot handle needs no aliases.
    let views = (
        BTreeMap::new(),
        BTreeMap::new(),
        identity_table(&device, 64)?,
    );
    let mut home =
        DeviceProgram::new_on_slot(&device, &graph, &weights, (64, width, width), views)?;
    assert_eq!(home.lane_external_owner.size(), width);
    assert_eq!(home.lane_external.len(), 1);
    assert_eq!(home.lane_fusion.len(), 1);
    let previous = vec![0.125; width];
    let expected = reference.step_greedy(7, 0, 0, Some(&previous))?;
    let actual = home.step_greedy(7, 0, 0, Some(&previous))?;
    assert_eq!(actual, expected);
    let expected = device.read(&reference.logits)?;
    let actual = device.read(&home.logits)?;
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn slot_target_handle_runs_only_through_the_pool() -> Result<()> {
    use crate::resident::slot_batch::SlotPool;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let width = 48;
    let graph = fixture_graph(width)?;
    let mut weights = fixture_weights(&device, width)?;
    weights.fusion = None;
    let mut reference = DeviceProgram::new(&device, &graph, &weights, 64, width, width)?;
    weights.prefill_width = 2048;
    let mut pool = SlotPool::new(&device, &graph, &weights, 2, 64, width, (width, 0), None)?;
    assert!(pool.take_slot(0));
    let views = (
        BTreeMap::new(),
        BTreeMap::new(),
        identity_table(&device, 64)?,
    );
    let mut home =
        DeviceProgram::new_on_slot(&device, &graph, &weights, (64, width, width), views)?;
    assert!(!home.forward_captured);
    assert!(home.step(7, 0, 0, None, true).is_err());
    home.reset()?;
    let expected = reference.step(7, 0, 0, None, true)?;
    let actual = pool.run_lanes(&device, &[(0, 7, 0, 0)])?;
    assert_eq!(
        actual[0].1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    Ok(())
}

/// A shared worker can capture fewer prompt rows than the model's global preference.
/// Every delegated scratch must be prepared before capture at that actual geometry.
#[test]
#[ignore = "requires CUDA hardware and cuBLAS; run under safe-run"]
fn delegated_worker_prewarms_its_actual_prompt_geometry() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let hidden = 2048;
    let graph = fixture_graph(hidden)?;
    let mut weights = fixture_weights(&device, hidden)?;
    weights.prefill_width = 2048;
    let geometry = ProgramGeometry::from_flags(0, 512, 0, false, true);
    let program = DeviceProgram::new_impl(
        &device,
        &graph,
        &weights,
        (1024, hidden, hidden),
        geometry,
        None,
    )?;
    assert!(program.delegated.is_enabled());
    assert_eq!(program.prefill_width(), 512);
    Ok(())
}

/// Eviction must not invalidate another program's captured shared prompt storage.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn prompt_scratch_reclaims_only_after_all_captured_users_drop() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let hidden = 48;
    let graph = fixture_graph(hidden)?;
    let mut weights = fixture_weights(&device, hidden)?;
    weights.fusion = None;
    weights.prefill_width = 64;
    weights.narrow_prefill_width = 32;
    let mut first = DeviceProgram::new(&device, &graph, &weights, 128, hidden, hidden)?;
    let mut second = DeviceProgram::new(&device, &graph, &weights, 128, hidden, hidden)?;
    let prompt: Vec<_> = (0..64)
        .map(|i| u32::try_from(i % hidden).unwrap())
        .collect();
    let expected = first.prefill_batch(&prompt, 0, true)?;
    assert!(weights.prompt_scratch.is_allocated(64)?);
    assert!(weights.prompt_scratch.is_allocated(32)?);
    assert_eq!(weights.prompt_scratch.reclaim_unused(&device)?, 0);
    drop(first);
    assert_eq!(weights.prompt_scratch.reclaim_unused(&device)?, 0);
    let actual = second.prefill_batch(&prompt, 0, true)?;
    assert_eq!(actual, expected);
    drop(second);
    assert_eq!(weights.prompt_scratch.reclaim_unused(&device)?, 2);
    assert!(!weights.prompt_scratch.is_allocated(64)?);
    assert!(!weights.prompt_scratch.is_allocated(32)?);
    assert_eq!(weights.prompt_scratch.reclaim_unused(&device)?, 0);
    let mut replacement = DeviceProgram::new(&device, &graph, &weights, 128, hidden, hidden)?;
    assert_eq!(replacement.prefill_batch(&prompt, 0, true)?, expected);
    Ok(())
}

/// Shrinking pooled workers must leave private/long-context prompt geometry unchanged.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn pooled_worker_width_is_independent_of_private_capture() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let hidden = 48;
    let graph = fixture_graph(hidden)?;
    let mut weights = fixture_weights(&device, hidden)?;
    weights.fusion = None;
    weights.prefill_width = 64;
    weights.narrow_prefill_width = 0;
    let mut worker =
        DeviceProgram::new_worker(&device, &graph, &weights, (128, hidden, hidden), None, 32)?;
    let mut private = DeviceProgram::new(&device, &graph, &weights, 128, hidden, hidden)?;
    assert_eq!(worker.prefill_width(), 32);
    assert_eq!(private.prefill_width(), 64);
    let prompt: Vec<_> = (0..64)
        .map(|i| u32::try_from(i % hidden).unwrap())
        .collect();
    let expected = private.prefill_batch(&prompt, 0, true)?;
    worker.prefill_batch(&prompt[..32], 0, false)?;
    assert_eq!(
        worker.prefill_batch(&prompt[32..], 32, true)?,
        expected[32..]
    );
    Ok(())
}
