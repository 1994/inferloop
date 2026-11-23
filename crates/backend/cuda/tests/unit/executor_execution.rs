use super::{SlotPool, candidate_pool_capacity, pool_retry_budget, pooled_envelope};
use crate::executor::CudaExecutor;
use infer_core::{RequestId, StateId};
use infer_cuda_kernels::runtime::Allocation;
use infer_ir::{ExecutionInput, ExecutionTask, OutputReadout, TaskOutput, TokenBuffer};
use std::{env::var, error::Error};

/// The model context the candidate tests plan against.
const LIMIT: usize = 8192;

#[test]
fn incompatible_requests_cannot_create_or_enlarge_a_decode_pool() {
    let full = (128, OutputReadout::Full);
    let long = (LIMIT + 1, OutputReadout::Logits);
    let compact = (128, OutputReadout::Logits);
    for states in [
        [full, full],
        [long, long],
        [full, long],
        [compact, full],
        [long, compact],
    ] {
        assert_eq!(candidate_pool_capacity(states.into_iter(), LIMIT), None);
    }
}

#[test]
fn compatible_pool_geometry_is_independent_of_unrelated_request_order() {
    let short = (128, OutputReadout::Logits);
    let tail = (129, OutputReadout::Logits);
    let full = (LIMIT, OutputReadout::Full);
    let long = (LIMIT + 1, OutputReadout::Logits);
    for states in [[short, full, tail, long], [long, tail, full, short]] {
        assert_eq!(
            candidate_pool_capacity(states.into_iter(), LIMIT),
            Some(256)
        );
    }
    assert_eq!(candidate_pool_capacity([short].into_iter(), LIMIT), None);
    assert_eq!(candidate_pool_capacity([].into_iter(), LIMIT), None);
}

#[test]
fn the_pool_bucket_never_exceeds_the_model_context() {
    let context = 40_960;
    let long = (context - 1, OutputReadout::Logits);
    assert_eq!(
        candidate_pool_capacity([long, long].into_iter(), context),
        Some(context)
    );
}

#[test]
fn lazy_pool_preserves_live_private_reservations_and_scratch() {
    // 40 reserved by live sequences and the shared scratch is still owned by them.
    let lazy = pooled_envelope(1_000, 100, 40, 20);
    assert_eq!(lazy, 60);
    assert_eq!(lazy + 40 + 20, 120);
    // At load time that same scratch reservation can instead fund the worker.
    assert_eq!(pooled_envelope(1_000, 100, 0, 20), 120);
}

#[test]
fn pool_envelope_is_capped_by_physical_storage_and_saturates() {
    assert_eq!(pooled_envelope(50, 100, 0, 20), 50);
    assert_eq!(pooled_envelope(50, 100, 40, 20), 50);
    assert_eq!(pooled_envelope(1_000, 100, 101, 20), 0);
    assert_eq!(pooled_envelope(u64::MAX, u64::MAX, 0, 20), u64::MAX);
}

#[test]
#[ignore = "real MTP2 checkpoint and CUDA; run inside safe-run"]
fn lazy_pool_retries_after_a_live_private_reservation_is_released() -> Result<(), Box<dyn Error>> {
    use crate::{
        device::CudaDevice,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::ModelId;

    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            mtp_depth: 2,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    assert!(loaded.model().max_sequence >= 32768);
    let budget = loaded.device().allocatable_memory_bytes()?;
    // Start without a pool, then admit one long and two short private programs. The long one
    // reads its full history, so no pool serves it and it is not a solo the others wait for.
    let mut backend = CudaExecutor::new(loaded, budget, 1)?;
    backend.maximum_states = 16;
    let long = StateId::new(1)?;
    backend.reserve(long, 32768, OutputReadout::Full, true)?;
    backend.reserve(StateId::new(2)?, 512, OutputReadout::Logits, true)?;
    backend.reserve(StateId::new(3)?, 512, OutputReadout::Logits, true)?;
    let mut lower = 0;
    let mut upper = budget;
    assert!(backend.pooled_plan_for(512, 16, upper, 0)?.is_some());
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if backend.pooled_plan_for(512, 16, middle, 0)?.is_some() {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    // Existing reservations are legal; the additional pool is one byte short.
    let active = backend.active_budget()?;
    backend.budget = active.checked_add(lower - 1).ok_or("budget overflow")?;
    backend.ensure_slot_pool();
    assert!(backend.slots.is_none());
    assert!(!backend.slots_disabled);
    assert_eq!(backend.deferred_pool_plan, Some((512, active)));
    backend.ensure_slot_pool();
    assert_eq!(backend.deferred_pool_plan, Some((512, active)));

    backend.release(long)?;
    while backend.pool.evict() {}
    backend.ensure_slot_pool();
    assert!(backend.slots.is_some(), "{:?}", backend.pool_failure_reason);
    assert!(backend.worker.is_some());
    assert!(!backend.slots_disabled);
    assert!(backend.deferred_pool_plan.is_none());
    assert!(backend.homes.iter().all(Option::is_some));
    Ok(())
}

#[test]
fn unquoted_capture_storage_cannot_expand_the_serving_envelope() {
    // A quote nearly fills the envelope; capture materializes another 20 bytes.
    let funding = pool_retry_budget(100, 99, 119, 8);
    assert_eq!(funding, 80);
    assert!(funding + 20 <= 100);
    // Even exact tensor quotes leave one physical allocation quantum of headroom.
    assert_eq!(pool_retry_budget(100, 100, 100, 8), 92);
    assert_eq!(pool_retry_budget(8, 8, 100, 8), 0);
    assert_eq!(pool_retry_budget(100, 100, 100, 0), 99);
}

#[test]
fn retry_quote_accounts_for_storage_retained_by_previous_captures() {
    let envelope = 100;
    let first = pool_retry_budget(envelope, 99, 119, 8);
    assert_eq!(first, 80);
    // The next tensor capture uses 90 bytes but the first left 20 resident:
    // measuring only the second capture would incorrectly publish 110 bytes.
    let total_after_second = 20 + 90;
    assert!(total_after_second > envelope);
    let next = pool_retry_budget(first, 79, total_after_second, 8);
    assert_eq!(next, 49);
    // Both retained increments fit alongside the smaller final capture.
    assert!(next + 20 + 10 <= envelope);
}

fn media_position_task(
    state: StateId,
    request: RequestId,
    tokens: ExecutionInput,
) -> ExecutionTask {
    ExecutionTask {
        request,
        state,
        tokens,
        sampling: None,
        plain_greedy: false,
        publish_prefix: false,
        media: None,
    }
}

fn run_media_position_tasks(
    executor: &mut CudaExecutor,
    tasks: &[ExecutionTask],
) -> Result<Vec<TaskOutput>, Box<dyn Error>> {
    use crate::device::asynchronous::block_on;
    executor.prepare_execution()?;
    let outputs = block_on(executor.execute(tasks));
    let fence = block_on(executor.loaded.device().fence());
    executor.busy = false;
    fence?;
    Ok(outputs?)
}

/// A pool slot and the full logits it produced.
type SlotLogits = (usize, Vec<f32>);

/// Logits of `state`'s next token replayed through the pool on `lanes` slots at once, each slot
/// bound to a copy of the same sequence and positioned at an explicitly supplied rotary
/// coordinate. Returns the leased slots with their logits, in ascending slot order.
///
/// The pool replays the narrowest captured graph that covers its highest busy slot, so a
/// bitwise comparison only holds against a step that occupies the same slots; callers check
/// that the step under test leased exactly these slots.
fn pooled_position_reference(
    executor: &mut CudaExecutor,
    state: StateId,
    (token, rotary): (u32, usize),
    lanes: usize,
) -> Result<Vec<SlotLogits>, Box<dyn Error>> {
    use crate::device::asynchronous::block_on;
    use std::mem::take;
    let sequence = executor.states.get(&state).ok_or("missing sequence")?;
    let device = executor.loaded.device();
    let pool = executor.slots.as_mut().ok_or("missing pool")?;
    let position = sequence.history.len();
    let mut slots = Vec::with_capacity(lanes);
    for _ in 0..lanes {
        let slot = pool.take_free().ok_or("missing free slot")?;
        assert!(pool.assign(slot, sequence.capacity, device)?);
        pool.bind(
            slot,
            sequence.program.states(),
            sequence.program.fp8_states(),
            position,
            device,
        )?;
        slots.push(slot);
    }
    slots.sort_unstable();
    let (rows, output): (Vec<usize>, _) = if let Some(width) = pool.verify() {
        let lanes: Vec<_> = slots
            .iter()
            .map(|&slot| (slot, 0, token, rotary, position))
            .collect();
        (
            slots.iter().map(|&slot| slot * width).collect(),
            block_on(pool.run_verify_readout_async(device, &lanes, false))?,
        )
    } else {
        let lanes: Vec<_> = slots
            .iter()
            .map(|&slot| (slot, token, rotary, position))
            .collect();
        (
            (0..slots.len()).collect(),
            block_on(pool.run_lanes_readout_async(device, &lanes, false))?,
        )
    };
    block_on(device.fence())?;
    for &slot in &slots {
        pool.release(slot);
    }
    let mut readout = output.rows;
    slots
        .into_iter()
        .zip(rows)
        .map(|(slot, row)| {
            let logits = readout.get_mut(row).ok_or("missing logits")?;
            Ok((slot, take(&mut logits.1)))
        })
        .collect()
}

/// Panics with the size of the disagreement when `actual` is not bitwise `expected`.
fn assert_bitwise(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: logit count");
    let differing = actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
        .count();
    let largest = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        differing == 0,
        "{context}: {differing} of {} logits differ, largest |difference| {largest}, top {:?} vs {:?}",
        expected.len(),
        top_logit(actual),
        top_logit(expected),
    );
}

fn top_logit(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
}

fn media_position_fixture(
    mtp_depth: usize,
) -> Result<(CudaExecutor, TokenBuffer, [StateId; 3]), Box<dyn Error>> {
    use crate::device::CudaDevice;
    use crate::loading::{LoadOptions, LoadedModel};
    use infer_core::ModelId;
    use infer_ir::{MediaItem, MediaPlan, Modality};
    use std::sync::Arc;

    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 32,
            mtp_depth,
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let budget = loaded.device().allocatable_memory_bytes()?;
    let mut executor = if mtp_depth == 0 {
        CudaExecutor::new(loaded, budget, 3)?
    } else {
        // Only the target verification graph is needed: media reservations do not draft.
        let mut executor = CudaExecutor::new_bounded(loaded, budget, 3)?;
        executor.slots = Some(executor.loaded.slot_pool(2, 128, Some(256))?);
        executor
    };
    assert!(
        executor.slots.is_some(),
        "test needs the pooled decode path"
    );
    let prompt = TokenBuffer::from((1..=64).collect::<Vec<u32>>());
    let plan = MediaPlan {
        modality: Modality::Image,
        prompt: prompt.clone(),
        items: vec![MediaItem {
            modality: Modality::Image,
            pixels: Arc::<[f32]>::from([1.0_f32]).into(),
            grid: (1, 2, 2),
            span: 1..2,
        }],
    };
    let states = [StateId::ONE, StateId::new(2)?, StateId::new(3)?];
    let state_count = if mtp_depth == 0 { states.len() } else { 1 };
    for (index, &state) in states.iter().take(state_count).enumerate() {
        executor.reserve_media(state, 128, OutputReadout::Logits, &plan)?;
        let request = RequestId::new(u64::try_from(index)? + 1)?;
        let task = media_position_task(
            state,
            request,
            ExecutionInput::Prefill {
                span: prompt.span(0..prompt.len())?,
                readout: OutputReadout::Logits,
            },
        );
        run_media_position_tasks(&mut executor, &[task])?;
        let sequence = executor.states.get_mut(&state).ok_or("missing sequence")?;
        sequence.rotary_base = 8;
        sequence.media_prompt_len = prompt.len();
    }
    Ok((executor, prompt, states))
}

#[test]
#[ignore = "real Qwen3-VL text checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn pooled_decode_preserves_multimodal_rotary_position() -> Result<(), Box<dyn Error>> {
    let (mut executor, prompt, states) = media_position_fixture(0)?;
    let decode = |state, request| {
        media_position_task(
            state,
            request,
            ExecutionInput::Decode {
                token: 65,
                position: prompt.len(),
            },
        )
    };
    // Replay the pooled graph directly with an independently supplied rotary position, on the
    // same two slots the batch will lease. This keeps the full-logit comparison bitwise while
    // exercising the executor's lane construction.
    let expected = pooled_position_reference(&mut executor, states[0], (65, 8), 2)?;
    assert!(
        expected.iter().all(|(_, logits)| !logits.is_empty()),
        "pooled reference needs complete logits"
    );
    let single = run_media_position_tasks(&mut executor, &[decode(states[0], RequestId::ONE)])?;
    let batch = run_media_position_tasks(
        &mut executor,
        &[
            decode(states[1], RequestId::new(2)?),
            decode(states[2], RequestId::new(3)?),
        ],
    )?;
    assert!(executor.states[&states[0]].slot.is_none());
    for (output, state) in batch.iter().zip(&states[1..]) {
        let slot = executor.states[state]
            .slot
            .as_ref()
            .map(|lease| lease.slot)
            .ok_or("batched sequence has no slot")?;
        let (_, reference) = expected
            .iter()
            .find(|(reference, _)| *reference == slot)
            .ok_or("batch leased a slot the reference did not replay")?;
        assert_bitwise(
            &output.output.logits,
            reference,
            &format!("pooled slot {slot} against the positioned reference"),
        );
    }
    let serial = &single[0].output.logits;
    let (_, reference) = &expected[0];
    assert_eq!(serial.len(), reference.len());
    assert_eq!(top_logit(serial), top_logit(reference));
    Ok(())
}

#[test]
#[ignore = "real MTP2 checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn pooled_verification_preserves_multimodal_rotary_position() -> Result<(), Box<dyn Error>> {
    let (mut executor, prompt, states) = media_position_fixture(2)?;
    assert_eq!(executor.slots.as_ref().and_then(SlotPool::verify), Some(3));
    assert!(executor.states[&states[0]].speculation.is_none());
    let [(slot, expected)] = <[_; 1]>::try_from(pooled_position_reference(
        &mut executor,
        states[0],
        (65, 8),
        1,
    )?)
    .map_err(|_| "one reference lane")?;
    assert!(!expected.is_empty(), "verification needs complete logits");
    let task = media_position_task(
        states[0],
        RequestId::ONE,
        ExecutionInput::Decode {
            token: 65,
            position: prompt.len(),
        },
    );
    let outputs = run_media_position_tasks(&mut executor, &[task])?;
    let leased = executor.states[&states[0]]
        .slot
        .as_ref()
        .map(|lease| lease.slot);
    assert_eq!(
        leased,
        Some(slot),
        "verification replays the reference slot"
    );
    assert_bitwise(
        &outputs[0].output.logits,
        &expected,
        "pooled verification against the positioned reference",
    );
    Ok(())
}
#[test]
#[ignore = "real Qwen text checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn failed_media_chunk_retains_its_device_source_until_the_fence() -> Result<(), Box<dyn Error>> {
    use crate::device::asynchronous::block_on;
    use infer_ir::{MediaItem, MediaPlan, Modality};
    use infer_models::MediaLayout;
    use std::sync::Arc;

    let (mut executor, _, states) = media_position_fixture(0)?;
    let device = executor.loaded.device();
    let hidden = executor.loaded.model().hidden_size;
    let owner = device.upload(vec![0.0; hidden], &[1, hidden])?;
    let prompt = TokenBuffer::from((1..=66).collect::<Vec<u32>>());
    let plan = MediaPlan {
        modality: Modality::Image,
        prompt: prompt.clone(),
        items: vec![MediaItem {
            modality: Modality::Image,
            pixels: Arc::<[f32]>::from([1.0_f32]).into(),
            grid: (1, 2, 2),
            span: 64..66,
        }],
    };
    let task = media_position_task(
        states[0],
        RequestId::ONE,
        ExecutionInput::Prefill {
            span: prompt.span(64..66)?,
            readout: OutputReadout::Logits,
        },
    );
    let sequence = executor
        .states
        .get_mut(&states[0])
        .ok_or("missing sequence")?;
    sequence.media = Some(super::CachedMedia {
        tensor: Allocation::adopt(Arc::clone(&owner))?,
        features: None,
        layout: MediaLayout {
            spans: Vec::new(),
            next_rotary: 0,
            positions: None,
        },
        bases: vec![0],
    });
    // The first row can be submitted, but the deliberately short source makes the
    // second row fail. Keep this extra owner so the failing reproduction is safe.
    let result = block_on(sequence.run_media(&task, &plan, &executor.loaded));
    let retained = sequence
        .media
        .as_ref()
        .is_some_and(|media| Arc::ptr_eq(media.tensor.tensor(), &owner));
    block_on(device.fence())?;
    let error = result.err().ok_or("short media tensor should fail")?;
    assert_eq!(error.message, "device copy range outside tensor");
    assert!(
        retained,
        "failed media chunk dropped the source before the completion fence"
    );
    Ok(())
}
