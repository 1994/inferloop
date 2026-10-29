use super::{append_encoded_media, candidate_pool_capacity, pool_retry_budget, pooled_envelope};
use crate::constants::CB_SLOT_TOKENS;
use crate::executor::CudaExecutor;
use infer_core::{RequestId, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, OutputReadout, TaskOutput, TokenBuffer};
use std::{env::var, error::Error};

#[test]
fn incompatible_requests_cannot_create_or_enlarge_a_decode_pool() {
    let full = (128, OutputReadout::Full);
    let long = (CB_SLOT_TOKENS + 1, OutputReadout::Logits);
    let compact = (128, OutputReadout::Logits);
    for states in [
        [full, full],
        [long, long],
        [full, long],
        [compact, full],
        [long, compact],
    ] {
        assert_eq!(candidate_pool_capacity(states.into_iter()), None);
    }
}

#[test]
fn compatible_pool_geometry_is_independent_of_unrelated_request_order() {
    let short = (128, OutputReadout::Logits);
    let tail = (129, OutputReadout::Logits);
    let full = (CB_SLOT_TOKENS, OutputReadout::Full);
    let long = (CB_SLOT_TOKENS + 1, OutputReadout::Logits);
    for states in [[short, full, tail, long], [long, tail, full, short]] {
        assert_eq!(candidate_pool_capacity(states.into_iter()), Some(256));
    }
    assert_eq!(candidate_pool_capacity([short].into_iter()), None);
    assert_eq!(candidate_pool_capacity([].into_iter()), None);
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
        executor::CudaExecutor,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{ModelId, StateId};
    use infer_ir::OutputReadout;

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
    // Start without a pool, then admit one long and two short private programs.
    let mut backend = CudaExecutor::new(loaded, budget, 1)?;
    backend.maximum_states = 16;
    let long = StateId::new(1)?;
    backend.reserve(long, 32768, OutputReadout::Logits, true)?;
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

#[test]
fn media_rows_reuse_the_first_buffer_and_keep_image_order() {
    let mut output = Vec::new();
    let first = vec![1.0, -2.0, 3.0, 4.0];
    let address = first.as_ptr();
    append_encoded_media(&mut output, first);
    assert_eq!(output.as_ptr(), address);
    append_encoded_media(&mut output, vec![5.0, 6.0]);
    append_encoded_media(&mut output, vec![7.0, 8.0]);
    assert_eq!(output, [1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn empty_media_rows_do_not_change_other_images() {
    let mut output = Vec::new();
    append_encoded_media(&mut output, Vec::new());
    append_encoded_media(&mut output, vec![1.0, 2.0]);
    append_encoded_media(&mut output, Vec::new());
    assert_eq!(output, [1.0, 2.0]);
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

fn pooled_position_reference(
    executor: &mut CudaExecutor,
    state: StateId,
    token: u32,
    rotary: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    use crate::device::asynchronous::block_on;
    let sequence = executor.states.get(&state).ok_or("missing sequence")?;
    let device = executor.loaded.device();
    let pool = executor.slots.as_mut().ok_or("missing pool")?;
    let slot = pool.take_free().ok_or("missing free slot")?;
    assert!(pool.assign(slot, sequence.capacity, device)?);
    pool.bind(
        slot,
        sequence.program.states(),
        sequence.program.fp8_states(),
        sequence.history.len(),
        device,
    )?;
    let output = block_on(pool.run_lanes_readout_async(
        device,
        &[(slot, token, rotary, sequence.history.len())],
        false,
    ))?;
    block_on(device.fence())?;
    pool.release(slot);
    let (_, logits) = output.rows.into_iter().next().ok_or("missing logits")?;
    Ok(logits)
}

fn media_position_fixture() -> Result<(CudaExecutor, TokenBuffer, [StateId; 3]), Box<dyn Error>> {
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
            mtp_depth: 0,
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let budget = loaded.device().allocatable_memory_bytes()?;
    let mut executor = CudaExecutor::new(loaded, budget, 3)?;
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
            pixels: Arc::from([1.0_f32]),
            grid: (1, 2, 2),
            span: 1..2,
        }],
    };
    let states = [StateId::ONE, StateId::new(2)?, StateId::new(3)?];
    for (index, &state) in states.iter().enumerate() {
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
    let (mut executor, prompt, states) = media_position_fixture()?;
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
    // Use the pooled graph directly with an independently supplied rotary position. This keeps
    // the full-logit comparison bitwise while exercising the executor's lane construction.
    let expected = pooled_position_reference(&mut executor, states[0], 65, 8)?;
    let single = run_media_position_tasks(&mut executor, &[decode(states[0], RequestId::ONE)])?;
    let batch = run_media_position_tasks(
        &mut executor,
        &[
            decode(states[1], RequestId::new(2)?),
            decode(states[2], RequestId::new(3)?),
        ],
    )?;
    assert!(executor.states[&states[0]].slot.is_none());
    assert!(executor.states[&states[1]].slot.is_some());
    assert!(executor.states[&states[2]].slot.is_some());
    for output in batch {
        assert_eq!(output.output.logits.len(), expected.len());
        let differences = output
            .output
            .logits
            .iter()
            .zip(&expected)
            .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
            .count();
        assert_eq!(
            differences, 0,
            "pooled logits differ from the positioned reference"
        );
    }
    let serial = &single[0].output.logits;
    assert_eq!(serial.len(), expected.len());
    let top = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
    };
    assert_eq!(top(serial), top(&expected));
    Ok(())
}
