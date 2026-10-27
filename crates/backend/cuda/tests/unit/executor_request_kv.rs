use super::*;
use crate::AttentionProfile;
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::{CudaDevice, asynchronous::block_on};
use crate::loading::{LoadOptions, LoadedModel};
use infer_core::{ModelId, RequestId, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, TokenBuffer};
use std::env::var;

fn fixture() -> Result<CudaExecutor> {
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL").map_err(|error| Error::invalid(error.to_string()))?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 0,
            mtp_depth: 2,
            autotune: false,
            attention_profile: AttentionProfile::Bf16,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let budget = loaded
        .device()
        .allocatable_memory_bytes()?
        .saturating_sub(loaded.device().profile()?.device_headroom_bytes())
        .saturating_sub(loaded.shared_scratch_bytes()?);
    CudaExecutor::new_automatic(loaded, budget, MAX_POOL_SLOTS)
}

fn step(backend: &mut CudaExecutor, state: StateId, tokens: ExecutionInput) -> Result<()> {
    backend.prepare_execution()?;
    let result = block_on(backend.execute(&[ExecutionTask {
        plain_greedy: false,
        request: RequestId::ONE,
        state,
        tokens,
        sampling: None,
        media: None,
    }]));
    let fence = block_on(backend.loaded.device().fence());
    backend.busy = false;
    fence?;
    let output = result?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].output.logits.len(), backend.model().vocab_size);
    assert!(
        output[0]
            .output
            .logits
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

fn committed_state(backend: &CudaExecutor, state: StateId) -> Result<Vec<Vec<u32>>> {
    backend.states[&state]
        .program
        .states()
        .values()
        .flatten()
        .map(|part| {
            backend
                .loaded
                .device()
                .read_borrowed(part)
                .map(|values| values.into_iter().map(f32::to_bits).collect())
        })
        .collect()
}

fn check_cold_quote(backend: &mut CudaExecutor) -> Result<usize> {
    assert!(backend.slots.is_none() && backend.worker.is_none());
    let initial = backend.execution_profile().concurrency.unwrap();
    assert!(initial.selected > 1);
    let cold = backend.kv_cache();
    assert_eq!(cold.pool_bytes, 0);
    let planned = backend.pool_quote.unwrap();
    assert!(cold.total_blocks >= planned.width * planned.capacity / KV_BLOCK_TOKENS);
    assert_eq!(cold.available_blocks, 0);
    assert_eq!(cold.block_size, KV_BLOCK_TOKENS);
    backend.prepare_state_reservation(384, OutputReadout::Logits)?;
    assert!(
        backend
            .state_reservation_bytes_for(384, OutputReadout::Logits)?
            .is_some()
    );
    assert!(backend.slots.is_none(), "a quote captured CUDA graphs");

    Ok(initial.selected)
}

/// Exercise the real owner-facing reservation path and complete live-state readbacks.
#[test]
#[ignore = "real CUDA/MTP2 checkpoint; run under safe-run with an explicit fixture"]
fn request_length_pool_preserves_live_state_and_restores_after_long_request() -> Result<()> {
    let mut backend = fixture()?;
    let short = StateId::ONE;
    let long = StateId::new(2)?;
    let initial = check_cold_quote(&mut backend)?;
    backend.reserve_state_for(short, 384, OutputReadout::Logits)?;
    let short_width = backend.slots.as_ref().unwrap().width();
    let (physical, free) = backend.slots.as_ref().unwrap().kv_pages();
    assert_eq!(physical - free, 12);
    assert_eq!(backend.kv_cache().available_blocks, free);
    assert_eq!(
        backend.state_page_growth(short)?.unwrap().allocated_pages,
        12
    );
    let prompt = TokenBuffer::from(vec![1, 2]);
    step(
        &mut backend,
        short,
        ExecutionInput::Prefill {
            span: prompt.span(0..prompt.len())?,
            readout: OutputReadout::Logits,
        },
    )?;
    assert_eq!(backend.kv_cache().active_blocks, 1);
    let before = committed_state(&backend, short)?;
    let quote = backend.pool_quote.unwrap();
    match backend.reserve_state_for(long, 4352, OutputReadout::Logits) {
        Ok(()) => backend.release_state(long)?,
        Err(error) => assert_eq!(error.code, ErrorCode::Capacity),
    }
    assert_eq!(backend.pool_quote.unwrap().width, quote.width);
    assert_eq!(committed_state(&backend, short)?, before);
    let duplicate = backend
        .reserve_state_for(short, 384, OutputReadout::Logits)
        .unwrap_err();
    assert_eq!(duplicate.code, ErrorCode::Conflict);
    assert_eq!(committed_state(&backend, short)?, before);
    step(
        &mut backend,
        short,
        ExecutionInput::Decode {
            token: 3,
            position: 2,
        },
    )?;
    backend.release_state(short)?;
    let (total, free) = backend.slots.as_ref().unwrap().kv_pages();
    assert_eq!(total, free);

    backend.reserve_state_for(long, 4352, OutputReadout::Logits)?;
    let long_width = backend.slots.as_ref().unwrap().width();
    assert!(long_width <= short_width);
    backend.release_state(long)?;
    backend.prepare_state_reservation(CB_SLOT_TOKENS + 1, OutputReadout::Logits)?;
    assert!(backend.slots.is_none());
    backend.reserve_state_for(long, CB_SLOT_TOKENS + 1, OutputReadout::Logits)?;
    assert!(backend.states[&long].home.is_none());
    backend.release_state(long)?;
    backend.prepare_state_reservation(384, OutputReadout::Logits)?;
    assert!(
        backend.slots.is_none(),
        "restore happened in the quote phase"
    );
    backend.reserve_state_for(short, 384, OutputReadout::Logits)?;
    let restored = backend.slots.as_ref().unwrap().width();
    assert!(backend.states[&short].home.is_some());
    backend.release_state(short)?;
    let (total, free) = backend.slots.as_ref().unwrap().kv_pages();
    assert_eq!(total, free);
    println!(
        "request KV widths: initial={initial}, short={short_width}, long={long_width}, restored={restored}"
    );
    Ok(())
}
