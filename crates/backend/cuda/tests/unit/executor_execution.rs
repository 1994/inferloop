use super::{candidate_pool_capacity, pool_retry_budget, pooled_envelope};
use crate::constants::CB_SLOT_TOKENS;
use infer_ir::OutputReadout;
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
