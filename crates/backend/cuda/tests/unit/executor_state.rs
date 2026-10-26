use super::private_funding;
use crate::constants::CB_DECODE_SLOTS;
use crate::device::CudaDevice;
use crate::executor::{bound_concurrency, funded_width, knee_concurrency, recommendation};
use std::env;

#[test]
fn private_request_cannot_be_admitted_without_its_shared_workspace() {
    // The reproduced long request fits the logical state budget but not physical
    // storage after the missing shared prompt workspace is included.
    let needed = 2_683_745_676;
    let (free, capacity) = private_funding(2_886_559_264, 0, 4_789_067_288, 0, 2_300_000_000);
    assert!(needed < 2_886_559_264);
    assert!(needed > free);
    assert!(needed > capacity);
}

#[test]
fn private_capacity_preserves_live_state_and_credits_reclaimable_storage() {
    let (free, capacity) = private_funding(100, 40, 50, 20, 30);
    assert_eq!((free, capacity), (40, 80));
    // Once shared workspace is allocated, it is no longer charged as pending.
    assert_eq!(private_funding(100, 40, 50, 20, 0), (60, 100));
}

#[test]
fn private_capacity_saturates_when_workspace_cannot_fit() {
    assert_eq!(private_funding(100, 0, 20, 0, 30), (0, 0));
    assert_eq!(
        private_funding(u64::MAX, u64::MAX, u64::MAX, 1, 0),
        (0, u64::MAX)
    );
}

#[test]
#[ignore = "real CUDA/MTP checkpoint; reserve graphs only after quote acknowledgement"]
fn quotes_do_not_recapture_a_retired_pool_and_reservation_restores_it() -> infer_core::Result<()> {
    use crate::{
        AttentionProfile,
        constants::{CB_DECODE_SLOTS, CB_SLOT_TOKENS},
        device::CudaDevice,
        executor::CudaExecutor,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{Error, ModelId, StateId};
    use infer_ir::OutputReadout;
    use std::env::var;

    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?,
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
    let mut backend = CudaExecutor::new(loaded, budget, CB_DECODE_SLOTS)?;
    assert!(backend.slots.is_some());
    backend.prepare_state_reservation(CB_SLOT_TOKENS + 1, OutputReadout::Logits)?;
    assert!(backend.slots.is_none());
    assert_eq!(backend.reclaimed_slot_capacity, Some(CB_SLOT_TOKENS));
    backend.prepare_state_reservation(512, OutputReadout::Logits)?;
    let quoted = backend
        .state_reservation_bytes_for(512, OutputReadout::Logits)?
        .ok_or_else(|| Error::invariant("missing private quote"))?;
    assert!(quoted > 0);
    assert!(
        backend.slots.is_none(),
        "admission quotes must not capture a pool"
    );
    assert!(backend.homes.is_empty() && backend.worker.is_none());
    backend.reserve_state_for(StateId::ONE, 512, OutputReadout::Logits)?;
    assert!(backend.slots.is_some() && backend.worker.is_some());
    assert!(backend.states[&StateId::ONE].home.is_some());
    assert_eq!(backend.states[&StateId::ONE].budget, 0);
    assert!(!backend.slots_disabled);
    backend.release_state(StateId::ONE)?;
    Ok(())
}

#[test]
fn automatic_concurrency_is_the_smallest_real_bound() {
    assert_eq!(bound_concurrency(4, None, 64), 4);
    assert_eq!(bound_concurrency(64, Some(7), 64), 7);
    assert_eq!(bound_concurrency(64, None, 5), 5);
    assert_eq!(bound_concurrency(64, None, 64), CB_DECODE_SLOTS);
    assert_eq!(bound_concurrency(0, None, 0), 0);
}

#[test]
fn funded_width_never_pretends_an_unfunded_sequence() -> infer_core::Result<()> {
    let fixed = 20_u64;
    let per_sequence = 10_u64;
    let budget = 100_u64;
    let width = funded_width(16, |width| {
        let bytes = fixed + (width as u64) * per_sequence;
        Ok((bytes <= budget).then_some(bytes))
    })?;
    assert_eq!(width, 8);
    assert!(fixed + (width as u64) * per_sequence <= budget);
    assert!(fixed + ((width + 1) as u64) * per_sequence > budget);
    assert_eq!(funded_width(16, |_| Ok(None))?, 0);
    // Nonlinear retained tables: a padding step at width 5 changes the funded maximum.
    assert_eq!(
        funded_width(16, |width| {
            let bytes = fixed + (width as u64) * per_sequence + u64::from(width >= 5) * 40;
            Ok((bytes <= budget).then_some(bytes))
        })?,
        4
    );
    // Width 15/16 must be checked individually, not extrapolated from 1/2.
    assert_eq!(funded_width(16, |width| Ok((width < 15).then_some(1)))?, 14);
    assert_eq!(
        funded_width(16, |width| Ok((width <= 16).then_some(1)))?,
        16
    );
    Ok(())
}

#[test]
fn knee_formula_uses_the_documented_roofline_units() -> infer_core::Result<()> {
    // (100/10) * (50/10) / 2 / 1 = 25.
    assert_eq!(knee_concurrency(100, 10, 50, 10, 1)?, Some(25));
    assert_eq!(knee_concurrency(0, 10, 50, 10, 1)?, None);
    assert_eq!(knee_concurrency(100, 0, 50, 10, 1)?, None);
    assert_eq!(knee_concurrency(100, 10, 0, 10, 1)?, None);
    assert_eq!(knee_concurrency(100, 10, 50, 0, 1)?, None);
    assert_eq!(knee_concurrency(100, 10, 50, 10, 0)?, None);
    assert!(knee_concurrency(u64::MAX, 1, u64::MAX, 1, 1).is_err());
    assert!(knee_concurrency(1, u64::MAX, 1, u64::MAX, u64::MAX).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA and INFER_TEST_MODEL; real 27B concurrency calibration, safe-run"]
fn actual_model_recommendation_reports_raw_knee_inputs() -> infer_core::Result<()> {
    use crate::loading::{LoadOptions, LoadedModel};
    let path =
        env::var("INFER_TEST_MODEL").map_err(|e| infer_core::Error::invalid(e.to_string()))?;
    let device = CudaDevice::new(0)?;
    let free = device.allocatable_memory_bytes()?;
    let loaded = LoadedModel::open(
        device,
        path,
        infer_core::ModelId::ONE,
        LoadOptions {
            prefill_width: 0,
            mtp_depth: 2,
            ..LoadOptions::default()
        },
    )?;
    let used = free
        .saturating_sub(loaded.device().allocatable_memory_bytes()?)
        .saturating_add(loaded.shared_scratch_bytes()?);
    let budget = free
        .checked_sub(used)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| infer_core::Error::invalid("model leaves no state budget"))?;
    let metrics = loaded.weight_metrics()?;
    let profile = loaded.profile().clone();
    let maximum_states = profile.resident_states();
    let recommendation = recommendation(&loaded, budget, maximum_states)?;
    println!(
        "{}",
        serde_json::json!({
            "device": profile.name,
            "architecture": profile.architecture,
            "multiprocessors": profile.multiprocessors,
            "memory_clock_khz": profile.memory_clock_khz,
            "memory_bus_width_bits": profile.memory_bus_width_bits,
            "memory_bandwidth_bytes_per_second": profile.memory_bandwidth_bytes_per_second(),
            "state_budget": budget,
            "weight_bytes": metrics.bytes,
            "parameters": metrics.parameters,
            "tokens_per_step": 3,
            "memory_bound": recommendation.memory_bound,
            "knee_bound": recommendation.knee_bound,
            "max_sequences": recommendation.max_sequences,
            "legal_cap": CB_DECODE_SLOTS.min(infer_gpu_api::MAX_SUBMISSION_BATCH),
        })
    );
    Ok(())
}
