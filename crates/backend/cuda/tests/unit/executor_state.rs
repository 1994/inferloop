use super::private_funding;

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
