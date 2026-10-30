use super::private_funding;
use infer_core::ErrorCode;

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
#[ignore = "real Qwen3-VL checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn repeated_media_reservations_do_not_accumulate_transient_bytes() -> infer_core::Result<()> {
    use crate::{
        device::CudaDevice,
        executor::{CachedMedia, CudaExecutor},
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{Error, ModelId, StateId};
    use infer_ir::{MediaItem, MediaPlan, Modality, OutputReadout, TokenBuffer};
    use infer_models::MediaLayout;
    use std::{env::var, sync::Arc};

    const GRID_SIDE: usize = 8;
    const REUSES: u64 = 3;
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL").map_err(|error| Error::invalid(error.to_string()))?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 32,
            mtp_depth: 0,
            autotune: false,
            vision: true,
            ..LoadOptions::default()
        },
    )?;
    let encoder = loaded
        .vision()
        .ok_or_else(|| Error::invalid("test model has no vision encoder"))?
        .encoder();
    let patch_width = 3 * encoder.temporal_patch_size * encoder.patch_size * encoder.patch_size;
    let rows = GRID_SIDE * GRID_SIDE / (encoder.spatial_merge_size * encoder.spatial_merge_size);
    let plan = MediaPlan {
        modality: Modality::Image,
        prompt: TokenBuffer::from(vec![0; rows]),
        items: vec![MediaItem {
            modality: Modality::Image,
            pixels: vec![0.0; GRID_SIDE * GRID_SIDE * patch_width].into(),
            grid: (1, GRID_SIDE, GRID_SIDE),
            span: 0..rows,
        }],
    };
    let budget = loaded.device().allocatable_memory_bytes()?;
    let mut executor = CudaExecutor::new_bounded(loaded, budget, 2)?;
    let initial = executor
        .state_reservation_bytes_for_media(128, OutputReadout::Logits, &plan)?
        .ok_or_else(|| Error::invariant("missing media quote"))?;
    for id in 1..=REUSES {
        let state = StateId::new(id)?;
        executor.reserve_media(state, 128, OutputReadout::Logits, &plan)?;
        let tensor = executor.loaded.device().upload(vec![0.0_f32], &[1])?;
        let reader = Arc::downgrade(&tensor);
        executor
            .states
            .get_mut(&state)
            .ok_or_else(|| Error::invariant("missing media state"))?
            .media = Some(CachedMedia {
            tensor,
            layout: MediaLayout {
                spans: Vec::new(),
                next_rotary: 0,
                positions: None,
            },
            bases: Vec::new(),
        });
        executor.release(state)?;
        assert!(reader.upgrade().is_none(), "cached shell retained media");
        let next = executor
            .state_reservation_bytes_for_media(128, OutputReadout::Logits, &plan)?
            .ok_or_else(|| Error::invariant("missing media quote"))?;
        assert_eq!(next, initial, "media quote grew after state was released");
    }
    Ok(())
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
    assert!(backend.slots.is_none());
    let waiting = StateId::new(2)?;
    let error = backend
        .reserve_state_for(waiting, 512, OutputReadout::Logits)
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Capacity);
    backend.release_state(StateId::ONE)?;
    backend.reserve_state_for(waiting, 512, OutputReadout::Logits)?;
    assert!(backend.slots.is_some() && backend.worker.is_some());
    assert!(backend.states[&waiting].home.is_some());
    assert_eq!(backend.states[&waiting].budget, 0);
    assert!(!backend.slots_disabled);
    backend.release_state(waiting)?;
    Ok(())
}
