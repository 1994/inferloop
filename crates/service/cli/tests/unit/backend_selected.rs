use super::*;
use std::path::Path;

fn selection(device: usize) -> Selection {
    Selection {
        kind: BackendChoice::Metal,
        device,
        num_gpu_blocks_override: None,
        block_size: None,
        max_num_batched_tokens: None,
        max_model_len: None,
        max_output_tokens: None,
        max_num_seqs: None,
        served_model_name: None,
        upload_staging_mib: None,
        num_speculative_tokens: Some(super::super::SpeculativeTokens::Depth(0)),
        draft_model: None,
        kv_cache_dtype: None,
        gpu_memory_utilization: 0.0,
        autotune: false,
        device_timing: infer_ir::DeviceTimingMode::Off,
        text_only: false,
        media_limits: infer_models::MediaLimits::default(),
        pool_capture: super::super::PoolCapture::OnDemand,
        #[cfg(cuda_backend)]
        tuned_pool_width: None,
    }
}

#[test]
fn metal_rejects_an_explicit_nonzero_device_before_touching_the_device() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let Err(error) = metal::load(&root, 0, &selection(1)) else {
        panic!("Metal must reject a non-zero device ordinal");
    };
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
}

#[test]
fn selected_backend_preserves_compact_generation_reservation() -> Result<()> {
    if !metal::available() {
        return Ok(());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    // A zero budget derives the share from the Metal device; the exact per-token kernel is used
    // because this fixture checks the reservation, not prefill.
    let mut backend = metal::load(&root, 0, &selection(0))?;
    let capacity = backend.model_ir().max_sequence;
    let full = backend
        .state_reservation_bytes_for(capacity, infer_ir::OutputReadout::Full)?
        .ok_or_else(|| Error::invariant("missing full budget"))?;
    let compact = backend
        .state_reservation_bytes_for(capacity, infer_ir::OutputReadout::Logits)?
        .ok_or_else(|| Error::invariant("missing compact budget"))?;
    assert_eq!(
        full - compact,
        ((capacity - 1) * backend.model_ir().hidden_size * size_of::<f32>()) as u64
    );
    backend.reserve_state_for(StateId::ONE, capacity, infer_ir::OutputReadout::Logits)?;
    assert_eq!(
        backend.execution_stats().map(|stats| stats.reserved_bytes),
        Some(compact)
    );
    backend.release_state(StateId::ONE)?;
    assert_eq!(
        backend.execution_stats().map(|stats| stats.reserved_bytes),
        Some(0)
    );
    Ok(())
}
