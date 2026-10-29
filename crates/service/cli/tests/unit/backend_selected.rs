use super::*;
use std::path::Path;
#[test]
fn selected_backend_preserves_compact_generation_reservation() -> Result<()> {
    if !metal_available() {
        return Ok(());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut backend = metal::load(
        &root,
        // A zero budget derives the share from the Metal device; the CPU test-backend
        0,
        &Selection {
            kind: BackendChoice::Metal,
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
            gpu_memory_utilization: 0.0,
            autotune: false,
            text_only: false,
            media_limits: infer_models::MediaLimits::default(),
            pool_capture: super::super::PoolCapture::OnDemand,
            // The exact per-token kernel; this fixture checks the Metal reservation, not prefill.
        },
    )?;
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
