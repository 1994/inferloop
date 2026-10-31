use super::*;

#[test]
fn cuda_round_funds_a_full_prompt_chunk_beside_decodes() {
    let mut config = RuntimeConfig {
        max_num_seqs: 16,
        max_num_batched_tokens: 64,
        ..RuntimeConfig::default()
    };
    cuda_prefill_limits(&mut config, 1024, true);
    assert_eq!(config.scheduler.prefill_chunk_tokens, 1024);
    assert_eq!(config.max_num_batched_tokens, 1039);
    assert_eq!(config.max_num_seqs, 16);
}

#[test]
fn cuda_round_preserves_a_larger_budget_and_single_sequence_geometry() {
    let mut config = RuntimeConfig {
        max_num_seqs: 16,
        max_num_batched_tokens: 4096,
        ..RuntimeConfig::default()
    };
    cuda_prefill_limits(&mut config, 1024, true);
    assert_eq!(config.max_num_batched_tokens, 4096);
    config.max_num_seqs = 1;
    config.max_num_batched_tokens = 64;
    cuda_prefill_limits(&mut config, 1024, true);
    assert_eq!(config.max_num_batched_tokens, 1024);
}

#[test]
fn pooled_worker_reporting_does_not_reduce_private_prompt_chunks() {
    let profile = infer_ir::ExecutionProfileInspection {
        concurrency: None,
        prefill_width: 2048,
        pooled_prefill_width: Some(1024),
        batch_width: 3,
        draft_depth: 2,
        automatic_prefill: true,
        arena_budget_bytes: 0,
        attention_profile: "bf16".into(),
        precise_prefix_positions: Some(2048),
    };
    let mut config = RuntimeConfig {
        max_num_seqs: 16,
        ..RuntimeConfig::default()
    };
    cuda_prefill_limits(
        &mut config,
        profile.prefill_width,
        profile.automatic_prefill,
    );
    assert_eq!(config.scheduler.prefill_chunk_tokens, 2048);
    assert_eq!(config.max_num_batched_tokens, 2063);
}

#[test]
fn explicit_cuda_token_limit_does_not_gain_decode_headroom() {
    let mut config = RuntimeConfig {
        max_num_seqs: 16,
        max_num_batched_tokens: 1024,
        ..RuntimeConfig::default()
    };
    cuda_prefill_limits(&mut config, 1024, false);
    assert_eq!(config.max_num_batched_tokens, 1024);
}

#[test]
fn automatic_round_funds_one_prompt_segment_per_sequence() {
    let mut config = RuntimeConfig {
        max_num_seqs: 16,
        ..RuntimeConfig::default()
    };
    cuda_prefill_limits(&mut config, 128, true);
    assert_eq!(config.max_num_batched_tokens, 16 * 64 + 15);
    let mut explicit = RuntimeConfig {
        max_num_seqs: 16,
        ..RuntimeConfig::default()
    };
    cuda_prefill_limits(&mut explicit, 128, false);
    assert_eq!(explicit.max_num_batched_tokens, 128);
}
