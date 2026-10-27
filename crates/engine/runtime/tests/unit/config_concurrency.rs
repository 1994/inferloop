use super::*;
use infer_ir::{ConcurrencyInspection, ExecutionProfileInspection};
fn profile(memory: usize, roofline: Option<usize>, selected: usize) -> ExecutionProfileInspection {
    ExecutionProfileInspection {
        concurrency: Some(ConcurrencyInspection {
            memory_bound: memory,
            roofline_bound: roofline,
            selected,
        }),
        prefill_width: 64,
        pooled_prefill_width: Some(64),
        batch_width: 3,
        draft_depth: 2,
        automatic_prefill: true,
        arena_budget_bytes: 1,
        attention_profile: "bf16".into(),
        precise_prefix_positions: Some(0),
    }
}
#[test]
fn backend_funding_resolves_defaults_beyond_the_old_batch_limit() {
    let mut config = RuntimeConfig::default();
    config
        .resolve_concurrency(Some(&profile(48, Some(32), 32)))
        .unwrap();
    assert_eq!(config.max_num_seqs, 32);
    config.validate().unwrap();
    let mut smaller = RuntimeConfig::default();
    smaller
        .resolve_concurrency(Some(&profile(7, Some(32), 7)))
        .unwrap();
    assert_eq!(smaller.max_num_seqs, 7);
}
#[test]
fn explicit_smaller_caps_and_serial_backends_remain_bounded() {
    let mut config = RuntimeConfig {
        max_num_seqs: 4,
        ..RuntimeConfig::default()
    };
    config
        .resolve_concurrency(Some(&profile(48, Some(32), 32)))
        .unwrap();
    assert_eq!(config.max_num_seqs, 4);
    config
        .resolve_concurrency(Some(&profile(1, Some(1), 1)))
        .unwrap();
    assert_eq!(config.max_num_seqs, 1);
}
#[test]
fn malformed_backend_bounds_and_invalid_explicit_caps_do_not_publish() {
    for candidate in [
        profile(4, Some(3), 4),
        profile(0, None, 1),
        profile(8, Some(0), 1),
        profile(8, None, 0),
    ] {
        let mut config = RuntimeConfig::default();
        let original = config.clone();
        assert!(config.resolve_concurrency(Some(&candidate)).is_err());
        assert_eq!(config, original);
    }
    for cap in [0, infer_gpu_api::MAX_SUBMISSION_BATCH + 1] {
        let mut config = RuntimeConfig {
            max_num_seqs: cap,
            ..RuntimeConfig::default()
        };
        assert!(
            config
                .resolve_concurrency(Some(&profile(4, None, 4)))
                .is_err()
        );
    }
}
#[test]
fn old_profile_json_has_no_backend_recommendation_and_keeps_explicit_limit() {
    let mut old = serde_json::to_value(profile(8, Some(8), 8)).unwrap();
    old.as_object_mut().unwrap().remove("concurrency");
    let restored: ExecutionProfileInspection = serde_json::from_value(old).unwrap();
    assert!(restored.concurrency.is_none());
    let mut config = RuntimeConfig {
        max_num_seqs: 6,
        ..RuntimeConfig::default()
    };
    config.resolve_concurrency(Some(&restored)).unwrap();
    assert_eq!(config.max_num_seqs, 6);
}
