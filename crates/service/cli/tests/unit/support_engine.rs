use super::apply_recommendation;
use infer_runtime::RuntimeConfig;
use infer_spi::SchedulingRecommendation;

fn recommendation(max_sequences: usize) -> SchedulingRecommendation {
    SchedulingRecommendation {
        max_sequences,
        memory_bound: 3,
        knee_bound: None,
    }
}

#[test]
fn automatic_recommendation_replaces_the_legacy_default() {
    let mut config = RuntimeConfig::default();
    apply_recommendation(&mut config, false, Some(recommendation(3)));
    assert_eq!(config.max_num_seqs, 3);
}

#[test]
fn explicit_max_num_seqs_is_never_overwritten() {
    let mut config = RuntimeConfig {
        max_num_seqs: 7,
        ..RuntimeConfig::default()
    };
    apply_recommendation(&mut config, true, Some(recommendation(3)));
    assert_eq!(config.max_num_seqs, 7);
}

#[test]
fn absent_recommendation_keeps_the_portable_default() {
    let mut config = RuntimeConfig::default();
    apply_recommendation(&mut config, false, None);
    assert_eq!(config.max_num_seqs, 16);
}

#[test]
fn automatic_knee_bound_is_consumed_but_explicit_override_wins() {
    let mut rec = recommendation(5);
    rec.knee_bound = Some(3);
    let mut config = RuntimeConfig::default();
    apply_recommendation(&mut config, false, Some(rec));
    assert_eq!(config.max_num_seqs, 5);
    let mut explicit = RuntimeConfig {
        max_num_seqs: 7,
        max_num_seqs_explicit: true,
        ..RuntimeConfig::default()
    };
    let mut rec = recommendation(5);
    rec.knee_bound = Some(3);
    apply_recommendation(&mut explicit, true, Some(rec));
    assert_eq!(explicit.max_num_seqs, 7);
}
