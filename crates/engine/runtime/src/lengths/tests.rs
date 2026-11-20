use super::*;

fn config(
    max_model_len: Option<usize>,
    max_output_tokens: Option<usize>,
    max_input_tokens: usize,
) -> RuntimeConfig {
    RuntimeConfig {
        max_model_len,
        max_output_tokens,
        max_input_tokens,
        ..Default::default()
    }
}

#[test]
fn an_unset_deployment_limit_follows_the_model_ceiling() {
    let limits = ResolvedLengthLimits::resolve(4096, &config(None, None, 65536)).unwrap();
    assert_eq!(limits.model_limit, 4096);
    assert_eq!(limits.total, 4096);
    assert_eq!(limits.input_cap, 4096);
    assert_eq!(limits.output_cap, 4096);
    assert_eq!(limits.sources.total, LimitSource::Model);
    assert_eq!(limits.sources.input_cap, LimitSource::Model);
    assert_eq!(limits.sources.output_cap, LimitSource::Default);
}

#[test]
fn an_explicit_total_context_shrinks_the_input_and_output_caps() {
    let limits =
        ResolvedLengthLimits::resolve(262_144, &config(Some(8192), Some(1024), 65536)).unwrap();
    assert_eq!(limits.total, 8192);
    assert_eq!(limits.input_cap, 8192);
    assert_eq!(limits.output_cap, 1024);
    assert_eq!(limits.sources.total, LimitSource::Service);
    assert_eq!(limits.sources.output_cap, LimitSource::Service);
}

#[test]
fn an_input_cap_below_the_total_context_is_reported_as_the_service_cap() {
    let limits = ResolvedLengthLimits::resolve(262_144, &config(None, None, 4096)).unwrap();
    assert_eq!(limits.total, 262_144);
    assert_eq!(limits.input_cap, 4096);
    assert_eq!(limits.sources.input_cap, LimitSource::Service);
}

#[test]
fn unattainable_or_zero_explicit_limits_are_rejected_not_shrunk() {
    for config in [
        config(Some(4097), None, 65536),
        config(Some(0), None, 65536),
        config(None, Some(0), 65536),
        RuntimeConfig {
            max_model_len: Some(64),
            max_output_tokens: Some(128),
            ..Default::default()
        },
    ] {
        assert!(
            ResolvedLengthLimits::resolve(4096, &config).is_err(),
            "{config:?}"
        );
    }
    assert!(ResolvedLengthLimits::resolve(0, &config(None, None, 65536)).is_err());
}

#[test]
fn a_request_fits_only_when_prompt_plus_output_stays_inside_the_total() {
    let limits = ResolvedLengthLimits::resolve(64, &config(Some(8), Some(4), 8)).unwrap();
    assert!(limits.check(4, 4).is_ok());
    assert!(limits.check(5, 4).is_err());
    assert!(limits.check(4, 5).is_err());
    assert!(limits.check(0, 1).is_err());
    // Checked arithmetic stays defensive even though both caps already bound the sum.
    assert!(limits.check(usize::MAX, 1).is_err());
}

#[test]
fn the_remaining_output_budget_respects_the_cap_and_the_total() {
    let limits = ResolvedLengthLimits::resolve(64, &config(Some(16), Some(4), 16)).unwrap();
    assert_eq!(limits.remaining_output(1).unwrap(), 4);
    assert_eq!(limits.remaining_output(14).unwrap(), 2);
    assert!(limits.remaining_output(16).is_err());
}

#[test]
fn default_prompt_cap_tracks_large_model_contexts() -> Result<()> {
    for model_limit in [65_537, 131_072, 262_144, 1_048_576] {
        let limits = ResolvedLengthLimits::resolve(model_limit, &RuntimeConfig::default())?;
        assert_eq!(limits.total, model_limit);
        assert_eq!(limits.input_cap, model_limit);
        assert_eq!(limits.sources.input_cap, LimitSource::Model);
        limits.check(model_limit - 256, 256)?;
        assert!(limits.check(model_limit, 1).is_err());
    }
    Ok(())
}
