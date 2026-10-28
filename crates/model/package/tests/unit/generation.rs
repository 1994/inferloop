use super::*;
use crate::providers::qwen::QwenProvider;
use infer_spi::ModelProvider;

#[test]
fn mode_defaults_and_explicit_zero_overrides_are_distinct() -> Result<()> {
    let defaults = GenerationDefaults {
        base: Sampling {
            temperature: 1.0,
            top_p: 0.95,
            top_k: Some(20),
            eos_tokens: vec![10, 11],
            ..Sampling::default()
        },
        policy: QwenProvider
            .generation_policy(b"{}", Some("---\nbase_model:\n- Qwen/Qwen3.8-27B\n---"))?,
        sources: BTreeMap::new(),
    };
    let thinking = defaults.resolve(&SamplingOverrides::default())?;
    assert!(thinking.enable_thinking);
    assert_eq!(thinking.sampling.temperature, 1.0);
    let overrides = SamplingOverrides {
        enable_thinking: Some(false),
        ..SamplingOverrides::default()
    };
    let instruct = defaults.resolve(&overrides)?;
    assert_eq!(instruct.sampling.temperature, 0.7);
    assert_eq!(instruct.sampling.top_p, 0.8);
    assert_eq!(instruct.sampling.presence_penalty, 1.5);
    let custom = defaults.resolve(&SamplingOverrides {
        temperature: Some(0.0),
        top_k: Some(0),
        presence_penalty: Some(0.0),
        ..overrides
    })?;
    assert_eq!(custom.sampling.temperature, 0.0);
    assert_eq!(custom.sampling.top_k, None);
    assert_eq!(custom.sampling.presence_penalty, 0.0);
    assert!(custom.sampling.is_eos(11));
    assert!(
        defaults
            .resolve(&SamplingOverrides {
                top_p: Some(0.0),
                ..SamplingOverrides::default()
            })
            .is_err()
    );
    Ok(())
}

#[test]
fn every_resolved_parameter_records_the_layer_that_set_it() -> Result<()> {
    let defaults = GenerationDefaults {
        base: Sampling {
            temperature: 1.0,
            top_p: 0.95,
            ..Sampling::default()
        },
        policy: QwenProvider
            .generation_policy(b"{}", Some("---\nbase_model:\n- Qwen/Qwen3.8-27B\n---"))?,
        sources: BTreeMap::from([
            ("temperature".to_string(), "package default".to_string()),
            ("top_p".to_string(), "package default".to_string()),
        ]),
    };
    // The recognized model's non-thinking preset is recorded as its own source.
    let instruct = defaults.resolve(&SamplingOverrides {
        enable_thinking: Some(false),
        ..SamplingOverrides::default()
    })?;
    assert_eq!(
        instruct.sources.get("temperature").map(String::as_str),
        Some("Qwen3.8 model-card non-thinking recommendation")
    );
    // A request field wins and says so, rather than inheriting the preset's description.
    let overridden = defaults.resolve(&SamplingOverrides {
        temperature: Some(0.25),
        ..SamplingOverrides::default()
    })?;
    assert_eq!(overridden.sampling.temperature, 0.25);
    assert_eq!(
        overridden.sources.get("temperature").map(String::as_str),
        Some("request override")
    );
    // A parameter the request did not touch keeps the package's own description.
    assert_eq!(
        overridden.sources.get("top_p").map(String::as_str),
        Some("package default")
    );
    Ok(())
}

#[test]
fn qwen_generation_policy_is_shared_across_family_sizes() {
    for name in ["0.6B", "8B", "9B", "30B-A3B"] {
        let card = format!("---\nbase_model:\n- Qwen/Qwen3-{name}-Base\n---\n");
        let policy = QwenProvider.generation_policy(b"{}", Some(&card)).unwrap();
        assert!(policy.enable_thinking, "{name}");
        assert_eq!(policy.non_thinking.unwrap().temperature, Some(0.7));
    }
    for name in ["27B", "60B-A6B"] {
        for card in [
            format!("---\nbase_model:\n- Qwen/Qwen3.8-{name}\n---\n"),
            format!("---\nlibrary_name: transformers\n---\n# Qwen3.8-{name}\n"),
        ] {
            let policy = QwenProvider.generation_policy(b"{}", Some(&card)).unwrap();
            assert!(policy.enable_thinking, "{name}");
            assert_eq!(policy.non_thinking.unwrap().presence_penalty, Some(1.5));
        }
    }
    for name in ["Qwen3-VL-2B", "Qwen3-4B-Instruct-2507", "Qwen3-8B-Coder"] {
        let card = format!("---\nbase_model:\n- Qwen/{name}-Base\n---\n");
        assert!(
            !QwenProvider
                .generation_policy(b"{}", Some(&card))
                .unwrap()
                .enable_thinking,
            "{name}"
        );
    }
}
