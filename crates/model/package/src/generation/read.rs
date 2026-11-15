use super::{GenerationDefaults, apply_preset};
use crate::constants::CONFIG_MAX_BYTES;
use crate::{ModelRegistry, PackageSource};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::Sampling;
use infer_spi::GenerationPolicy;
use std::collections::BTreeMap;

pub fn load(registry: &ModelRegistry, source: &PackageSource) -> Result<GenerationDefaults> {
    let config = config(source)?;
    let generation = source.generation_config()?;
    let mut base = Sampling::default();
    let mut sources = BTreeMap::new();
    let policy = generation_policy(registry, source, &config)?;
    for key in [
        "temperature",
        "top_p",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "top_k",
        "seed",
        "eos_tokens",
    ] {
        sources.insert(key.into(), "runtime fallback".into());
    }
    let sampling = generation.get("do_sample").map_or_else(
        || Ok(policy.sample_by_default),
        |v| {
            v.as_bool()
                .ok_or_else(|| Error::invalid("do_sample must be boolean"))
        },
    )?;
    base.temperature = if sampling { 1.0 } else { 0.0 };
    if let Some(defaults) = &policy.sampling_defaults {
        apply_preset(&mut base, &mut sources, defaults);
    }
    for (key, target) in [
        ("temperature", &mut base.temperature),
        ("top_p", &mut base.top_p),
        ("min_p", &mut base.min_p),
        ("presence_penalty", &mut base.presence_penalty),
        ("repetition_penalty", &mut base.repetition_penalty),
    ] {
        if let Some(value) = generation.get(key) {
            *target =
                serde_json::from_value(value.clone()).map_err(|e| Error::invalid(e.to_string()))?;
            sources.insert(key.into(), "generation_config.json".into());
        }
    }
    if !sampling {
        base.temperature = 0.0;
        if generation.get("do_sample").is_some() {
            sources.insert(
                "temperature".into(),
                "generation_config.json do_sample=false".into(),
            );
        }
    }
    if let Some(value) = generation.get("top_k") {
        let number = value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| Error::invalid("invalid top_k"))?;
        base.top_k = (number != 0).then_some(number);
        sources.insert("top_k".into(), "generation_config.json".into());
    }
    let config = config.get("text_config").unwrap_or(&config);
    if let Some(eos) = generation
        .get("eos_token_id")
        .or_else(|| config.get("eos_token_id"))
    {
        let values = eos
            .as_array()
            .map_or_else(|| vec![eos], |a| a.iter().collect());
        for value in values {
            base.eos_tokens.push(
                value
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| Error::invalid("invalid EOS token"))?,
            );
        }
        sources.insert(
            "eos_tokens".into(),
            if generation.get("eos_token_id").is_some() {
                "generation_config.json"
            } else {
                "config.json"
            }
            .into(),
        );
    }
    base.validate()?;
    Ok(GenerationDefaults {
        base,
        policy,
        sources,
    })
}

fn generation_policy(
    registry: &ModelRegistry,
    source: &PackageSource,
    config: &serde_json::Value,
) -> Result<GenerationPolicy> {
    let card = source.readme()?;
    let config_bytes = serde_json::to_vec(config).map_err(|e| Error::invalid(e.to_string()))?;
    match registry.resolve(&config_bytes) {
        Ok(provider) => provider.generation_policy(&config_bytes, card),
        Err(error) if error.code == ErrorCode::Unsupported => Ok(GenerationPolicy::default()),
        Err(error) => Err(error),
    }
}

/// Parse the package configuration, treating absence as an empty object.
fn config(source: &PackageSource) -> Result<serde_json::Value> {
    let Some(bytes) = source.config() else {
        return Ok(serde_json::json!({}));
    };
    if bytes.len() as u64 > CONFIG_MAX_BYTES {
        return Err(Error::new(
            ErrorCode::Capacity,
            "package metadata exceeds budget",
        ));
    }
    serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string()))
}
