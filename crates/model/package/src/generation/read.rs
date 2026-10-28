use super::GenerationDefaults;
use crate::ModelRegistry;
use crate::constants::CONFIG_MAX_BYTES;
use crate::constants::MIB_U64;
use crate::{package_path, storage::package::read_bounded};
use infer_core::{Error, ErrorCode, Result};
use infer_ir::Sampling;
use infer_spi::GenerationPolicy;
use std::{collections::BTreeMap, path::Path};

pub fn load(root: &Path, registry: &ModelRegistry) -> Result<GenerationDefaults> {
    let config = json(root, "config.json")?;
    let generation = json(root, "generation_config.json")?;
    let mut base = Sampling::default();
    let mut sources = BTreeMap::new();
    let card = if root.join("README.md").exists() {
        let bytes = read_bounded(&package_path(root, "README.md")?, 2 * MIB_U64)?;
        Some(String::from_utf8(bytes).map_err(|e| Error::invalid(e.to_string()))?)
    } else {
        None
    };
    let config_bytes = serde_json::to_vec(&config).map_err(|e| Error::invalid(e.to_string()))?;
    let policy = match registry.resolve(&config_bytes) {
        Ok(provider) => provider.generation_policy(&config_bytes, card.as_deref())?,
        Err(error) if error.code == ErrorCode::Unsupported => GenerationPolicy::default(),
        Err(error) => return Err(error),
    };
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

fn json(root: &Path, name: &str) -> Result<serde_json::Value> {
    if !root.join(name).exists() {
        return Ok(serde_json::json!({}));
    }
    let bytes = read_bounded(&package_path(root, name)?, CONFIG_MAX_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|e| Error::invalid(e.to_string()))
}
