use super::{GenerationDefaults, ModelProfile};
use crate::constants::CONFIG_MAX_BYTES;
use crate::constants::MIB_U64;
use crate::{package_path, storage::package::read_bounded};
use infer_core::{Error, Result};
use infer_ir::Sampling;
use std::{collections::BTreeMap, path::Path};

pub fn load(root: &Path) -> Result<GenerationDefaults> {
    let config = json(root, "config.json")?;
    let generation = json(root, "generation_config.json")?;
    let mut base = Sampling::default();
    let mut sources = BTreeMap::new();
    let card = root.join("README.md");
    let profile = if card.exists() {
        let bytes = read_bounded(&package_path(root, "README.md")?, 2 * MIB_U64)?;
        let card = String::from_utf8(bytes).map_err(|e| Error::invalid(e.to_string()))?;
        let front = card.strip_prefix("---").and_then(|s| s.split_once("---"));
        front.map_or(ModelProfile::Other, |(front, _)| profile(front))
    } else {
        ModelProfile::Other
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
        || Ok(profile == ModelProfile::Qwen38),
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
        thinking: profile != ModelProfile::Other,
        profile,
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

fn profile(front: &str) -> ModelProfile {
    let mut base_model = false;
    for line in front.lines().map(str::trim) {
        if line == "base_model:" {
            base_model = true;
            continue;
        }
        if !line.starts_with("- ") && !line.is_empty() {
            base_model = false;
        }
        match line {
            "- Qwen/Qwen3.8-27B" => return ModelProfile::Qwen38,
            "- Qwen/Qwen3-0.6B-Base"
            | "- Qwen/Qwen3-1.7B-Base"
            | "- Qwen/Qwen3-4B-Base"
            | "- Qwen/Qwen3-8B-Base"
            | "- Qwen/Qwen3-14B-Base"
            | "- Qwen/Qwen3-32B-Base"
            | "- Qwen/Qwen3-30B-A3B-Base"
            | "- Qwen/Qwen3-235B-A22B-Base"
                if base_model =>
            {
                return ModelProfile::Qwen3Hybrid;
            }
            _ => {}
        }
    }
    ModelProfile::Other
}
