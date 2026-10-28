//! Qwen model-card generation recommendations.
use infer_spi::{GenerationPolicy, GenerationPreset};

/// Recommended thinking temperature for original Qwen3 hybrid models.
const QWEN3_THINKING_TEMPERATURE: f32 = 0.6;
/// Recommended thinking nucleus probability for Qwen3 and Qwen3.8.
const THINKING_TOP_P: f32 = 0.95;
/// Recommended top-k for Qwen3 and Qwen3.8 generation.
const QWEN_TOP_K: usize = 20;
/// Recommended non-thinking temperature for Qwen3 and Qwen3.8.
const NON_THINKING_TEMPERATURE: f32 = 0.7;
/// Recommended non-thinking nucleus probability for Qwen3 and Qwen3.8.
const NON_THINKING_TOP_P: f32 = 0.8;
/// Recommended non-thinking presence penalty for Qwen3.8.
const QWEN38_PRESENCE_PENALTY: f32 = 1.5;

pub(super) fn policy(card: Option<&str>) -> GenerationPolicy {
    let Some(card) = card else {
        return GenerationPolicy::default();
    };
    let Some((front, body)) = card
        .strip_prefix("---")
        .and_then(|card| card.split_once("---"))
    else {
        return GenerationPolicy::default();
    };
    let mut base_model = false;
    for line in front.lines().map(str::trim) {
        if line == "base_model:" {
            base_model = true;
            continue;
        }
        if !line.starts_with("- ") && !line.is_empty() {
            base_model = false;
        }
        if base_model {
            if line.strip_prefix("- Qwen/Qwen3.8-").is_some_and(model_size) {
                return preset(QwenVariant::Qwen38);
            }
            if line.strip_prefix("- Qwen/Qwen3-").is_some_and(model_size) {
                return preset(QwenVariant::Hybrid);
            }
        }
    }
    // Original cards identify the actual chat model in the first title. A base-model lineage
    // alone is insufficient: embedding and reranking models may share the same base.
    let title = body
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("# "));
    if title
        .and_then(|title| title.strip_prefix("Qwen3-"))
        .is_some_and(model_size)
    {
        return preset(QwenVariant::Hybrid);
    }
    if title
        .and_then(|title| title.strip_prefix("Qwen3.8-"))
        .is_some_and(model_size)
    {
        return preset(QwenVariant::Qwen38);
    }
    GenerationPolicy::default()
}

// A Qwen size may be dense (e.g. 8B) or an MoE total/active pair (e.g. 30B-A3B).
fn model_size(name: &str) -> bool {
    match name.split_once("-A") {
        Some((total, active)) => size(total) && size(active),
        None => size(name),
    }
}

fn size(name: &str) -> bool {
    let Some(number) = name.strip_suffix("B") else {
        return false;
    };
    let mut parts = number.split('.');
    let first = parts.next().unwrap_or_default();
    !first.is_empty()
        && first.bytes().all(|byte| byte.is_ascii_digit())
        && parts
            .next()
            .is_none_or(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        && parts.next().is_none()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QwenVariant {
    Hybrid,
    Qwen38,
}

fn preset(variant: QwenVariant) -> GenerationPolicy {
    let qwen38 = variant == QwenVariant::Qwen38;
    GenerationPolicy {
        enable_thinking: true,
        sample_by_default: true,
        sampling_defaults: Some(GenerationPreset {
            temperature: Some(if qwen38 {
                1.0
            } else {
                QWEN3_THINKING_TEMPERATURE
            }),
            top_k: Some(QWEN_TOP_K),
            top_p: Some(THINKING_TOP_P),
            source: if qwen38 {
                "Qwen3.8 model-card sampling fallback"
            } else {
                "Qwen3 hybrid model-card sampling fallback"
            },
            ..GenerationPreset::default()
        }),
        non_thinking: Some(GenerationPreset {
            temperature: Some(NON_THINKING_TEMPERATURE),
            top_p: Some(NON_THINKING_TOP_P),
            presence_penalty: qwen38.then_some(QWEN38_PRESENCE_PENALTY),
            source: if qwen38 {
                "Qwen3.8 model-card non-thinking recommendation"
            } else {
                "Qwen3 hybrid model-card non-thinking recommendation"
            },
            ..GenerationPreset::default()
        }),
    }
}
