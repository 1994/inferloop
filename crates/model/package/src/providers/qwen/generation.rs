//! Qwen model-card generation recommendations.
use infer_spi::{GenerationPolicy, GenerationPreset, GreedyPreset};

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
            if line
                .strip_prefix("- Qwen/Qwen3-")
                .and_then(|name| name.strip_suffix("-Base"))
                .is_some_and(model_size)
            {
                return preset(QwenVariant::Hybrid);
            }
        }
    }
    // Original Qwen3.8 cards have no base_model field but identify themselves in the title.
    if body
        .lines()
        .map(str::trim)
        .any(|line| line.strip_prefix("# Qwen3.8-").is_some_and(model_size))
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
        sample_by_default: qwen38,
        non_thinking: Some(GenerationPreset {
            temperature: Some(NON_THINKING_TEMPERATURE),
            top_p: Some(NON_THINKING_TOP_P),
            presence_penalty: qwen38.then_some(QWEN38_PRESENCE_PENALTY),
            greedy: if qwen38 {
                GreedyPreset::Apply
            } else {
                GreedyPreset::Preserve
            },
            source: if qwen38 {
                "Qwen3.8 model-card non-thinking recommendation"
            } else {
                "Qwen3 hybrid model-card non-thinking recommendation"
            },
            ..GenerationPreset::default()
        }),
    }
}
