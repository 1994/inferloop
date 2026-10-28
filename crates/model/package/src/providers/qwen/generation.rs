//! Qwen model-card generation recommendations.
use infer_spi::{GenerationPolicy, GenerationPreset, GreedyPreset};

/// Recommended non-thinking temperature for Qwen3 and Qwen3.8.
const NON_THINKING_TEMPERATURE: f32 = 0.7;
/// Recommended non-thinking nucleus probability for Qwen3 and Qwen3.8.
const NON_THINKING_TOP_P: f32 = 0.8;
/// Recommended non-thinking presence penalty for Qwen3.8.
const QWEN38_PRESENCE_PENALTY: f32 = 1.5;

pub(super) fn policy(card: Option<&str>) -> GenerationPolicy {
    let Some(front) = card
        .and_then(|card| card.strip_prefix("---"))
        .and_then(|card| card.split_once("---"))
        .map(|(front, _)| front)
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
        match line {
            "- Qwen/Qwen3.8-27B" => return preset(QwenVariant::Qwen38),
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
                return preset(QwenVariant::Hybrid);
            }
            _ => {}
        }
    }
    GenerationPolicy::default()
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
