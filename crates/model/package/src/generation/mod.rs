//! Model generation defaults with explicit, per-field request overrides.
mod read;
#[cfg(test)]
#[path = "../../tests/unit/generation.rs"]
mod tests;
use infer_core::Result;
use infer_ir::Sampling;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

/// Qwen3 hybrid and Qwen3.8 model-card temperature for non-thinking generation.
const QWEN3_NON_THINKING_TEMPERATURE: f32 = 0.7;
/// Qwen3 hybrid and Qwen3.8 model-card top-p for non-thinking generation.
const QWEN3_NON_THINKING_TOP_P: f32 = 0.8;
/// Qwen3.8 model-card presence penalty for non-thinking generation.
const QWEN3_NON_THINKING_PRESENCE_PENALTY: f32 = 1.5;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingOverrides {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub do_sample: Option<bool>,
    pub eos_tokens: Option<Vec<u32>>,
    pub eos_token: Option<u32>,
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelProfile {
    Other,
    Qwen3Hybrid,
    Qwen38,
}

#[derive(Debug, Clone)]
pub struct GenerationDefaults {
    base: Sampling,
    thinking: bool,
    profile: ModelProfile,
    sources: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedGeneration {
    pub sampling: Sampling,
    pub enable_thinking: bool,
    pub sources: BTreeMap<String, String>,
}

impl GenerationDefaults {
    /// # Errors
    /// Rejects malformed, oversized or invalid model sampling configuration.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        read::load(root.as_ref())
    }

    /// Resolve package defaults, the recognized model's mode preset, then explicit overrides.
    /// # Errors
    /// Rejects invalid effective parameters; `top_k=0` explicitly disables filtering.
    pub fn resolve(&self, options: &SamplingOverrides) -> Result<ResolvedGeneration> {
        let mut sampling = self.base.clone();
        let mut sources = self.sources.clone();
        let thinking = options.enable_thinking.unwrap_or(self.thinking);
        if !thinking {
            let source = match self.profile {
                ModelProfile::Qwen3Hybrid if sampling.temperature != 0.0 => {
                    Some("Qwen3 hybrid model-card non-thinking recommendation")
                }
                ModelProfile::Qwen38 => {
                    sampling.presence_penalty = QWEN3_NON_THINKING_PRESENCE_PENALTY;
                    sources.insert(
                        "presence_penalty".into(),
                        "Qwen3.8 model-card non-thinking recommendation".into(),
                    );
                    Some("Qwen3.8 model-card non-thinking recommendation")
                }
                ModelProfile::Other | ModelProfile::Qwen3Hybrid => None,
            };
            if let Some(source) = source {
                sampling.temperature = QWEN3_NON_THINKING_TEMPERATURE;
                sampling.top_p = QWEN3_NON_THINKING_TOP_P;
                for key in ["temperature", "top_p"] {
                    sources.insert(key.into(), source.into());
                }
            }
        }
        macro_rules! apply {
            ($($name:ident),+) => { $(if let Some(value) = options.$name {
                sampling.$name = value;
                sources.insert(stringify!($name).into(), "request override".into());
            })+ };
        }
        apply!(
            temperature,
            top_p,
            min_p,
            presence_penalty,
            repetition_penalty,
            seed
        );
        if let Some(value) = options.top_k {
            sampling.top_k = (value != 0).then_some(value);
            sources.insert("top_k".into(), "request override".into());
        }
        if options.do_sample == Some(false) {
            sampling.temperature = 0.0;
            sources.insert("temperature".into(), "request do_sample=false".into());
        } else if options.do_sample == Some(true)
            && options.temperature.is_none()
            && sampling.temperature == 0.0
        {
            sampling.temperature = 1.0;
            sources.insert(
                "temperature".into(),
                "request do_sample=true fallback".into(),
            );
        }
        if let Some(tokens) = &options.eos_tokens {
            sampling.eos_tokens.clone_from(tokens);
            sampling.eos_token = None;
            sources.insert("eos_tokens".into(), "request override".into());
        }
        if let Some(token) = options.eos_token {
            sampling.eos_token = Some(token);
            sampling.eos_tokens.clear();
            sources.insert("eos_tokens".into(), "request eos_token override".into());
        }
        sampling.validate()?;
        sources.insert(
            "enable_thinking".into(),
            if options.enable_thinking.is_some() {
                "request override"
            } else {
                "model profile default"
            }
            .into(),
        );
        Ok(ResolvedGeneration {
            sampling,
            enable_thinking: thinking,
            sources,
        })
    }
}
