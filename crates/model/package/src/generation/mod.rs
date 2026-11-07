//! Model generation defaults with explicit, per-field request overrides.
mod read;
#[cfg(test)]
#[path = "../../tests/unit/generation.rs"]
mod tests;
use crate::{ModelRegistry, default_registry};
use infer_core::Result;
use infer_ir::Sampling;
use infer_spi::{GenerationPolicy, GenerationPreset, GreedyPreset};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

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

#[derive(Debug, Clone)]
pub struct GenerationDefaults {
    base: Sampling,
    policy: GenerationPolicy,
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
        Self::open_with(default_registry(), root)
    }

    /// Resolve a package with the same model registry used to load its model.
    /// # Errors
    /// Rejects malformed, oversized or invalid model sampling configuration.
    pub fn open_with(registry: &ModelRegistry, root: impl AsRef<Path>) -> Result<Self> {
        read::load(root.as_ref(), registry)
    }

    /// Resolve package defaults, the recognized model's mode preset, then explicit overrides.
    /// # Errors
    /// Rejects invalid effective parameters; `top_k=0` explicitly disables filtering.
    pub fn resolve(&self, options: &SamplingOverrides) -> Result<ResolvedGeneration> {
        let mut sampling = self.base.clone();
        let mut sources = self.sources.clone();
        let thinking = options
            .enable_thinking
            .unwrap_or(self.policy.enable_thinking);
        // A request can enable sampling even if the package defaults to greedy. Preserve the
        // model-mode penalty defaults when a sampled package is overridden to temperature zero.
        let sampling_active = sampling.temperature != 0.0
            || options.do_sample == Some(true)
            || options.temperature.is_some_and(|value| value != 0.0);
        if !thinking
            && let Some(preset) = &self.policy.non_thinking
            && (sampling_active || preset.greedy == GreedyPreset::Apply)
        {
            apply_preset(&mut sampling, &mut sources, preset);
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

fn apply_preset(
    sampling: &mut Sampling,
    sources: &mut BTreeMap<String, String>,
    preset: &GenerationPreset,
) {
    macro_rules! apply {
        ($($name:ident),+) => { $(if let Some(value) = preset.$name {
            sampling.$name = value;
            sources.insert(stringify!($name).into(), preset.source.into());
        })+ };
    }
    apply!(
        temperature,
        top_p,
        min_p,
        presence_penalty,
        repetition_penalty
    );
    if let Some(value) = preset.top_k {
        sampling.top_k = (value != 0).then_some(value);
        sources.insert("top_k".into(), preset.source.into());
    }
}
