//! Model-owned generation defaults consumed by the shared sampling resolver.

/// How a mode preset interacts with an explicitly greedy package configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GreedyPreset {
    /// Leave a greedy package configuration unchanged.
    #[default]
    Preserve,
    /// Apply the mode preset even when the package configured greedy decoding.
    Apply,
}

/// Optional per-field recommendation for a model generation mode.
#[derive(Debug, Clone, Default)]
pub struct GenerationPreset {
    /// Softmax temperature.
    pub temperature: Option<f32>,
    /// Zero disables top-k filtering.
    pub top_k: Option<usize>,
    /// Nucleus probability.
    pub top_p: Option<f32>,
    /// Minimum probability relative to the most likely token.
    pub min_p: Option<f32>,
    /// Penalty applied to tokens already generated.
    pub presence_penalty: Option<f32>,
    /// Penalty applied to tokens in prompt or output.
    pub repetition_penalty: Option<f32>,
    /// Whether the preset can replace a package configured for greedy decoding.
    pub greedy: GreedyPreset,
    /// Explanation returned with the resolved parameters.
    pub source: &'static str,
}

/// Generation defaults declared by a model provider at package load time.
#[derive(Debug, Clone, Default)]
pub struct GenerationPolicy {
    /// Default mode passed to the package chat template.
    pub enable_thinking: bool,
    /// Sampling fallback when the generation configuration does not specify `do_sample`.
    pub sample_by_default: bool,
    /// Per-field recommendation for explicit or default non-thinking mode.
    pub non_thinking: Option<GenerationPreset>,
}
