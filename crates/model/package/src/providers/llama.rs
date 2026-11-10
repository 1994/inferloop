//! Llama-family dense decoders: grouped-query attention without query/key norms, rotary
//! positions and a `SiLU`-gated MLP, under the same Hugging Face weight names as Qwen3.
use infer_core::{Error, ModelId, ProviderId, Result};
use infer_ir::{
    BackboneKind, DType, FeedForward, Head, Mixer, Modality, ModelIr, PositionSpec, StateKind,
    StateRequirement,
};
use infer_model_recipes::decoder::lower;
use infer_spi::{ImportedModel, ModelProvider, ProviderMetadata, SPI_VERSION};
use serde::Deserialize;
use std::num::NonZeroU64;

/// Maximum layer count accepted from a Llama configuration.
const MAX_HIDDEN_LAYERS: usize = 10_000;
/// `RoPE` theta of the original Llama release, which configurations may omit.
const DEFAULT_ROPE_THETA: f64 = 10_000.0;
/// `RMSNorm` epsilon applied when the configuration omits it.
const DEFAULT_RMS_NORM_EPS: f32 = 1e-6;
/// This provider's identity, after the Qwen provider's.
const PROVIDER: ProviderId = ProviderId::from_nonzero(match NonZeroU64::new(2) {
    Some(id) => id,
    None => NonZeroU64::MIN,
});

const fn theta() -> f64 {
    DEFAULT_ROPE_THETA
}
const fn epsilon() -> f32 {
    DEFAULT_RMS_NORM_EPS
}
fn silu() -> String {
    "silu".to_owned()
}

#[derive(Deserialize)]
struct LlamaConfig {
    model_type: String,
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    /// Absent in configurations without grouped-query attention.
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    intermediate_size: usize,
    max_position_embeddings: usize,
    /// Absent when it is `hidden_size / num_attention_heads`.
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "epsilon")]
    rms_norm_eps: f32,
    #[serde(default = "theta")]
    rope_theta: f64,
    /// Llama 3's frequency rescaling and `YaRN` are not modeled.
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "silu")]
    hidden_act: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    /// Mistral's sliding window; `null` attends the whole sequence.
    #[serde(default)]
    sliding_window: Option<usize>,
}

pub struct LlamaProvider;

/// # Errors
/// Returns an invalid-input error for malformed configuration, and an unsupported error for
/// semantics the decoder recipe does not model.
fn import_config(id: ModelId, bytes: &[u8]) -> Result<ImportedModel> {
    let config: LlamaConfig =
        serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string()))?;
    if !matches!(config.model_type.as_str(), "llama" | "mistral") {
        return Err(Error::unsupported(format!(
            "model type {}",
            config.model_type
        )));
    }
    if config.hidden_act != "silu" || config.attention_bias || config.mlp_bias {
        return Err(Error::unsupported(
            "unmodeled attention/FFN semantics in Llama configuration",
        ));
    }
    if config
        .rope_scaling
        .as_ref()
        .is_some_and(|scaling| !scaling.is_null())
    {
        return Err(Error::unsupported("RoPE scaling provider required"));
    }
    if config.num_hidden_layers == 0 || config.num_hidden_layers > MAX_HIDDEN_LAYERS {
        return Err(Error::invalid("invalid layer count"));
    }
    if config.num_attention_heads == 0 {
        return Err(Error::invalid("attention heads must be positive"));
    }
    let kv_heads = config
        .num_key_value_heads
        .unwrap_or(config.num_attention_heads);
    let head_dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads);
    let elements = kv_heads
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| Error::invalid("KV shape overflow"))?;
    let mixer = Mixer::Attention {
        query_heads: config.num_attention_heads,
        kv_heads,
        head_dim,
        sliding_window: config.sliding_window,
        output_gate: false,
        qk_norm: false,
    };
    let state = (0..config.num_hidden_layers)
        .map(|layer| StateRequirement {
            layer,
            kind: StateKind::AttentionKv,
            dtype: DType::Bf16,
            elements,
            per_token: true,
        })
        .collect();
    let model = ModelIr {
        id,
        backbone: BackboneKind::Decoder,
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        max_sequence: config.max_position_embeddings,
        mixers: vec![mixer; config.num_hidden_layers],
        feed_forward: FeedForward::Dense {
            intermediate: config.intermediate_size,
        },
        position: PositionSpec {
            rope_theta: config.rope_theta,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: config.rms_norm_eps,
        norm_weight_offset: 0.0,
        heads: vec![Head::LanguageModel],
        modalities: vec![Modality::Text],
        state,
        tied_embeddings: config.tie_word_embeddings,
    };
    model.validate()?;
    Ok(ImportedModel::new(model, 0))
}

impl ModelProvider for LlamaProvider {
    fn graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        lower(model)
    }
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: PROVIDER,
            name: "llama-hf-config".into(),
            spi_version: SPI_VERSION,
        }
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["llama", "mistral", "LlamaForCausalLM", "MistralForCausalLM"]
    }
    fn import(&self, id: ModelId, config: &[u8]) -> Result<ImportedModel> {
        import_config(id, config)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers_llama.rs"]
mod tests;
