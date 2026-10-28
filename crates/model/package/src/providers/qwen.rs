mod generation;
use crate::vision::validate;
use infer_core::{Error, ModelId, ProviderId, Result};
use infer_ir::{
    BackboneKind, DType, FeedForward, Head, Mixer, Modality, ModelIr, PositionSpec, PrecisionPlan,
    StateKind, StateRequirement,
};
use infer_model_recipes::decoder::lower;
use infer_spi::{
    FusionPlan, GenerationPolicy, HiddenActivation, ImportedModel, ModalityEncoder, ModalityPlan,
    ModelProvider, PrecisionPolicy, ProviderMetadata, SPI_VERSION, SpeculationPlan,
};
use serde::{Deserialize, Serialize};

/// Maximum layer count accepted from a Qwen text configuration.
const MAX_HIDDEN_LAYERS: usize = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub hidden_act: Option<String>,
}
#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    text_config: Option<TextConfig>,
    vision_config: Option<VisionConfig>,
    #[serde(default)]
    language_model_only: bool,
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
    #[serde(flatten)]
    text: serde_json::Map<String, serde_json::Value>,
}
/// Compressed-tensors section of `config.json`; only the declared weight formats matter here.
#[derive(Deserialize)]
struct QuantizationConfig {
    #[serde(default)]
    config_groups: Option<serde_json::Map<String, serde_json::Value>>,
}
/// Weight storage a checkpoint can declare.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WeightFormat {
    Dense,
    Fp8,
    Nvfp4,
}
/// Package prefix holding the speculative draft head.
const MTP_PREFIX: &str = "mtp.";
#[derive(Deserialize)]
struct RopeConfig {
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    #[serde(default)]
    mrope_section: Vec<usize>,
    #[serde(default)]
    mrope_interleaved: bool,
    #[serde(default)]
    rope_type: Option<String>,
}
/// `RoPE` theta applied when the model configuration omits it.
const DEFAULT_ROPE_THETA: f64 = 10000.0;
/// `RMSNorm` epsilon applied when the model configuration omits it.
const DEFAULT_RMS_NORM_EPS: f32 = 1e-6;
/// Attention head dimension applied when the model configuration omits it.
const DEFAULT_HEAD_DIM: usize = 128;
const fn theta() -> f64 {
    DEFAULT_ROPE_THETA
}
const fn one() -> f64 {
    1.0
}
const fn epsilon() -> f32 {
    DEFAULT_RMS_NORM_EPS
}
const fn head_dim() -> usize {
    DEFAULT_HEAD_DIM
}
#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "The external checkpoint configuration declares independent numerical policies"
)]
struct TextConfig {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "head_dim")]
    head_dim: usize,
    #[serde(default = "epsilon")]
    rms_norm_eps: f32,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    attn_output_gate: bool,
    #[serde(default)]
    output_gate_type: Option<String>,
    #[serde(default)]
    linear_num_key_heads: usize,
    #[serde(default)]
    linear_num_value_heads: usize,
    #[serde(default)]
    linear_key_head_dim: usize,
    #[serde(default)]
    linear_value_head_dim: usize,
    #[serde(default)]
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    rope_parameters: Option<RopeConfig>,
    #[serde(default)]
    rope_scaling: Option<RopeConfig>,
    #[serde(default = "theta")]
    rope_theta: f64,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    full_attention_interval: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_sparse_step")]
    decoder_sparse_step: usize,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
}
const fn default_sparse_step() -> usize {
    1
}
pub struct QwenProvider;
impl QwenProvider {
    /// # Errors
    /// Returns an invalid-input or unsupported error for malformed configuration or an unsupported Qwen architecture.
    pub fn import_manifest(&self, id: ModelId, bytes: &[u8]) -> Result<ImportedModel> {
        let config: HfConfig =
            serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string()))?;
        let hybrid = match config.model_type.as_str() {
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_next" => true,
            "qwen3" | "qwen3_moe" | "qwen3_vl" | "qwen3_vl_text" => false,
            other => return Err(Error::unsupported(format!("model type {other}"))),
        };
        // Read the family description before the configuration fields are moved into `text`.
        let placeholders = modality_placeholders(&config.text);
        let format = weight_format(config.quantization_config.as_ref());
        let mut text = if let Some(text) = config.text_config {
            text
        } else {
            serde_json::from_value(serde_json::Value::Object(config.text))
                .map_err(|e| Error::invalid(e.to_string()))?
        };
        if config.model_type == "qwen3_next" {
            text.attn_output_gate = true;
            if text.layer_types.is_empty() {
                if text.full_attention_interval == 0 {
                    return Err(Error::invalid("full_attention_interval must be positive"));
                }
                text.layer_types = (0..text.num_hidden_layers)
                    .map(|layer| {
                        if (layer + 1) % text.full_attention_interval == 0 {
                            "full_attention"
                        } else {
                            "linear_attention"
                        }
                        .to_owned()
                    })
                    .collect();
            }
        }
        let (mixers, state) = text.import_layers(hybrid)?;
        let feed_forward = text.import_feed_forward()?;
        let rope = text
            .rope_parameters
            .or(text.rope_scaling)
            .unwrap_or(RopeConfig {
                rope_theta: Some(text.rope_theta),
                partial_rotary_factor: text.partial_rotary_factor,
                mrope_section: vec![],
                mrope_interleaved: false,
                rope_type: None,
            });
        if rope.rope_type.as_deref().is_some_and(|s| s != "default") {
            return Err(Error::unsupported("RoPE scaling provider required"));
        }
        let vision = if config.language_model_only {
            None
        } else {
            config.vision_config
        };
        validate_vision(vision.as_ref(), text.hidden_size)?;
        let encoder = vision_encoder(vision.as_ref())?;
        if let Some(encoder) = &encoder {
            validate(encoder)?;
        }
        let model = ModelIr {
            id,
            backbone: if hybrid {
                BackboneKind::Hybrid
            } else {
                BackboneKind::Decoder
            },
            vocab_size: text.vocab_size,
            hidden_size: text.hidden_size,
            max_sequence: text.max_position_embeddings,
            mixers,
            feed_forward,
            position: PositionSpec {
                rope_theta: rope.rope_theta.unwrap_or(text.rope_theta),
                rotary_fraction: rope.partial_rotary_factor,
                multimodal_sections: rope.mrope_section,
                interleaved: rope.mrope_interleaved,
            },
            norm_epsilon: text.rms_norm_eps,
            norm_weight_offset: if hybrid { 1.0 } else { 0.0 },
            heads: vec![Head::LanguageModel],
            modalities: if vision.is_some() {
                vec![Modality::Text, Modality::Image, Modality::Video]
            } else {
                vec![Modality::Text]
            },
            state,
            tied_embeddings: text.tie_word_embeddings,
        };
        model.validate()?;
        Ok(family_description(
            model,
            text.mtp_num_hidden_layers,
            format,
            placeholders,
            encoder.as_ref(),
        ))
    }
}

/// Weight storage declared by the checkpoint's compressed-tensors groups.
fn weight_format(quantization: Option<&QuantizationConfig>) -> WeightFormat {
    let Some(groups) = quantization.and_then(|config| config.config_groups.as_ref()) else {
        return WeightFormat::Dense;
    };
    let mut format = WeightFormat::Dense;
    for group in groups.values() {
        match group["format"].as_str() {
            Some("nvfp4-pack-quantized") => return WeightFormat::Nvfp4,
            Some("float-quantized") => format = WeightFormat::Fp8,
            _ => {}
        }
    }
    format
}

/// Media placeholder tokens declared by the configuration, one plan per known modality.
///
/// An omni checkpoint that adds `audio_token_id` is declared here without any other change.
fn modality_placeholders(map: &serde_json::Map<String, serde_json::Value>) -> Vec<ModalityPlan> {
    const KEYS: [(&str, Modality); 3] = [
        ("image_token_id", Modality::Image),
        ("video_token_id", Modality::Video),
        ("audio_token_id", Modality::Audio),
    ];
    KEYS.iter()
        .filter_map(|(key, modality)| {
            let token = map
                .get(*key)?
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())?;
            Some(ModalityPlan {
                modality: modality.clone(),
                placeholder_tokens: vec![token],
                encoder: None,
            })
        })
        .collect()
}

impl ModelProvider for QwenProvider {
    fn generation_policy(&self, card: Option<&str>) -> GenerationPolicy {
        generation::policy(card)
    }
    fn graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        lower(model)
    }
    fn draft_graph(&self, model: &ModelIr) -> Result<(ModelIr, infer_ir::DataflowGraph)> {
        let mut block = model.clone();
        block.mixers = vec![
            model
                .mixers
                .iter()
                .find(|m| matches!(m, Mixer::Attention { .. }))
                .cloned()
                .ok_or_else(|| Error::invalid("MTP attention configuration"))?,
        ];
        block.state.clear();
        let graph = self.graph(&block)?;
        Ok((block, graph))
    }
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: ProviderId::ONE,
            name: "qwen-hf-config".into(),
            spi_version: SPI_VERSION,
        }
    }
    fn architectures(&self) -> &'static [&'static str] {
        // `model_type` and `architectures` values Qwen text and hybrid configurations use.
        &[
            "qwen3",
            "qwen3_moe",
            "qwen3_next",
            "Qwen3NextForCausalLM",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_vl",
            "qwen3_vl_text",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
        ]
    }
    fn import(&self, id: ModelId, config: &[u8]) -> Result<ImportedModel> {
        self.import_manifest(id, config)
    }
    fn weight_prefixes(&self) -> &'static [&'static str] {
        // Multimodal Qwen nests the text backbone under `model.language_model.`.
        &["model.language_model.", "model.", ""]
    }
    fn weight_source(&self, slot: &str, prefix: &str) -> String {
        // The language-model head is stored at the package root, outside the backbone prefix.
        if slot == "lm_head.weight" {
            slot.to_owned()
        } else {
            format!("{prefix}{slot}")
        }
    }
}

impl TextConfig {
    fn import_feed_forward(&self) -> Result<FeedForward> {
        if self.num_experts == 0 {
            if self.num_experts_per_tok != 0
                || self.moe_intermediate_size != 0
                || self.shared_expert_intermediate_size != 0
            {
                return Err(Error::invalid("MoE configuration requires num_experts"));
            }
            return Ok(FeedForward::Dense {
                intermediate: self.intermediate_size,
            });
        }
        if self.decoder_sparse_step == 0
            || self
                .mlp_only_layers
                .iter()
                .any(|layer| *layer >= self.num_hidden_layers)
        {
            return Err(Error::invalid("invalid sparse layer placement"));
        }
        let dense_layers = (0..self.num_hidden_layers)
            .filter(|layer| {
                self.mlp_only_layers.contains(layer) || (layer + 1) % self.decoder_sparse_step != 0
            })
            .collect();
        Ok(FeedForward::Moe {
            experts: self.num_experts,
            active: self.num_experts_per_tok,
            intermediate: self.moe_intermediate_size,
            normalize_topk: self.norm_topk_prob,
            shared_intermediate: (self.shared_expert_intermediate_size > 0)
                .then_some(self.shared_expert_intermediate_size),
            dense_layers,
            dense_intermediate: self.intermediate_size,
        })
    }
    fn import_layers(&self, hybrid: bool) -> Result<(Vec<Mixer>, Vec<StateRequirement>)> {
        if self.hidden_act.as_deref().is_some_and(|s| s != "silu")
            || self.attention_bias
            || self
                .output_gate_type
                .as_deref()
                .is_some_and(|s| s != "swish")
        {
            return Err(Error::unsupported(
                "unmodeled attention/FFN semantics in Qwen configuration",
            ));
        }
        if self.num_hidden_layers == 0 || self.num_hidden_layers > MAX_HIDDEN_LAYERS {
            return Err(Error::invalid("invalid layer count"));
        }
        let layer_types = if self.layer_types.is_empty() && !hybrid {
            vec!["full_attention".into(); self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        };
        if layer_types.len() != self.num_hidden_layers {
            return Err(Error::invalid("layer_types must cover every layer"));
        }
        let mut mixers = Vec::new();
        let mut state = Vec::new();
        for (layer, kind) in layer_types.iter().enumerate() {
            match kind.as_str() {
                "full_attention" => {
                    mixers.push(Mixer::Attention {
                        query_heads: self.num_attention_heads,
                        kv_heads: self.num_key_value_heads,
                        head_dim: self.head_dim,
                        sliding_window: None,
                        output_gate: self.attn_output_gate,
                        qk_norm: true,
                    });
                    let elements = self
                        .num_key_value_heads
                        .checked_mul(self.head_dim)
                        .and_then(|v| v.checked_mul(2))
                        .ok_or_else(|| Error::invalid("KV shape overflow"))?;
                    state.push(StateRequirement {
                        layer,
                        kind: StateKind::AttentionKv,
                        dtype: DType::Bf16,
                        elements,
                        per_token: true,
                    });
                }
                "linear_attention" if hybrid => {
                    mixers.push(Mixer::LinearAttention {
                        key_heads: self.linear_num_key_heads,
                        value_heads: self.linear_num_value_heads,
                        key_dim: self.linear_key_head_dim,
                        value_dim: self.linear_value_head_dim,
                        conv_kernel: self.linear_conv_kernel_dim,
                    });
                    self.import_linear_state(layer, &mut state)?;
                }
                other => return Err(Error::unsupported(format!("mixer {other}"))),
            }
        }
        Ok((mixers, state))
    }
    fn import_linear_state(&self, layer: usize, state: &mut Vec<StateRequirement>) -> Result<()> {
        let elements = self
            .linear_num_value_heads
            .checked_mul(self.linear_key_head_dim)
            .and_then(|v| v.checked_mul(self.linear_value_head_dim))
            .ok_or_else(|| Error::invalid("linear state shape overflow"))?;
        state.push(StateRequirement {
            layer,
            kind: StateKind::LinearAttention,
            dtype: DType::F32,
            elements,
            per_token: false,
        });
        let channels = self
            .linear_num_key_heads
            .checked_mul(self.linear_key_head_dim)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| {
                self.linear_num_value_heads
                    .checked_mul(self.linear_value_head_dim)
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or_else(|| Error::invalid("convolution shape overflow"))?;
        let elements = channels
            .checked_mul(
                self.linear_conv_kernel_dim
                    .checked_sub(1)
                    .ok_or_else(|| Error::invalid("empty convolution kernel"))?,
            )
            .ok_or_else(|| Error::invalid("convolution state overflow"))?;
        state.push(StateRequirement {
            layer,
            kind: StateKind::Conv,
            dtype: DType::Bf16,
            elements,
            per_token: false,
        });
        Ok(())
    }
}

/// Assemble the declared family description: precision policy, draft head and extra modalities.
///
/// This is the single place a Qwen checkpoint's non-IR customization is expressed; engines read
/// the result and never branch on a model name.
fn family_description(
    model: ModelIr,
    draft_layers: usize,
    format: WeightFormat,
    placeholders: Vec<ModalityPlan>,
    encoder: Option<&ModalityEncoder>,
) -> ImportedModel {
    let mut imported = ImportedModel::new(model, draft_layers);
    imported.precision = match format {
        WeightFormat::Nvfp4 => {
            PrecisionPolicy::Preferred(vec![PrecisionPlan::nvfp4_block(), PrecisionPlan::bf16()])
        }
        WeightFormat::Fp8 => {
            PrecisionPolicy::Preferred(vec![PrecisionPlan::fp8_channel(), PrecisionPlan::bf16()])
        }
        WeightFormat::Dense => PrecisionPolicy::default(),
    };
    imported.speculation = (draft_layers > 0).then(|| SpeculationPlan {
        prefix: MTP_PREFIX.into(),
        layers: draft_layers,
        fusion: Some(FusionPlan {
            projection: "fc.weight".into(),
            norms: [
                "pre_fc_norm_embedding.weight".into(),
                "pre_fc_norm_hidden.weight".into(),
            ],
        }),
    });
    imported.modalities = if encoder.is_some() {
        placeholders
            .into_iter()
            .map(|mut plan| {
                if matches!(plan.modality, Modality::Image | Modality::Video) {
                    plan.encoder = encoder.cloned();
                }
                plan
            })
            .collect()
    } else {
        Vec::new()
    };
    imported
}

/// Package prefix holding the vision tower.
const VISION_PREFIX: &str = "model.visual.";
/// Axial `RoPE` theta of the vision tower; the published configuration omits it.
const VISION_ROPE_THETA: f64 = 10_000.0;
/// `LayerNorm` epsilon of vision blocks and merger, fixed by the architecture.
const VISION_NORM_EPS: f32 = 1e-6;

/// Vision encoder geometry declared to backends, or `None` for a text-only checkpoint.
fn vision_encoder(vision: Option<&VisionConfig>) -> Result<Option<ModalityEncoder>> {
    let Some(vision) = vision else {
        return Ok(None);
    };
    let hidden_activation = match vision.hidden_act.as_deref().unwrap_or("gelu_pytorch_tanh") {
        "gelu_pytorch_tanh" => HiddenActivation::GeluTanh,
        "gelu" => HiddenActivation::GeluErf,
        "silu" => HiddenActivation::Silu,
        other => return Err(Error::unsupported(format!("vision activation {other}"))),
    };
    Ok(Some(ModalityEncoder {
        prefix: VISION_PREFIX.into(),
        depth: vision.depth,
        hidden_size: vision.hidden_size,
        intermediate_size: vision.intermediate_size,
        heads: vision.num_heads,
        in_channels: vision.in_channels,
        patch_size: vision.patch_size,
        temporal_patch_size: vision.temporal_patch_size,
        spatial_merge_size: vision.spatial_merge_size,
        out_hidden_size: vision.out_hidden_size,
        position_embeddings: vision.num_position_embeddings,
        rope_theta: VISION_ROPE_THETA,
        norm_epsilon: VISION_NORM_EPS,
        hidden_activation,
    }))
}

fn validate_vision(vision: Option<&VisionConfig>, hidden_size: usize) -> Result<()> {
    if let Some(vision) = vision
        && ([
            vision.depth,
            vision.hidden_size,
            vision.intermediate_size,
            vision.out_hidden_size,
            vision.num_heads,
            vision.in_channels,
            vision.patch_size,
            vision.temporal_patch_size,
            vision.spatial_merge_size,
            vision.num_position_embeddings,
        ]
        .contains(&0)
            || vision.out_hidden_size != hidden_size)
    {
        return Err(Error::invalid("invalid vision encoder configuration"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/providers_qwen.rs"]
mod tests;
