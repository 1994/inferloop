//! The `qwen3vl` GGUF architecture: the Qwen3-VL text backbone, with an optional mmproj.
use super::vision::{IN_CHANNELS, TEMPORAL_PATCH, deepstack_layers, position_grid};
use crate::storage::gguf::{
    Block, ExpectedTensors, GgufArchitecture, LayerPlan, MappedTensor, TensorSpec,
    common_block_field, count, is_float, token_id, trim_trailing_zeros, u64_array,
};
use infer_core::{Error, Result};
use infer_format::Metadata;
use serde::Serialize;

const ARCH: &str = "qwen3vl";

/// The `qwen3vl` architecture.
pub struct Qwen3VlGguf;

/// The `qwen3vl` adapter.
pub const QWEN3VL: Qwen3VlGguf = Qwen3VlGguf;

#[derive(Serialize)]
struct TextConfig {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_scaling: RopeScaling,
    hidden_act: &'static str,
    tie_word_embeddings: bool,
}

#[derive(Serialize)]
struct RopeScaling {
    mrope_section: Vec<u64>,
    mrope_interleaved: bool,
    rope_type: &'static str,
}

#[derive(Serialize)]
struct VisionConfig {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    out_hidden_size: usize,
    num_position_embeddings: usize,
    hidden_act: &'static str,
    deepstack_visual_indexes: Vec<usize>,
}

/// The HF `config.json` this adapter synthesizes; text-only without an mmproj.
#[derive(Serialize)]
struct Synthesized {
    architectures: [&'static str; 1],
    model_type: &'static str,
    image_token_id: u32,
    video_token_id: u32,
    tie_word_embeddings: bool,
    language_model_only: bool,
    text_config: TextConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision_config: Option<VisionConfig>,
}

fn key(name: &str) -> String {
    format!("{ARCH}.{name}")
}

impl GgufArchitecture for Qwen3VlGguf {
    fn name(&self) -> &'static str {
        ARCH
    }

    fn config(
        &self,
        text: &Metadata,
        _tensors: &[TensorSpec],
        vision: Option<&Metadata>,
    ) -> Result<Vec<u8>> {
        let config = Synthesized {
            architectures: ["Qwen3VLForConditionalGeneration"],
            model_type: "qwen3_vl",
            image_token_id: token_id(text, "<|image_pad|>")?,
            video_token_id: token_id(text, "<|video_pad|>")?,
            tie_word_embeddings: true,
            language_model_only: vision.is_none(),
            text_config: text_config(text)?,
            vision_config: vision.map(vision_config).transpose()?,
        };
        serde_json::to_vec(&config).map_err(|error| Error::invalid(error.to_string()))
    }

    fn tensor(&self, tensor: &TensorSpec, layers: &LayerPlan) -> Option<MappedTensor> {
        let slot = match tensor.name.as_str() {
            "token_embd.weight" => "model.embed_tokens.weight".to_owned(),
            "output_norm.weight" => "model.norm.weight".to_owned(),
            "output.weight" => "lm_head.weight".to_owned(),
            name => {
                let Block {
                    index,
                    draft: None,
                    field,
                } = layers.block(name)?
                else {
                    return None;
                };
                format!("model.layers.{index}.{}", common_block_field(field)?)
            }
        };
        Some(MappedTensor::named(slot))
    }

    fn validate(&self, text: &Metadata, tensors: &[TensorSpec]) -> Result<()> {
        validate_text(text, tensors)
    }

    fn accepts_vision(&self) -> bool {
        true
    }
}

fn text_config(text: &Metadata) -> Result<TextConfig> {
    Ok(TextConfig {
        hidden_size: count(text, &key("embedding_length"))?,
        vocab_size: text.require_strings("tokenizer.ggml.tokens")?.len(),
        num_hidden_layers: count(text, &key("block_count"))?,
        num_attention_heads: count(text, &key("attention.head_count"))?,
        num_key_value_heads: count(text, &key("attention.head_count_kv"))?,
        intermediate_size: count(text, &key("feed_forward_length"))?,
        max_position_embeddings: count(text, &key("context_length"))?,
        head_dim: count(text, &key("attention.key_length"))?,
        rms_norm_eps: text.require_f64(&key("attention.layer_norm_rms_epsilon"))?,
        rope_theta: text.require_f64(&key("rope.freq_base"))?,
        rope_scaling: RopeScaling {
            mrope_section: trim_trailing_zeros(u64_array(text, &key("rope.dimension_sections"))?),
            mrope_interleaved: true,
            rope_type: "default",
        },
        hidden_act: "silu",
        tie_word_embeddings: true,
    })
}

fn vision_config(mmproj: &Metadata) -> Result<VisionConfig> {
    Ok(VisionConfig {
        depth: count(mmproj, "clip.vision.block_count")?,
        hidden_size: count(mmproj, "clip.vision.embedding_length")?,
        intermediate_size: count(mmproj, "clip.vision.feed_forward_length")?,
        num_heads: count(mmproj, "clip.vision.attention.head_count")?,
        in_channels: IN_CHANNELS,
        patch_size: count(mmproj, "clip.vision.patch_size")?,
        temporal_patch_size: TEMPORAL_PATCH,
        spatial_merge_size: count(mmproj, "clip.vision.spatial_merge_size")?,
        out_hidden_size: count(mmproj, "clip.vision.projection_dim")?,
        num_position_embeddings: position_grid(mmproj)?,
        hidden_act: "gelu_pytorch_tanh",
        deepstack_visual_indexes: deepstack_layers(mmproj)?,
    })
}

/// Validate every text tensor against the `qwen3vl` geometry.
/// # Errors
/// Rejects a missing or extra tensor, a shape mismatch, or an unsupported storage type.
fn validate_text(metadata: &Metadata, tensors: &[TensorSpec]) -> Result<()> {
    let hidden = count(metadata, &key("embedding_length"))?;
    let heads = count(metadata, &key("attention.head_count"))?;
    let kv_heads = count(metadata, &key("attention.head_count_kv"))?;
    let head_dim = count(metadata, &key("attention.key_length"))?;
    let ffn = count(metadata, &key("feed_forward_length"))?;
    let vocab = metadata.require_strings("tokenizer.ggml.tokens")?.len();
    let width = |heads: usize| {
        heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::invalid("Qwen3-VL attention width overflow"))
    };
    let (q_width, kv_width) = (width(heads)?, width(kv_heads)?);
    let mut expected = ExpectedTensors::new();
    expected.insert("token_embd.weight", vec![vocab, hidden])?;
    expected.insert("output_norm.weight", vec![hidden])?;
    if tensors.iter().any(|tensor| tensor.name == "output.weight") {
        expected.insert("output.weight", vec![vocab, hidden])?;
    }
    for layer in 0..count(metadata, &key("block_count"))? {
        for (field, shape) in [
            ("attn_norm.weight", vec![hidden]),
            ("attn_q.weight", vec![q_width, hidden]),
            ("attn_k.weight", vec![kv_width, hidden]),
            ("attn_v.weight", vec![kv_width, hidden]),
            ("attn_output.weight", vec![hidden, q_width]),
            ("attn_q_norm.weight", vec![head_dim]),
            ("attn_k_norm.weight", vec![head_dim]),
            ("ffn_norm.weight", vec![hidden]),
            ("ffn_gate.weight", vec![ffn, hidden]),
            ("ffn_up.weight", vec![ffn, hidden]),
            ("ffn_down.weight", vec![hidden, ffn]),
        ] {
            expected.insert(format!("blk.{layer}.{field}"), shape)?;
        }
    }
    expected.check(tensors, |ty| is_float(ty) || ty.grouped_int().is_some())
}
