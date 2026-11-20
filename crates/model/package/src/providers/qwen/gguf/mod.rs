//! Qwen3-VL GGUF configuration and tensor mapping.

pub mod tokenizer;
pub mod validate;

use crate::vision::{VisionSlot, slots};
use infer_core::{Error, Result};
use infer_format::{Metadata, MetadataValue};
use infer_spi::{HiddenActivation, ModalityEncoder};
use serde::Serialize;

/// RGB channels of the vision patch embedding.
const IN_CHANNELS: u64 = 3;
/// Package prefix the existing vision inventory expects.
const VISION_PREFIX: &str = "model.visual.";
/// Axial vision `RoPE` theta used by the Qwen3-VL family.
const VISION_ROPE_THETA: f64 = 10_000.0;

/// HF text-backbone configuration synthesized from `qwen3vl.*` metadata.
#[derive(Debug, Serialize)]
pub struct Qwen3VlTextConfig {
    pub hidden_size: u64,
    pub vocab_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub intermediate_size: u64,
    pub max_position_embeddings: u64,
    pub head_dim: u64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Qwen3VlRopeScaling,
    pub hidden_act: &'static str,
    pub tie_word_embeddings: bool,
}

/// Multimodal `RoPE` declaration.
#[derive(Debug, Serialize)]
pub struct Qwen3VlRopeScaling {
    pub mrope_section: Vec<u64>,
    pub mrope_interleaved: bool,
    pub rope_type: &'static str,
}

/// HF vision-tower configuration synthesized from `clip.*` metadata.
#[derive(Debug, Serialize)]
pub struct Qwen3VlVisionConfig {
    pub depth: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_heads: u64,
    pub in_channels: u64,
    pub patch_size: u64,
    pub temporal_patch_size: u64,
    pub spatial_merge_size: u64,
    pub out_hidden_size: u64,
    pub num_position_embeddings: u64,
    pub hidden_act: &'static str,
    pub deepstack_visual_indexes: Vec<u64>,
}

/// Top-level HF configuration.
#[derive(Debug, Serialize)]
pub struct Qwen3VlConfig {
    pub architectures: [&'static str; 1],
    pub model_type: &'static str,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub tie_word_embeddings: bool,
    pub text_config: Qwen3VlTextConfig,
    pub vision_config: Qwen3VlVisionConfig,
}

/// Top-level HF configuration for a text-only package.
#[derive(Debug, Serialize)]
pub struct Qwen3VlTextOnlyConfig {
    pub architectures: [&'static str; 1],
    pub model_type: &'static str,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub tie_word_embeddings: bool,
    pub language_model_only: bool,
    pub text_config: Qwen3VlTextConfig,
}

use super::VisionTensor;

/// Synthesize the HF configuration for a Qwen3-VL text GGUF and its mmproj.
/// # Errors
/// Rejects missing or malformed geometry, a tokenizer without the image/video markers, or a
/// vision tower whose position grid is not square.
pub fn config(text: &Metadata, mmproj: &Metadata) -> Result<Vec<u8>> {
    let tokens = text.require_strings("tokenizer.ggml.tokens")?;
    let vocab_size = u64::try_from(tokens.len())
        .map_err(|_| Error::invalid("Qwen3-VL vocabulary exceeds u64"))?;
    let image_token_id = token_id(&tokens, "<|image_pad|>")?;
    let video_token_id = token_id(&tokens, "<|video_pad|>")?;
    let sections = u64_array(text, "qwen3vl.rope.dimension_sections")?;
    let sections = sections
        .into_iter()
        .rev()
        .skip_while(|value| *value == 0)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let image_size = mmproj.require_u64("clip.vision.image_size")?;
    let patch_size = mmproj.require_u64("clip.vision.patch_size")?;
    if patch_size == 0 || !image_size.is_multiple_of(patch_size) {
        return Err(Error::invalid(
            "Qwen3-VL image size must divide by patch size",
        ));
    }
    let side = image_size / patch_size;
    let positions = side
        .checked_mul(side)
        .ok_or_else(|| Error::invalid("Qwen3-VL position grid overflow"))?;
    let deepstack = bool_array(mmproj, "clip.vision.is_deepstack_layers")?
        .into_iter()
        .enumerate()
        .filter_map(|(index, enabled)| enabled.then_some(index as u64))
        .collect();
    let config = Qwen3VlConfig {
        architectures: ["Qwen3VLForConditionalGeneration"],
        model_type: "qwen3_vl",
        image_token_id,
        video_token_id,
        tie_word_embeddings: true,
        text_config: Qwen3VlTextConfig {
            hidden_size: text.require_u64("qwen3vl.embedding_length")?,
            vocab_size,
            num_hidden_layers: text.require_u64("qwen3vl.block_count")?,
            num_attention_heads: text.require_u64("qwen3vl.attention.head_count")?,
            num_key_value_heads: text.require_u64("qwen3vl.attention.head_count_kv")?,
            intermediate_size: text.require_u64("qwen3vl.feed_forward_length")?,
            max_position_embeddings: text.require_u64("qwen3vl.context_length")?,
            head_dim: text.require_u64("qwen3vl.attention.key_length")?,
            rms_norm_eps: text.require_f64("qwen3vl.attention.layer_norm_rms_epsilon")?,
            rope_theta: text.require_f64("qwen3vl.rope.freq_base")?,
            rope_scaling: Qwen3VlRopeScaling {
                mrope_section: sections,
                mrope_interleaved: true,
                rope_type: "default",
            },
            hidden_act: "silu",
            tie_word_embeddings: true,
        },
        vision_config: Qwen3VlVisionConfig {
            depth: mmproj.require_u64("clip.vision.block_count")?,
            hidden_size: mmproj.require_u64("clip.vision.embedding_length")?,
            intermediate_size: mmproj.require_u64("clip.vision.feed_forward_length")?,
            num_heads: mmproj.require_u64("clip.vision.attention.head_count")?,
            in_channels: IN_CHANNELS,
            patch_size,
            temporal_patch_size: 2,
            spatial_merge_size: mmproj.require_u64("clip.vision.spatial_merge_size")?,
            out_hidden_size: mmproj.require_u64("clip.vision.projection_dim")?,
            num_position_embeddings: positions,
            hidden_act: "gelu_pytorch_tanh",
            deepstack_visual_indexes: deepstack,
        },
    };
    serde_json::to_vec(&config).map_err(|error| Error::invalid(error.to_string()))
}

/// Synthesize a text-only HF configuration for `infer-cli --package model.gguf`.
/// # Errors
/// Rejects missing text geometry or tokenizer markers.
pub fn config_text(text: &Metadata) -> Result<Vec<u8>> {
    let tokens = text.require_strings("tokenizer.ggml.tokens")?;
    let vocab_size = u64::try_from(tokens.len())
        .map_err(|_| Error::invalid("Qwen3-VL vocabulary exceeds u64"))?;
    let sections = u64_array(text, "qwen3vl.rope.dimension_sections")?;
    let sections = sections
        .into_iter()
        .rev()
        .skip_while(|value| *value == 0)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let config = Qwen3VlTextOnlyConfig {
        architectures: ["Qwen3VLForConditionalGeneration"],
        model_type: "qwen3_vl",
        image_token_id: token_id(&tokens, "<|image_pad|>")?,
        video_token_id: token_id(&tokens, "<|video_pad|>")?,
        tie_word_embeddings: true,
        language_model_only: true,
        text_config: Qwen3VlTextConfig {
            hidden_size: text.require_u64("qwen3vl.embedding_length")?,
            vocab_size,
            num_hidden_layers: text.require_u64("qwen3vl.block_count")?,
            num_attention_heads: text.require_u64("qwen3vl.attention.head_count")?,
            num_key_value_heads: text.require_u64("qwen3vl.attention.head_count_kv")?,
            intermediate_size: text.require_u64("qwen3vl.feed_forward_length")?,
            max_position_embeddings: text.require_u64("qwen3vl.context_length")?,
            head_dim: text.require_u64("qwen3vl.attention.key_length")?,
            rms_norm_eps: text.require_f64("qwen3vl.attention.layer_norm_rms_epsilon")?,
            rope_theta: text.require_f64("qwen3vl.rope.freq_base")?,
            rope_scaling: Qwen3VlRopeScaling {
                mrope_section: sections,
                mrope_interleaved: true,
                rope_type: "default",
            },
            hidden_act: "silu",
            tie_word_embeddings: true,
        },
    };
    serde_json::to_vec(&config).map_err(|error| Error::invalid(error.to_string()))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the positive LayerNorm epsilon is well inside the f32 range"
)]
/// Build the existing vision-encoder declaration from mmproj metadata.
/// # Errors
/// Rejects malformed geometry, a non-square position grid, or a zero norm epsilon.
pub fn encoder(mmproj: &Metadata) -> Result<ModalityEncoder> {
    let image_size = mmproj.require_u64("clip.vision.image_size")?;
    let patch_size = mmproj.require_u64("clip.vision.patch_size")?;
    if patch_size == 0 || !image_size.is_multiple_of(patch_size) {
        return Err(Error::invalid(
            "Qwen3-VL image size must divide by patch size",
        ));
    }
    let side = image_size / patch_size;
    let position_embeddings = side
        .checked_mul(side)
        .ok_or_else(|| Error::invalid("Qwen3-VL position grid overflow"))?;
    let deepstack_layers = bool_array(mmproj, "clip.vision.is_deepstack_layers")?
        .into_iter()
        .enumerate()
        .filter_map(|(index, enabled)| enabled.then_some(index))
        .collect();
    Ok(ModalityEncoder {
        prefix: VISION_PREFIX.into(),
        depth: usize::try_from(mmproj.require_u64("clip.vision.block_count")?)
            .map_err(|_| Error::invalid("Qwen3-VL depth exceeds usize"))?,
        hidden_size: usize::try_from(mmproj.require_u64("clip.vision.embedding_length")?)
            .map_err(|_| Error::invalid("Qwen3-VL hidden size exceeds usize"))?,
        intermediate_size: usize::try_from(mmproj.require_u64("clip.vision.feed_forward_length")?)
            .map_err(|_| Error::invalid("Qwen3-VL intermediate size exceeds usize"))?,
        heads: usize::try_from(mmproj.require_u64("clip.vision.attention.head_count")?)
            .map_err(|_| Error::invalid("Qwen3-VL head count exceeds usize"))?,
        in_channels: usize::try_from(IN_CHANNELS)
            .map_err(|_| Error::invalid("Qwen3-VL channel count exceeds usize"))?,
        patch_size: usize::try_from(patch_size)
            .map_err(|_| Error::invalid("Qwen3-VL patch size exceeds usize"))?,
        temporal_patch_size: 2,
        spatial_merge_size: usize::try_from(mmproj.require_u64("clip.vision.spatial_merge_size")?)
            .map_err(|_| Error::invalid("Qwen3-VL merge size exceeds usize"))?,
        out_hidden_size: usize::try_from(mmproj.require_u64("clip.vision.projection_dim")?)
            .map_err(|_| Error::invalid("Qwen3-VL projection width exceeds usize"))?,
        position_embeddings: usize::try_from(position_embeddings)
            .map_err(|_| Error::invalid("Qwen3-VL position count exceeds usize"))?,
        rope_theta: VISION_ROPE_THETA,
        norm_epsilon: mmproj.require_f64("clip.vision.attention.layer_norm_epsilon")? as f32,
        hidden_activation: HiddenActivation::GeluTanh,
        deepstack_layers,
    })
}

/// The existing vision-tensor inventory implied by mmproj metadata.
/// # Errors
/// Rejects encoder geometry that the inventory cannot describe.
pub fn vision_slots(mmproj: &Metadata) -> Result<Vec<VisionSlot>> {
    slots(&encoder(mmproj)?)
}

/// Map one text GGUF tensor to the HF backbone name.
#[must_use]
pub fn text_tensor(name: &str) -> Option<String> {
    let mapped = match name {
        "token_embd.weight" => "model.embed_tokens.weight".to_owned(),
        "output_norm.weight" => "model.norm.weight".to_owned(),
        "output.weight" => "lm_head.weight".to_owned(),
        _ => {
            let rest = name.strip_prefix("blk.")?;
            let (layer, field) = rest.split_once('.')?;
            let field = match field {
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                _ => return None,
            };
            format!("model.layers.{layer}.{field}")
        }
    };
    Some(mapped)
}

/// Map one mmproj tensor to the HF vision inventory, or a `Conv3D` patch slice.
#[must_use]
pub fn vision_tensor(name: &str, deepstack_layers: &[u64]) -> Option<VisionTensor> {
    if let Some(rest) = name.strip_prefix("v.deepstack.") {
        let (layer, field) = rest.split_once('.')?;
        let index = deepstack_layers
            .iter()
            .position(|candidate| candidate.to_string() == layer)?;
        let field = match field {
            "norm.weight" => "norm.weight",
            "norm.bias" => "norm.bias",
            "fc1.weight" => "linear_fc1.weight",
            "fc1.bias" => "linear_fc1.bias",
            "fc2.weight" => "linear_fc2.weight",
            "fc2.bias" => "linear_fc2.bias",
            _ => return None,
        };
        return Some(VisionTensor::Slot(format!(
            "model.visual.deepstack_merger_list.{index}.{field}"
        )));
    }
    if name == "v.patch_embd.weight" {
        return Some(VisionTensor::PatchSlice(0));
    }
    if name == "v.patch_embd.weight.1" {
        return Some(VisionTensor::PatchSlice(1));
    }
    if name.starts_with("v.patch_embd.bias") {
        return Some(VisionTensor::Slot(
            "model.visual.patch_embed.proj.bias".to_owned(),
        ));
    }
    if name == "v.position_embd.weight" {
        return Some(VisionTensor::Slot(
            "model.visual.pos_embed.weight".to_owned(),
        ));
    }
    if name.starts_with("v.post_ln.") {
        return Some(VisionTensor::Slot(format!(
            "model.visual.merger.norm.{}",
            name.trim_start_matches("v.post_ln.")
        )));
    }
    if let Some(rest) = name.strip_prefix("mm.") {
        let (index, field) = rest.split_once('.')?;
        let target = match index {
            "0" => "merger.linear_fc1",
            "2" => "merger.linear_fc2",
            _ => return None,
        };
        return Some(VisionTensor::Slot(format!("model.visual.{target}.{field}")));
    }
    let rest = name.strip_prefix("v.blk.")?;
    let (layer, field) = rest.split_once('.')?;
    let field = match field {
        "attn_qkv.weight" => "attn.qkv.weight",
        "attn_qkv.bias" => "attn.qkv.bias",
        "attn_out.weight" => "attn.proj.weight",
        "attn_out.bias" => "attn.proj.bias",
        "ffn_up.weight" => "mlp.linear_fc1.weight",
        "ffn_up.bias" => "mlp.linear_fc1.bias",
        "ffn_down.weight" => "mlp.linear_fc2.weight",
        "ffn_down.bias" => "mlp.linear_fc2.bias",
        "ln1.weight" => "norm1.weight",
        "ln1.bias" => "norm1.bias",
        "ln2.weight" => "norm2.weight",
        "ln2.bias" => "norm2.bias",
        _ => return None,
    };
    Some(VisionTensor::Slot(format!(
        "model.visual.blocks.{layer}.{field}"
    )))
}

fn token_id(tokens: &[&str], marker: &str) -> Result<u32> {
    let index = tokens
        .iter()
        .position(|token| *token == marker)
        .ok_or_else(|| Error::invalid(format!("Qwen3-VL tokenizer is missing {marker}")))?;
    u32::try_from(index).map_err(|_| Error::invalid("Qwen3-VL token id exceeds u32"))
}

fn u64_array(metadata: &Metadata, key: &str) -> Result<Vec<u64>> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| Error::invalid(format!("GGUF metadata {key} is not an array")))?
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                Error::invalid(format!("GGUF metadata {key} is not an integer array"))
            })
        })
        .collect()
}

fn bool_array(metadata: &Metadata, key: &str) -> Result<Vec<bool>> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| Error::invalid(format!("GGUF metadata {key} is not an array")))?
        .iter()
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| Error::invalid(format!("GGUF metadata {key} is not a bool array")))
        })
        .collect()
}

#[cfg(test)]
mod tests;
