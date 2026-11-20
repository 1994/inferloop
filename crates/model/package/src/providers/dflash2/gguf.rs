//! The `dflash` GGUF architecture: a `DFlash2` draft as llama.cpp stores it.
//!
//! The file records the draft's own geometry and its tapped target layers, numbered from the
//! embedding output (one above HF's `target_layer_ids`), but not how deep the target is: the
//! synthesized configuration leaves `num_target_layers` to the target it is attached to.
use super::ARCHITECTURE;
use crate::providers::qwen::gguf::validate::TensorSpec;
use crate::storage::gguf_arch::{
    Block, GgufArchitecture, LayerPlan, MappedTensor, common_block_field, count, u64_array,
};
use infer_core::{Error, Result};
use infer_format::{Metadata, MetadataValue};
use serde::Serialize;

const ARCH: &str = "dflash";

/// The `dflash` architecture.
pub struct DFlash2Gguf;

/// The `dflash` adapter.
pub const DFLASH2: DFlash2Gguf = DFlash2Gguf;

#[derive(Serialize)]
struct Rope {
    rope_type: &'static str,
    rope_theta: f64,
}

#[derive(Serialize)]
struct DraftBlock {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    selector_rank: usize,
    selector_top_k: usize,
    mask_token_id: usize,
    target_layer_ids: Vec<usize>,
}

/// The HF `config.json` this adapter synthesizes, without `num_target_layers`.
#[derive(Serialize)]
struct Synthesized {
    architectures: [&'static str; 1],
    model_type: &'static str,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    max_position_embeddings: usize,
    is_causal: bool,
    tie_word_embeddings: bool,
    pad_token_id: Option<usize>,
    rms_norm_eps: f64,
    rope_parameters: Rope,
    dflash_config: DraftBlock,
}

fn key(name: &str) -> String {
    format!("{ARCH}.{name}")
}

fn token(metadata: &Metadata, name: &str) -> Result<Option<usize>> {
    metadata
        .get(&format!("tokenizer.ggml.{name}"))
        .and_then(MetadataValue::as_u64)
        .map(|id| usize::try_from(id).map_err(|_| Error::invalid("GGUF token id exceeds usize")))
        .transpose()
}

impl GgufArchitecture for DFlash2Gguf {
    fn name(&self) -> &'static str {
        ARCH
    }

    fn config(
        &self,
        text: &Metadata,
        _tensors: &[TensorSpec],
        _vision: Option<&Metadata>,
    ) -> Result<Vec<u8>> {
        let taps = u64_array(text, &key("target_layers"))?
            .into_iter()
            .map(|layer| {
                usize::try_from(layer)
                    .ok()
                    .and_then(|layer| layer.checked_sub(1))
                    .ok_or_else(|| Error::invalid("dflash target layer 0 taps the embedding"))
            })
            .collect::<Result<_>>()?;
        let mask = token(text, "mask_token_id")?
            .ok_or_else(|| Error::invalid("dflash GGUF declares no mask token"))?;
        let config = Synthesized {
            architectures: [ARCHITECTURE],
            model_type: "qwen3",
            hidden_size: count(text, &key("embedding_length"))?,
            intermediate_size: count(text, &key("feed_forward_length"))?,
            vocab_size: text.require_strings("tokenizer.ggml.tokens")?.len(),
            num_hidden_layers: count(text, &key("block_count"))?,
            num_attention_heads: count(text, &key("attention.head_count"))?,
            num_key_value_heads: count(text, &key("attention.head_count_kv"))?,
            head_dim: count(text, &key("attention.key_length"))?,
            sliding_window: count(text, &key("attention.sliding_window"))?,
            max_position_embeddings: count(text, &key("context_length"))?,
            is_causal: false,
            tie_word_embeddings: false,
            // llama.cpp records the chat end as EOS rather than the HF `eos_token_id`, so the
            // draft declares none and leaves end-of-sequence to its target.
            pad_token_id: token(text, "padding_token_id")?,
            rms_norm_eps: text.require_f64(&key("attention.layer_norm_rms_epsilon"))?,
            rope_parameters: Rope {
                rope_type: "default",
                rope_theta: text.require_f64(&key("rope.freq_base"))?,
            },
            dflash_config: DraftBlock {
                block_size: count(text, &key("block_size"))?,
                conv_group_size: count(text, &key("conv_group_size"))?,
                conv_kernel_size: count(text, &key("conv_kernel_size"))?,
                selector_rank: count(text, &key("selector_rank"))?,
                selector_top_k: count(text, &key("selector_top_k"))?,
                mask_token_id: mask,
                target_layer_ids: taps,
            },
        };
        serde_json::to_vec(&config).map_err(|error| Error::invalid(error.to_string()))
    }

    fn tensor(&self, tensor: &TensorSpec, layers: &LayerPlan) -> Option<MappedTensor> {
        let slot = match tensor.name.as_str() {
            "selector_hidden.weight" => "candidate_selector.hidden_projection.weight",
            "selector_predecessor.weight" => "candidate_selector.predecessor_codebook",
            "selector_successor.weight" => "candidate_selector.successor_codebook",
            "fc.weight" => "fc.weight",
            "enc.output_norm.weight" => "hidden_norm.weight",
            "output_norm.weight" => "norm.weight",
            name => {
                let Block {
                    index,
                    draft,
                    field,
                } = layers.block(name)?;
                if draft.is_some() {
                    return None;
                }
                let field = common_block_field(field).or(match field {
                    "attn_conv_base" => Some("attention_conv.base_kernel"),
                    "attn_conv_proj.weight" => Some("attention_conv.kernel_projection.weight"),
                    "ffn_conv_base" => Some("mlp_conv.base_kernel"),
                    "ffn_conv_proj.weight" => Some("mlp_conv.kernel_projection.weight"),
                    _ => None,
                })?;
                return Some(MappedTensor::named(format!("layers.{index}.{field}")));
            }
        };
        Some(MappedTensor::named(slot))
    }

    fn validate(&self, text: &Metadata, _tensors: &[TensorSpec]) -> Result<()> {
        let causal = text
            .get(&key("attention.causal"))
            .and_then(MetadataValue::as_bool)
            .unwrap_or(true);
        let sliding = text
            .get(&key("attention.sliding_window_pattern"))
            .and_then(MetadataValue::as_array)
            .is_some_and(|pattern| pattern.iter().all(|layer| layer.as_bool() == Some(true)));
        if causal || !sliding {
            return Err(Error::unsupported(
                "dflash GGUF must declare non-causal sliding attention in every layer",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/gguf_dflash2.rs"]
mod tests;
