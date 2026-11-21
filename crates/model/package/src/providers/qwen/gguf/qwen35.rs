//! The `qwen35` GGUF architecture: Qwen3.5's hybrid gated-delta backbone and its `nextn` MTP head.
//!
//! llama.cpp's converter changes four things this adapter undoes: every norm except the gated
//! delta one is stored as `1 + w`, `A_log` as `-exp(A_log)`, the depthwise convolution without its
//! singleton axis, and the MTP layer as a trailing `nextn` block. A fifth it declares instead:
//! with more value heads than key heads, every value-side tensor is stored in tiled head order,
//! down to the columns of the quantized output projection, so no load-time permutation can undo it.
use crate::storage::gguf::{
    Block, GgufArchitecture, LayerPlan, MappedTensor, TensorSpec, ValueTransform,
    common_block_field, count, token_id, trim_trailing_zeros, u64_array,
};
use infer_core::{Error, Result};
use infer_format::Metadata;
use infer_ir::ValueHeadOrder;
use serde::Serialize;

const ARCH: &str = "qwen35";
/// HF prefix of the text backbone, as the released multimodal checkpoints nest it.
const BACKBONE: &str = "model.language_model.";
/// HF prefix of the MTP head.
const MTP: &str = "mtp.";

/// The `qwen35` architecture.
pub struct Qwen35Gguf;

/// The `qwen35` adapter.
pub const QWEN35: Qwen35Gguf = Qwen35Gguf;

#[derive(Serialize)]
struct RopeParameters {
    rope_type: &'static str,
    rope_theta: f64,
    partial_rotary_factor: f64,
    mrope_section: Vec<u64>,
    mrope_interleaved: bool,
}

#[derive(Serialize)]
struct TextConfig {
    model_type: &'static str,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    vocab_size: usize,
    max_position_embeddings: usize,
    hidden_act: &'static str,
    attn_output_gate: bool,
    full_attention_interval: usize,
    layer_types: Vec<&'static str>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    linear_value_head_order: ValueHeadOrder,
    mtp_num_hidden_layers: usize,
    tie_word_embeddings: bool,
    rope_parameters: RopeParameters,
}

/// The HF `config.json` this adapter synthesizes.
#[derive(Serialize)]
struct Synthesized {
    architectures: [&'static str; 1],
    model_type: &'static str,
    image_token_id: u32,
    video_token_id: u32,
    tie_word_embeddings: bool,
    language_model_only: bool,
    text_config: TextConfig,
}

/// The linear-attention geometry `qwen35.ssm.*` declares.
struct Linear {
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
}

impl Linear {
    /// llama.cpp tiles the value heads whenever they outnumber the key heads.
    const fn order(&self) -> ValueHeadOrder {
        if self.key_heads == self.value_heads {
            ValueHeadOrder::Grouped
        } else {
            ValueHeadOrder::Tiled
        }
    }

    fn of(metadata: &Metadata) -> Result<Self> {
        let value_heads = count(metadata, "qwen35.ssm.time_step_rank")?;
        let inner = count(metadata, "qwen35.ssm.inner_size")?;
        if value_heads == 0 || !inner.is_multiple_of(value_heads) {
            return Err(Error::invalid(
                "qwen35 inner size must divide by value heads",
            ));
        }
        Ok(Self {
            key_heads: count(metadata, "qwen35.ssm.group_count")?,
            value_heads,
            key_dim: count(metadata, "qwen35.ssm.state_size")?,
            value_dim: inner / value_heads,
        })
    }
}

impl GgufArchitecture for Qwen35Gguf {
    fn name(&self) -> &'static str {
        ARCH
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "the rotary fraction is a ratio of two small head widths"
    )]
    fn config(
        &self,
        text: &Metadata,
        tensors: &[TensorSpec],
        _vision: Option<&Metadata>,
    ) -> Result<Vec<u8>> {
        let layers = LayerPlan::of(text, ARCH)?;
        let interval = count(text, "qwen35.full_attention_interval")?;
        if interval == 0 {
            return Err(Error::invalid(
                "qwen35 full attention interval must be positive",
            ));
        }
        let head_dim = count(text, "qwen35.attention.key_length")?;
        let rotary = count(text, "qwen35.rope.dimension_count")?;
        let linear = Linear::of(text)?;
        // A tied checkpoint stores no separate output projection.
        let tied = !tensors.iter().any(|tensor| tensor.name == "output.weight");
        let config = Synthesized {
            architectures: ["Qwen3_5ForConditionalGeneration"],
            model_type: "qwen3_5",
            image_token_id: token_id(text, "<|image_pad|>")?,
            video_token_id: token_id(text, "<|video_pad|>")?,
            tie_word_embeddings: tied,
            language_model_only: true,
            text_config: TextConfig {
                model_type: "qwen3_5_text",
                hidden_size: count(text, "qwen35.embedding_length")?,
                intermediate_size: count(text, "qwen35.feed_forward_length")?,
                num_hidden_layers: layers.backbone,
                num_attention_heads: count(text, "qwen35.attention.head_count")?,
                num_key_value_heads: count(text, "qwen35.attention.head_count_kv")?,
                head_dim,
                rms_norm_eps: text.require_f64("qwen35.attention.layer_norm_rms_epsilon")?,
                vocab_size: text.require_strings("tokenizer.ggml.tokens")?.len(),
                max_position_embeddings: count(text, "qwen35.context_length")?,
                hidden_act: "silu",
                attn_output_gate: true,
                full_attention_interval: interval,
                layer_types: (1..=layers.backbone)
                    .map(|layer| {
                        if layer.is_multiple_of(interval) {
                            "full_attention"
                        } else {
                            "linear_attention"
                        }
                    })
                    .collect(),
                linear_conv_kernel_dim: count(text, "qwen35.ssm.conv_kernel")?,
                linear_key_head_dim: linear.key_dim,
                linear_num_key_heads: linear.key_heads,
                linear_num_value_heads: linear.value_heads,
                linear_value_head_dim: linear.value_dim,
                linear_value_head_order: linear.order(),
                mtp_num_hidden_layers: layers.draft,
                tie_word_embeddings: tied,
                rope_parameters: RopeParameters {
                    rope_type: "default",
                    rope_theta: text.require_f64("qwen35.rope.freq_base")?,
                    partial_rotary_factor: rotary as f64 / head_dim as f64,
                    mrope_section: trim_trailing_zeros(u64_array(
                        text,
                        "qwen35.rope.dimension_sections",
                    )?),
                    // The family interleaves its multimodal sections; the file does not say so.
                    mrope_interleaved: true,
                },
            },
        };
        serde_json::to_vec(&config).map_err(|error| Error::invalid(error.to_string()))
    }

    fn tensor(&self, tensor: &TensorSpec, layers: &LayerPlan) -> Option<MappedTensor> {
        let name = tensor.name.as_str();
        let norm = |slot: String| MappedTensor::named(slot).transformed(ValueTransform::PlusOne);
        match name {
            "token_embd.weight" => {
                return Some(MappedTensor::named(format!(
                    "{BACKBONE}embed_tokens.weight"
                )));
            }
            "output_norm.weight" => return Some(norm(format!("{BACKBONE}norm.weight"))),
            "output.weight" => return Some(MappedTensor::named("lm_head.weight")),
            _ => {}
        }
        let Block {
            index,
            draft,
            field,
        } = layers.block(name)?;
        let layer = match draft {
            Some(draft) => {
                if let Some((head, transform)) = nextn_field(field) {
                    return Some(
                        MappedTensor::named(format!("{MTP}{head}")).transformed(transform),
                    );
                }
                format!("{MTP}layers.{draft}.")
            }
            None => format!("{BACKBONE}layers.{index}."),
        };
        if let Some(common) = common_block_field(field) {
            let mapped = MappedTensor::named(format!("{layer}{common}"));
            return Some(if common.ends_with("norm.weight") {
                mapped.transformed(ValueTransform::PlusOne)
            } else {
                mapped
            });
        }
        let linear = format!("{layer}linear_attn.");
        Some(match field {
            "attn_qkv.weight" => MappedTensor::named(format!("{linear}in_proj_qkv.weight")),
            "attn_gate.weight" => MappedTensor::named(format!("{linear}in_proj_z.weight")),
            "ssm_alpha.weight" => MappedTensor::named(format!("{linear}in_proj_a.weight")),
            "ssm_beta.weight" => MappedTensor::named(format!("{linear}in_proj_b.weight")),
            "ssm_out.weight" => MappedTensor::named(format!("{linear}out_proj.weight")),
            "ssm_norm.weight" => MappedTensor::named(format!("{linear}norm.weight")),
            "ssm_dt.bias" => MappedTensor::named(format!("{linear}dt_bias")),
            "ssm_a" => MappedTensor::named(format!("{linear}A_log"))
                .transformed(ValueTransform::NegatedExp),
            // Stored `[channels, kernel]`; HF keeps the depthwise `[channels, 1, kernel]` axis.
            "ssm_conv1d.weight" => {
                let [channels, kernel] = tensor.shape.as_slice() else {
                    return None;
                };
                MappedTensor::named(format!("{linear}conv1d.weight"))
                    .reshaped(vec![*channels, 1, *kernel])
            }
            _ => return None,
        })
    }

    fn validate(&self, text: &Metadata, tensors: &[TensorSpec]) -> Result<()> {
        let linear = Linear::of(text)?;
        if linear.key_heads == 0 || !linear.value_heads.is_multiple_of(linear.key_heads) {
            return Err(Error::invalid(
                "qwen35 value heads must be a multiple of key heads",
            ));
        }
        let layers = LayerPlan::of(text, ARCH)?;
        if layers.draft > 1 {
            return Err(Error::unsupported(
                "qwen35 GGUF declares more than one nextn layer",
            ));
        }
        if tensors.is_empty() {
            return Err(Error::invalid("qwen35 GGUF stores no tensors"));
        }
        Ok(())
    }
}

/// The MTP-head tensor a `nextn` field names, relative to [`MTP`], and its stored form: the
/// three head norms are stored as `1 + w` like every other Qwen3.5 norm.
fn nextn_field(field: &str) -> Option<(&'static str, ValueTransform)> {
    Some(match field {
        "nextn.eh_proj.weight" => ("fc.weight", ValueTransform::Identity),
        "nextn.enorm.weight" => ("pre_fc_norm_embedding.weight", ValueTransform::PlusOne),
        "nextn.hnorm.weight" => ("pre_fc_norm_hidden.weight", ValueTransform::PlusOne),
        "nextn.shared_head_norm.weight" => ("norm.weight", ValueTransform::PlusOne),
        _ => return None,
    })
}
