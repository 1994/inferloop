//! The adapter contracts one GGUF architecture or mmproj projector fills in, and the helpers
//! every adapter shares.
use super::tensor::{MappedTensor, TensorSpec, VisionTensor};
use infer_core::{Error, Result};
use infer_format::{Metadata, MetadataValue};
use infer_spi::ModalityEncoder;

/// How the stored blocks divide between the backbone and a speculative draft head.
///
/// llama.cpp stores a model's `nextn` (multi-token prediction) layers as trailing blocks and counts
/// them in `block_count`; `{arch}.nextn_predict_layers` says how many there are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPlan {
    pub backbone: usize,
    pub draft: usize,
}

impl LayerPlan {
    /// The plan `metadata` declares for `architecture`.
    /// # Errors
    /// Rejects a missing block count or more draft layers than blocks.
    pub fn of(metadata: &Metadata, architecture: &str) -> Result<Self> {
        let blocks = count(metadata, &format!("{architecture}.block_count"))?;
        let draft = metadata
            .get(&format!("{architecture}.nextn_predict_layers"))
            .and_then(MetadataValue::as_u64)
            .map_or(Ok(0), |layers| {
                usize::try_from(layers)
                    .map_err(|_| Error::invalid("GGUF nextn layers exceed usize"))
            })?;
        let backbone = blocks
            .checked_sub(draft)
            .filter(|layers| *layers > 0)
            .ok_or_else(|| Error::invalid("GGUF nextn layers leave no backbone"))?;
        Ok(Self { backbone, draft })
    }

    /// The block a stored `blk.{index}.{field}` name addresses.
    #[must_use]
    pub fn block<'a>(&self, name: &'a str) -> Option<Block<'a>> {
        let (index, field) = name.strip_prefix("blk.")?.split_once('.')?;
        let index: usize = index.parse().ok()?;
        let draft = index.checked_sub(self.backbone);
        if draft.is_some_and(|layer| layer >= self.draft) {
            return None;
        }
        Some(Block {
            index,
            draft,
            field,
        })
    }
}

/// One `blk.{index}.{field}` tensor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block<'a> {
    pub index: usize,
    /// The draft layer this block is, or `None` for a backbone layer.
    pub draft: Option<usize>,
    pub field: &'a str,
}

/// The HF field of a llama.cpp block tensor every decoder architecture names the same way.
#[must_use]
pub fn common_block_field(field: &str) -> Option<&'static str> {
    Some(match field {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_norm.weight" | "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    })
}

/// What one GGUF architecture fills in.
pub trait GgufArchitecture: Sync {
    /// The `general.architecture` value this adapter reads.
    fn name(&self) -> &'static str;
    /// The HF configuration of the text package, and of its mmproj when one is attached.
    /// `tensors` answers what the metadata does not record, such as a tied output projection.
    /// # Errors
    /// Rejects missing or malformed metadata.
    fn config(
        &self,
        text: &Metadata,
        tensors: &[TensorSpec],
        vision: Option<&Metadata>,
    ) -> Result<Vec<u8>>;
    /// Where a stored text tensor goes; `None` for a tensor the architecture does not know.
    fn tensor(&self, tensor: &TensorSpec, layers: &LayerPlan) -> Option<MappedTensor>;
    /// Architecture checks beyond "every tensor maps", before any tensor is read.
    /// # Errors
    /// Rejects geometry or storage the architecture cannot load.
    fn validate(&self, text: &Metadata, tensors: &[TensorSpec]) -> Result<()> {
        let _ = (text, tensors);
        Ok(())
    }
    /// Whether a sibling mmproj can attach to this architecture.
    fn accepts_vision(&self) -> bool {
        false
    }
}

/// What one mmproj projector (`clip.projector_type`) fills in.
pub trait GgufProjector: Sync {
    /// The `clip.projector_type` value this adapter reads.
    fn name(&self) -> &'static str;
    /// Check the stored tensor set against the projector geometry, before any tensor is read.
    /// # Errors
    /// Rejects a missing or extra tensor, a shape mismatch, or an unsupported type.
    fn validate(&self, metadata: &Metadata, tensors: &[TensorSpec]) -> Result<()>;
    /// The vision encoder the metadata declares.
    /// # Errors
    /// Rejects missing or malformed geometry.
    fn encoder(&self, metadata: &Metadata) -> Result<ModalityEncoder>;
    /// Where a stored tensor goes; `None` for one the encoder does not read.
    fn tensor(&self, name: &str, encoder: &ModalityEncoder) -> Option<VisionTensor>;
    /// The HF `preprocessor_config.json` of the encoder's image processor.
    /// # Errors
    /// Rejects missing or malformed normalization metadata.
    fn preprocessor(&self, metadata: &Metadata, encoder: &ModalityEncoder) -> Result<Vec<u8>>;
}

/// A count stored under `key`.
/// # Errors
/// Rejects a missing key or a value beyond `usize`.
pub fn count(metadata: &Metadata, key: &str) -> Result<usize> {
    usize::try_from(metadata.require_u64(key)?)
        .map_err(|_| Error::invalid(format!("GGUF metadata {key} exceeds usize")))
}

/// An integer array stored under `key`.
/// # Errors
/// Rejects a missing key or a non-integer element.
pub fn u64_array(metadata: &Metadata, key: &str) -> Result<Vec<u64>> {
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

/// A boolean array stored under `key`.
/// # Errors
/// Rejects a missing key or a non-boolean element.
pub fn bool_array(metadata: &Metadata, key: &str) -> Result<Vec<bool>> {
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

/// `values` without trailing zeros, as llama.cpp pads `RoPE` section arrays.
#[must_use]
pub fn trim_trailing_zeros(mut values: Vec<u64>) -> Vec<u64> {
    while values.last() == Some(&0) {
        values.pop();
    }
    values
}

/// The id of `marker` in the stored vocabulary.
/// # Errors
/// Rejects a vocabulary without the marker.
pub fn token_id(metadata: &Metadata, marker: &str) -> Result<u32> {
    let tokens = metadata.require_strings("tokenizer.ggml.tokens")?;
    let index = tokens
        .iter()
        .position(|token| *token == marker)
        .ok_or_else(|| Error::invalid(format!("GGUF tokenizer is missing {marker}")))?;
    u32::try_from(index).map_err(|_| Error::invalid("GGUF token id exceeds u32"))
}
