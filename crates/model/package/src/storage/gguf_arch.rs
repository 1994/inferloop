//! GGUF architecture adapters: what one `general.architecture` fills in on the shared GGUF path.
//!
//! The platform owns the container, the tokenizer, the llama.cpp names every architecture shares,
//! the split of trailing `nextn` blocks into a speculative draft head, value transforms and the
//! rule that every stored tensor maps somewhere. An adapter only declares what its architecture
//! adds: the HF configuration it synthesizes, its own tensor names and the inverse of what its
//! converter changed. The result is an ordinary HF configuration and canonical tensor names, so
//! providers, recipes and backends are the same for every checkpoint format.
use crate::providers::dflash2::DFLASH2;
use crate::providers::qwen::gguf::validate::TensorSpec;
use crate::providers::qwen::gguf::{QWEN3VL, QWEN35};
use infer_core::{Error, Result};
use infer_format::{Metadata, MetadataValue};

/// The inverse of a value change a converter applied, undone when a float tensor is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTransform {
    /// Stored as is.
    Identity,
    /// Stored as `1 + w`, as llama.cpp stores the norms of families that scale by `1 + w`.
    PlusOne,
    /// Stored as `-exp(x)`, as llama.cpp stores gated-delta `A_log`.
    NegatedExp,
}

impl ValueTransform {
    /// Recover the original values in place.
    pub fn undo(self, values: &mut [f32]) {
        match self {
            Self::Identity => {}
            Self::PlusOne => values.iter_mut().for_each(|value| *value -= 1.0),
            Self::NegatedExp => values.iter_mut().for_each(|value| *value = (-*value).ln()),
        }
    }
}

/// Where one stored tensor goes and how to read it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedTensor {
    /// Canonical HF name.
    pub slot: String,
    pub transform: ValueTransform,
    /// The canonical shape when the converter reshaped the tensor; `None` keeps the stored one.
    pub shape: Option<Vec<usize>>,
}

impl MappedTensor {
    /// A tensor stored under its canonical meaning.
    #[must_use]
    pub fn named(slot: impl Into<String>) -> Self {
        Self {
            slot: slot.into(),
            transform: ValueTransform::Identity,
            shape: None,
        }
    }

    /// The same mapping with a value transform to undo.
    #[must_use]
    pub const fn transformed(mut self, transform: ValueTransform) -> Self {
        self.transform = transform;
        self
    }

    /// The same mapping with a canonical shape.
    #[must_use]
    pub fn reshaped(mut self, shape: Vec<usize>) -> Self {
        self.shape = Some(shape);
        self
    }
}

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

/// Every architecture the GGUF path reads.
const ARCHITECTURES: [&dyn GgufArchitecture; 3] = [&QWEN3VL, &QWEN35, &DFLASH2];

/// The adapter for `metadata`'s `general.architecture`.
/// # Errors
/// Rejects an architecture no adapter declares, naming the ones that exist.
pub fn architecture(metadata: &Metadata) -> Result<&'static dyn GgufArchitecture> {
    let name = metadata.require_str("general.architecture")?;
    ARCHITECTURES
        .iter()
        .copied()
        .find(|architecture| architecture.name() == name)
        .ok_or_else(|| {
            let known: Vec<_> = ARCHITECTURES.iter().map(|known| known.name()).collect();
            Error::unsupported(format!(
                "GGUF architecture {name} has no adapter (known: {})",
                known.join(", ")
            ))
        })
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

#[cfg(test)]
#[path = "../../tests/unit/gguf_arch.rs"]
mod tests;
