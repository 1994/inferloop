use crate::{DType, Modality};
use infer_core::{Error, ModelId, Result};
use serde::{Deserialize, Serialize};

/// Which key head each value head of a gated-delta layer reads.
///
/// It matters when value heads outnumber key heads, and it is a weight layout: every value-side tensor (the value rows of the fused input
/// projection, the gate, `a`/`b`, `A_log`, `dt_bias`, the value convolution channels and the
/// output projection's columns) is stored in this order, so the recurrence only has to look its
/// key up the same way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueHeadOrder {
    /// Value head `v` reads key head `v / (value_heads / key_heads)`: each key head's value
    /// heads are adjacent, as Hugging Face checkpoints store them.
    #[default]
    Grouped,
    /// Value head `v` reads key head `v % key_heads`: the key heads repeat, as llama.cpp's
    /// converter stores them for broadcasting.
    Tiled,
}

impl ValueHeadOrder {
    /// How many adjacent value heads share one key head, so value head `v` reads key head
    /// `v / key_run % key_heads` in either order.
    #[must_use]
    pub const fn key_run(self, key_heads: usize, value_heads: usize) -> usize {
        match self {
            Self::Grouped if key_heads > 0 => value_heads / key_heads,
            Self::Grouped | Self::Tiled => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackboneKind {
    Encoder,
    Decoder,
    EncoderDecoder,
    Hybrid,
    Recurrent,
    Multimodal,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mixer {
    Attention {
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
        output_gate: bool,
        qk_norm: bool,
    },
    LinearAttention {
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        #[serde(default)]
        value_order: ValueHeadOrder,
    },
    Recurrent {
        state_width: usize,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedForward {
    Dense {
        intermediate: usize,
    },
    Moe {
        experts: usize,
        active: usize,
        intermediate: usize,
        #[serde(default)]
        normalize_topk: bool,
        #[serde(default)]
        shared_intermediate: Option<usize>,
        #[serde(default)]
        dense_layers: Vec<usize>,
        #[serde(default)]
        dense_intermediate: usize,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionSpec {
    pub rope_theta: f64,
    pub rotary_fraction: f64,
    pub multimodal_sections: Vec<usize>,
    pub interleaved: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateKind {
    AttentionKv,
    LinearAttention,
    Recurrent,
    Ssm,
    Conv,
    Speculation,
    Multimodal,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRequirement {
    pub layer: usize,
    pub kind: StateKind,
    pub dtype: DType,
    pub elements: usize,
    pub per_token: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Head {
    LanguageModel,
    Embedding,
    Rank,
    Decision,
    Classification,
    Reward,
    LateInteraction,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelIr {
    pub id: ModelId,
    pub backbone: BackboneKind,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub max_sequence: usize,
    pub mixers: Vec<Mixer>,
    pub feed_forward: FeedForward,
    pub position: PositionSpec,
    pub norm_epsilon: f32,
    pub norm_weight_offset: f32,
    pub heads: Vec<Head>,
    pub modalities: Vec<Modality>,
    pub state: Vec<StateRequirement>,
    pub tied_embeddings: bool,
    /// Leading decoder layers whose output adds one per-token media feature, in layer order
    /// (vision deepstack); zero for families without them.
    #[serde(default)]
    pub media_features: usize,
}
impl ModelIr {
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size < 2
            || self.hidden_size == 0
            || self.max_sequence == 0
            || self.mixers.is_empty()
            || self.heads.is_empty()
        {
            return Err(Error::invalid(
                "invalid model dimensions or empty layers/heads",
            ));
        }
        if !self.norm_weight_offset.is_finite()
            || !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || !self.position.rope_theta.is_finite()
            || self.position.rope_theta <= 0.0
            || !self.position.rotary_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.position.rotary_fraction)
        {
            return Err(Error::invalid(
                "invalid normalization or position parameters",
            ));
        }
        for mixer in &self.mixers {
            match mixer {
                Mixer::Attention {
                    query_heads,
                    kv_heads,
                    head_dim,
                    sliding_window,
                    ..
                } if *query_heads == 0
                    || *kv_heads == 0
                    || *head_dim == 0
                    || query_heads % kv_heads != 0
                    || *sliding_window == Some(0) =>
                {
                    return Err(Error::invalid("invalid attention dimensions"));
                }
                Mixer::LinearAttention {
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                    conv_kernel,
                    ..
                } if [*key_heads, *value_heads, *key_dim, *value_dim, *conv_kernel]
                    .contains(&0)
                    || value_heads % key_heads != 0 =>
                {
                    return Err(Error::invalid("invalid linear attention dimensions"));
                }
                Mixer::Recurrent { state_width: 0 } => {
                    return Err(Error::invalid("empty recurrent state"));
                }
                _ => {}
            }
        }
        match &self.feed_forward {
            FeedForward::Dense { intermediate: 0 } => return Err(Error::invalid("empty FFN")),
            FeedForward::Moe {
                experts,
                active,
                intermediate,
                shared_intermediate,
                dense_layers,
                dense_intermediate,
                ..
            } if *experts == 0
                || *active == 0
                || active > experts
                || *intermediate == 0
                || *shared_intermediate == Some(0)
                || dense_layers.iter().any(|layer| *layer >= self.mixers.len())
                || (!dense_layers.is_empty() && *dense_intermediate == 0) =>
            {
                return Err(Error::invalid("invalid MoE"));
            }
            _ => {}
        }
        if self.media_features > self.mixers.len() {
            return Err(Error::invalid("media features exceed the decoder layers"));
        }
        for state in &self.state {
            if state.layer >= self.mixers.len() || state.elements == 0 {
                return Err(Error::invalid("invalid state requirement"));
            }
        }
        Ok(())
    }
}
