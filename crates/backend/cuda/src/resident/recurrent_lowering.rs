//! One lowering for every recurrent node. The node's recurrence, the capture mode and what the
//! prompt workspace prepared decide what records before anything does, so per-lane decode, the
//! pooled replay and token-chunk prefill cannot drift apart: each passes its mode and records
//! the decision.
use super::capture::error;
use super::recurrent_prefill::{ORDERED_BLOCK, ORDERED_DIM, ORDERED_WIDTHS};
use crate::constants::CONV_KERNEL_SIZE;
use infer_core::Result;
use infer_ir::TensorOp;

/// The geometry of a gated delta recurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Delta {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl Delta {
    /// Square power-of-two heads, with whole groups of value heads per key head: what every
    /// resident delta kernel addresses.
    pub(super) const fn supported(self) -> bool {
        self.key_dim == self.value_dim
            && self.key_dim.is_power_of_two()
            && self.key_heads > 0
            && self.value_heads.is_multiple_of(self.key_heads)
    }
}

/// A recurrent node's operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Recurrence {
    /// A causal depthwise convolution over `channels` with the resident kernel width.
    Conv {
        channels: usize,
    },
    Delta(Delta),
}

impl Recurrence {
    /// The recurrence `op` describes, or `None` for any other operation.
    /// # Errors
    /// A convolution width or delta geometry no resident kernel addresses.
    pub(super) fn of(op: &TensorOp) -> Result<Option<Self>> {
        match *op {
            TensorOp::Conv { channels, kernel } => {
                if kernel != CONV_KERNEL_SIZE {
                    return Err(error("resident convolution width"));
                }
                Ok(Some(Self::Conv { channels }))
            }
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                let delta = Delta {
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                };
                if !delta.supported() {
                    return Err(error("resident delta dimensions"));
                }
                Ok(Some(Self::Delta(delta)))
            }
            _ => Ok(None),
        }
    }
}

/// What the prompt workspace prepared for one recurrent node of a `lanes`-row chunk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Chunk {
    pub lanes: usize,
    /// The node's projected inputs are activations, not constants.
    pub dynamic: bool,
    /// Head-major scratch exists for this delta's value geometry.
    pub scratch: bool,
    /// The WY chunk path is preferred, so only nodes with a tuned block take it.
    pub prefer_wy: bool,
    /// The tuned or ordered value block of this delta, if any.
    pub wy_block: Option<usize>,
    /// Internal tests compare against the per-row recurrence.
    pub rows_only: bool,
    /// Internal tests compare against the per-row delta with the chunked convolution.
    pub rows_delta: bool,
}

/// How a capture mode presents a recurrent node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    /// One lane's row against its own state.
    Lane,
    /// The pooled graph over state roots, so one replay advances every busy lane.
    Pooled,
    /// A prompt chunk.
    Prompt(Chunk),
}

/// The value-column block of a chunked delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Block {
    /// A measured or ordered block width.
    Fixed(usize),
    /// The per-lane path's value split.
    Split,
}

/// What records one recurrent node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lowering {
    /// One lane's step.
    Step,
    /// One launch over every lane's pooled state.
    Lanes,
    /// The convolution over a prompt chunk's rows.
    ConvPrefill,
    /// The delta over a prompt chunk with state held across rows. `wy` takes the chunked WY
    /// form.
    Chunked { block: Block, wy: bool },
    /// Each prompt row records its own step.
    Rows,
}

/// Lower `recurrence` captured in `mode`.
pub(super) fn lower(recurrence: Recurrence, mode: Mode) -> Lowering {
    match (mode, recurrence) {
        (Mode::Lane, _) => Lowering::Step,
        (Mode::Pooled, _) => Lowering::Lanes,
        (Mode::Prompt(chunk), _) if chunk.rows_only || !chunk.dynamic => Lowering::Rows,
        (Mode::Prompt(_), Recurrence::Conv { .. }) => Lowering::ConvPrefill,
        (Mode::Prompt(chunk), Recurrence::Delta(delta)) => {
            if chunk.rows_delta || !chunk.scratch || (chunk.prefer_wy && chunk.wy_block.is_none()) {
                return Lowering::Rows;
            }
            // The measured ordered path splits independent value columns without changing sums.
            // Other shapes use their existing qualified block policy.
            let ordered = delta.key_dim == ORDERED_DIM && ORDERED_WIDTHS.contains(&chunk.lanes);
            let block = if ordered {
                Block::Fixed(ORDERED_BLOCK)
            } else {
                chunk.wy_block.map_or(Block::Split, Block::Fixed)
            };
            Lowering::Chunked {
                block,
                wy: chunk.wy_block.is_some(),
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/recurrent_lowering.rs"]
mod tests;
