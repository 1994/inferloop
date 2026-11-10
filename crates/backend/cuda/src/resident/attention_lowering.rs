//! One lowering for every attention node. The tuned choice, what the workspace prepared at
//! load and whether the pool's operands fit the selected kernel decide what records before
//! anything does, so single-lane decode, grouped verification and prompt tiles cannot drift
//! apart: each passes its mode and records the decision.
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use crate::tuning::attention::Choice;
use infer_cuda_kernels::contract::VERIFY_LANES;

/// What the attention workspace prepared at load for one `(heads, dim)` geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Prepared {
    /// Split partials exist for this geometry.
    pub split: bool,
    /// A grouped launch and its stacked partials serve three verification lanes.
    pub grouped_verify: bool,
}

/// How a capture mode presents an attention node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    /// One lane's query over its cache.
    Decode,
    /// The lanes of one verification step. `operands` says whether the node's query and cache
    /// can be shared across them: a dynamic query, an FP8 cache with scales, no checkpoints.
    Verify { lanes: usize, operands: bool },
    /// Prompt tiles of the batched prefill. `selected` is the tuned choice for this output and
    /// lane count, and `fits` whether its kernel was loaded and accepts the pool's operands.
    Prefill {
        selected: Option<Choice>,
        fp8: bool,
        fits: bool,
    },
}

/// What records one attention node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lowering {
    /// Split-KV decode through the geometry's partials.
    Split,
    /// The contract's single-pass decode.
    Decode,
    /// One grouped launch reads each KV block once for the three verification lanes.
    Grouped,
    /// Each verification lane records its own decode.
    PerLane,
    /// The selected tensor-core prefill kernel.
    Mma,
    /// The selected Rust SIMT prefill kernel.
    Simt,
    /// The contract's prefill decode with a tuned query tile.
    Tiled(usize),
    /// The contract's prefill decode.
    Prefill,
}

/// Lower an attention node over a sliding `window` (none for full attention) in `mode`.
pub(super) fn lower(mode: Mode, window: Option<usize>, prepared: Prepared) -> Lowering {
    // Split partials only pay for their reduction over a long enough span.
    let spans = window.is_none_or(|w| w >= ATTENTION_SPLIT_THRESHOLD);
    match mode {
        Mode::Decode => {
            if prepared.split && spans {
                Lowering::Split
            } else {
                Lowering::Decode
            }
        }
        Mode::Verify { lanes, operands } => {
            if lanes == VERIFY_LANES && spans && prepared.grouped_verify && operands {
                Lowering::Grouped
            } else {
                Lowering::PerLane
            }
        }
        Mode::Prefill {
            selected,
            fp8,
            fits,
        } => match selected {
            Some(Choice::Mma { .. } | Choice::MmaF16) if fits => Lowering::Mma,
            Some(choice) if choice.uses_oxide() && fits => Lowering::Simt,
            // The tiled kernel reads quantized pages only.
            Some(choice) if fp8 => choice
                .query_tile()
                .map_or(Lowering::Prefill, Lowering::Tiled),
            _ => Lowering::Prefill,
        },
    }
}

#[cfg(test)]
#[path = "../../tests/unit/attention_lowering.rs"]
mod tests;
