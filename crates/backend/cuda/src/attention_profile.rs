//! Explicit numerical profiles for the target prefill attention path.
//!
//! The profile is resolved once when the model loads and is immutable for every sequence the
//! loaded model creates. It never changes weights, KV storage or the draft proposal path.
use serde::Serialize;
#[cfg(all(target_os = "linux", feature = "cuda"))]
use std::env::var;

/// Initial context kept on F32 before the target BF16 bulk path.
pub const PRECISE_PREFIX_POSITIONS: usize = 2048;

/// Zero is an explicit experiment opt-out through `INFER_CUDA_ATTENTION_PRECISE_PREFIX=0`.
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[must_use]
pub fn precise_prefix_positions() -> usize {
    let opt_out = var("INFER_CUDA_ATTENTION_PRECISE_PREFIX").is_ok_and(|v| v == "0");
    precise_prefix_from_opt_out(opt_out)
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
const fn precise_prefix_from_opt_out(opt_out: bool) -> usize {
    if opt_out { 0 } else { PRECISE_PREFIX_POSITIONS }
}

/// Target prefill attention arithmetic selected by the operator.
///
/// `Canonical` keeps the established F32 path. `Bf16` keeps initial context through 2048
/// positions on that path and uses FA4 BF16-QKV/F32 accumulation for bulk prefill afterwards.
/// The CLI pairs it with MTP only when the package and device support both; library callers
/// retain the canonical default. `Fp16` remains an explicit experimental SM120 profile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum AttentionProfile {
    /// Existing canonical attention, unchanged.
    #[default]
    Canonical,
    /// Canonical initial context followed by FA4 BF16 target FP8-KV bulk prefill.
    Bf16,
    /// FA4 plain-FP16 attention for validated SM120 target FP8-KV prefill; not quality-approved.
    Fp16,
}
impl AttentionProfile {
    /// Whether every target attention node has an available BF16 prefill implementation.
    /// This is an allocation-free geometry/artifact check; the CLI separately requires a
    /// native FP4 package, declared FP8 KV and a complete supported optional head.
    /// A zero width denotes the loader's automatic, lane-aligned prompt geometry.
    #[must_use]
    pub fn supports_bf16_prefill(
        architecture: &str,
        graph: &infer_ir::DataflowGraph,
        width: usize,
    ) -> bool {
        use crate::resident::attention_mma::catalog;
        let width = if width == 0 {
            catalog::LANE_MULTIPLE
        } else {
            width
        };
        let mut found = false;
        for node in &graph.nodes {
            if let infer_ir::TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window,
            } = node.op
            {
                found = true;
                if node.states.is_empty()
                    || !catalog::supported(
                        architecture,
                        query_heads,
                        kv_heads,
                        head_dim,
                        width,
                        window.unwrap_or(0),
                    )
                {
                    return false;
                }
            }
        }
        found
    }

    /// Stable readback/log name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
        }
    }
    /// Whether the BF16 target profile is selected.
    #[must_use]
    pub const fn is_bf16(self) -> bool {
        matches!(self, Self::Bf16)
    }
    /// Whether the explicit FP16 target profile is selected.
    #[must_use]
    pub const fn is_fp16(self) -> bool {
        matches!(self, Self::Fp16)
    }
    /// Whether a target prefill node may use a 16-bit FA4 profile.
    ///
    /// The caller still decides node-level FP8-KV, architecture and geometry eligibility; this
    /// only separates canonical attention from the target FA4 profiles.
    #[must_use]
    pub const fn is_target_mma(self) -> bool {
        matches!(self, Self::Bf16 | Self::Fp16)
    }
}

#[cfg(test)]
#[path = "../tests/unit/attention_profile_capabilities.rs"]
mod capability_tests;
