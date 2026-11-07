//! Identities of candidate kernels, which tuning results are keyed by.
//!
//! Every identity here names a cuTile JIT candidate. AOT kernels are fixed artifacts with no JIT
//! candidates to search, so each identity reports it is not provided.
use super::runtime::{DType, Device};
use crate::contract::{Fp4ScalePlan, GEMM_RANK, GreedyPlan, RECURRENT_RANK};
use infer_core::{Error, Result};

pub use crate::contract::{AttentionShape, DenseSearch, Fp4Search, SwigluSearch};

fn unsupported<T>(identity: &str) -> Result<T> {
    Err(Error::unsupported(format!(
        "{identity} identifies JIT candidates, which AOT kernels do not have"
    )))
}

/// The prompt recurrence and its transpose at each block split.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn recurrent_wy(
    _device: &Device,
    _shape: [usize; RECURRENT_RANK],
    _splits: &[usize],
) -> Result<Vec<String>> {
    unsupported("recurrent_wy")
}

/// The FP8 quantizer, every candidate tile of its product, and the vendor path's scale.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn fp8_candidates(
    _device: &Device,
    _shape: [usize; GEMM_RANK],
    _row_tiles: &[usize],
    _column_tiles: &[usize],
    _aux_tile: usize,
) -> Result<Vec<String>> {
    unsupported("fp8_candidates")
}

/// The FP4 quantizer and every candidate tile and main loop of its product.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn fp4_candidates(
    _device: &Device,
    _shape: [usize; GEMM_RANK],
    _search: Fp4Search<'_>,
) -> Result<Vec<String>> {
    unsupported("fp4_candidates")
}

/// Every prompt-tile candidate of a dense projection.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn dense_prompt_candidates(
    _device: &Device,
    _shape: [usize; GEMM_RANK],
    _tiles: &[[usize; 2]],
) -> Result<Vec<String>> {
    unsupported("dense_prompt_candidates")
}

/// The pooled dense candidates of a projection.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn dense_candidates(
    _device: &Device,
    _shape: [usize; GEMM_RANK],
    _search: DenseSearch<'_>,
) -> Result<Vec<String>> {
    unsupported("dense_candidates")
}

/// The row-grouped quantizer and every candidate tile of the fused `SwiGLU` product.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn swiglu_candidates(
    _device: &Device,
    _shape: [usize; GEMM_RANK],
    _search: SwigluSearch<'_>,
) -> Result<Vec<String>> {
    unsupported("swiglu_candidates")
}

/// The gated norm at each candidate occupancy; `None` is the compiler's default.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn gated_norm_candidates(
    _device: &Device,
    _shape: [usize; 2],
    _epsilon: f32,
    _occupancies: &[Option<i32>],
) -> Result<Vec<String>> {
    unsupported("gated_norm_candidates")
}

/// The native prepared recurrence: normalize, gates, precomputed delta and transpose.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn exact_recurrence(
    _device: &Device,
    _shape: [usize; RECURRENT_RANK],
    _value_columns: usize,
    _metadata_fields: usize,
) -> Result<Vec<String>> {
    unsupported("exact_recurrence")
}

/// The prompt append and compute of KV element `E`; a query tile selects the tiled compute.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn attention_candidates<E: DType>(
    _device: &Device,
    _shape: &AttentionShape,
    _query_tile: Option<usize>,
) -> Result<Vec<String>> {
    unsupported("attention_candidates")
}

/// Both greedy reductions at their exact geometry.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn greedy_reductions(_device: &Device, _plan: &GreedyPlan) -> Result<String> {
    unsupported("greedy_reductions")
}

/// The FP4 scale swizzle a vendor FP4 GEMM consumes, into an `out_len`-byte destination.
/// # Errors
/// Always: AOT kernels have no JIT candidates.
pub fn fp4_scale_swizzle(
    _device: &Device,
    _plan: &Fp4ScalePlan,
    _out_len: usize,
) -> Result<String> {
    unsupported("fp4_scale_swizzle")
}
