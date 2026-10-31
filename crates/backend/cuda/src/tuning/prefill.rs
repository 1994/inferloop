//! Default projection policies; kernel modules retain only layout and ABI constraints.
use crate::constants::{MID_PREFILL_LANES, PREFILL_GEMM_TILE_N};
use infer_cuda_kernels::contract::target::Arch;

/// Vendor BF16 projection wins from 2048x8192 weights upwards in prompt measurements.
const MIN_DELEGATED_WEIGHT_ELEMENTS: usize = 8 * 1024 * 1024;
/// Narrowing tile chosen from prompt measurements (1024 elements: 5.2 us versus 8.1 us at 8192).
pub const DENSE_CAST_TILE: usize = 1024;

pub const fn delegate_dense(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(k) > 0 && n.saturating_mul(k) >= MIN_DELEGATED_WEIGHT_ELEMENTS
}

/// The fused draft projection has been qualified on this architecture.
pub fn fused_draft(architecture: &str) -> bool {
    Arch::from_sm_name(architecture).is_some_and(Arch::qualifies_fused_draft)
}

pub fn narrow_dense_layout(architecture: &str, shape: (usize, usize, usize)) -> bool {
    let (lanes, rows, columns) = shape;
    Arch::from_sm_name(architecture).is_some_and(Arch::qualifies_narrow_dense)
        && lanes == MID_PREFILL_LANES
        && (1..=PREFILL_GEMM_TILE_N).contains(&rows)
        && columns > 0
}
