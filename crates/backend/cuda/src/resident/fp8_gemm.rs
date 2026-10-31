//! Checkpoint-declared dynamic per-token E4M3 activation quantization.

pub use infer_cuda_kernels::tile::projection::fp8_gemm as kernels;

#[cfg(test)]
#[path = "../../tests/unit/resident_fp8_gemm.rs"]
mod tests;
