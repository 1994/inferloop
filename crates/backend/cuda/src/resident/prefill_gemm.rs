//! Prompt GEMM: quantized weights are decoded in tiles, BF16 MMA accumulates in F32.

#[cfg(test)]
#[path = "../../tests/unit/resident_prefill_gemm_bench_check.rs"]
mod bench_check;
pub use infer_cuda_kernels::tile::prefill_gemm as gemm;
