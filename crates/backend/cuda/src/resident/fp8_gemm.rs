//! Checkpoint-declared dynamic per-token E4M3 activation quantization.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::projection::fp8_gemm as kernels;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests;
