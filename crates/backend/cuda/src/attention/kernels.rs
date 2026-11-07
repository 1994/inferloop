//! Compensated SDPA kernels moved to `infer-cuda-kernels`.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::attention::sdpa;
