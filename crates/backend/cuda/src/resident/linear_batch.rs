//! Row-group projection kernels moved to `infer-cuda-kernels`.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::projection::linear_batch as batched;
