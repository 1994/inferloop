//! RMS normalization and padded row maximum moved to `infer-cuda-kernels`.
pub use infer_cuda_kernels::tile::norm_maximum as kernels;
