//! CUDA toolkit library candidates for the currently supported Linux host.
//!
//! The ordering preserves each consumer's existing fallback. These names do not establish
//! device capability, ABI compatibility, or availability of embedded platform artifacts.
//! Other platforms require their own SDK-verified names and calling conventions.

/// cuBLAS used by projection GEMMs.
pub const CUBLAS: [&str; 2] = ["libcublas.so.13", "libcublas.so"];
/// cuBLASLt used by vendor FP8 projections.
pub const CUBLAS_LT: [&str; 2] = ["libcublasLt.so.13", "libcublasLt.so"];
/// CUDA runtime required by the existing AOT host artifact.
pub const CUDA_RUNTIME: [&str; 2] = ["libcudart.so.13", "/opt/cuda/lib64/libcudart.so.13"];
