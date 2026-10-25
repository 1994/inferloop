//! Native CUDA execution with GPU kernels authored in Rust through cuTile.
mod attention_profile;
pub use attention_profile::AttentionProfile;
#[cfg(cuda_backend)]
pub mod attention;
#[cfg(cuda_backend)]
pub mod benchmark;
pub(crate) mod constants;
#[cfg(cuda_backend)]
pub mod device;
#[cfg(cuda_backend)]
mod kernels;
#[cfg(cuda_backend)]
pub mod loading;
#[cfg(cuda_backend)]
pub mod mlp;
pub mod nvfp4;
#[cfg(cuda_backend)]
mod quantized;
#[cfg(cuda_backend)]
pub mod resident;
pub mod strategy;
pub mod target;
#[cfg(cuda_backend)]
pub mod tuning;
#[cfg(cuda_backend)]
pub mod vision;

#[cfg(cuda_backend)]
pub mod executor;
pub mod registry;

// Fixtures that two different module trees both need. Gated exactly like the modules that use it,
// so it is neither dead code nor built where CUDA does not compile.
#[cfg(all(test, cuda_backend))]
#[path = "../tests/unit/test_support.rs"]
pub(crate) mod test_support;
