//! Native CUDA execution with GPU kernels authored in Rust through cuTile.
#![cfg(cuda_backend)]

mod attention_profile;
pub use attention_profile::AttentionProfile;
pub mod attention;
pub mod benchmark;
pub(crate) mod constants;
pub mod device;
mod kernels;
pub mod loading;
pub mod mlp;
pub mod nvfp4;
mod quantized;
pub mod resident;
pub mod strategy;
pub mod target;
pub mod tuning;
pub mod vision;

pub mod executor;
pub mod registry;

// Fixtures shared by tests in different module trees.
#[cfg(test)]
#[path = "../tests/unit/test_support.rs"]
pub(crate) mod test_support;
