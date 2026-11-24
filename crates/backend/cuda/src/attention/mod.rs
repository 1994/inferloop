//! Reusable CUDA attention kernels; model-specific projection and `RoPE` stay with callers.
pub(crate) mod kernels;
mod plan;
pub use plan::DenseAttentionPlan;
pub(crate) use plan::tile_rows;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod gate_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
pub(crate) mod benchmark_tests;
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests;
