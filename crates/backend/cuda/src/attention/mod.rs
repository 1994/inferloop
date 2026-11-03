//! Reusable CUDA attention kernels; model-specific projection and `RoPE` stay with callers.
pub(crate) mod kernels;
mod plan;
pub use plan::DenseAttentionPlan;
pub(crate) use plan::tile_rows;

#[cfg(test)]
#[path = "../../tests/unit/attention_gate_check.rs"]
mod gate;

#[cfg(test)]
#[path = "benchmark_check.rs"]
pub(crate) mod benchmark;
#[cfg(test)]
#[path = "../../tests/unit/attention.rs"]
mod tests;
