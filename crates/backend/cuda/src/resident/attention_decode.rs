//! Split long KV reductions across independent CTAs before stable softmax merging.
pub use infer_cuda_kernels::tile::resident_attention_decode as kernels;

#[cfg(test)]
#[path = "../../tests/unit/resident_attention_decode.rs"]
mod tests;

pub(super) mod pooled;
mod workspace;
pub use workspace::VERIFY_LANES;
pub use workspace::Workspace;

#[cfg(test)]
pub use workspace::taps;
