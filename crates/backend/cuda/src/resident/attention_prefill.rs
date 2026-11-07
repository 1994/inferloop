//! Parallel causal queries over a contiguous prompt chunk; KV layout remains head-major.
/// Legacy full-chunk loop retained only for comparison in CUDA tests.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub const FULL_CHUNK_BOUND: &str = "0";

/// Independently captured bounds for the numerical and cost diagnostic.
#[cfg(test)]
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalBlockBound {
    FullChunk,
    QueryTile,
}
#[cfg(test)]
#[cfg(feature = "cuda")]
impl CausalBlockBound {
    pub(crate) fn generic(self) -> String {
        match self {
            Self::FullChunk => FULL_CHUNK_BOUND.into(),
            Self::QueryTile => "1".into(),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::attention::resident_attention_prefill as kernels;

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_attention_prefill.rs"]
mod tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_prefill_page_append.rs"]
mod page_tests;
