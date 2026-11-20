//! Token-chunk recurrent execution: keep the state in registers across prompt rows.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::recurrent::recurrent_prefill as kernels;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests;

/// Key/value dimension whose scalar F32 reduction trees are preserved.
pub(super) const ORDERED_DIM: usize = 128;
/// Independent value columns per CTA, selected by paired CUDA event measurements.
pub(super) const ORDERED_BLOCK: usize = 16;
/// Prompt capture widths covered by exact fixture and checkpoint comparisons.
pub(super) const ORDERED_WIDTHS: [usize; 5] = [64, 256, 512, 1024, 2048];

mod budget;
mod chunk;
mod prepared;
mod workspace;
pub use budget::required_bytes;
#[cfg(test)]
use std::cell::Cell;
pub use workspace::Workspace;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod integration;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod benchmark;

#[cfg(test)]
mod model_check;

#[cfg(test)]
thread_local! {
    pub(super) static ORDERED_CAPTURE_COUNT: Cell<usize> = const { Cell::new(0) };
    pub(super) static LEGACY_CAPTURE: Cell<bool> = const { Cell::new(false) };
    // Keep convolution identical when comparing only the Delta capture policy.
    pub(super) static LEGACY_DELTA_CAPTURE: Cell<bool> = const { Cell::new(false) };
}
