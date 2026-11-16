//! Grouped integer weight contracts.
//!
//! `Q4_K`/`Q6_K` are precision encodings, not a file format. They stay in the projection family
//! and are selected by [`crate::contract::kernels::WeightRef::GroupedInt`].
use crate::contract::buffer::Buffer;

/// Geometry of one grouped-integer GEMV launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedIntPlan {
    /// Output rows one CTA covers; currently 16.
    pub tile_rows: usize,
    /// Reduction length.
    pub columns: usize,
}

/// A grouped-integer matrix in the tiled device layout.
#[derive(Clone, Copy, Debug)]
pub struct GroupedIntWeight<'a> {
    /// Tile-major packed 4-bit codes.
    pub codes: Buffer<'a, u8>,
    /// Tile-major per-group `d * scale`.
    pub factors: Buffer<'a, f32>,
    /// Tile-major per-group `dmin * min`.
    pub biases: Buffer<'a, f32>,
}
