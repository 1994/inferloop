//! GGML/GGUF-specific quantization contracts.
//!
//! These types stay out of the native projection contract: the GGML block formats are a
//! separate kernel family, selected only by the GGUF package path.
use crate::contract::buffer::Buffer;

/// Geometry of one GGML GEMV launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q4KPlan {
    /// Output rows one CTA covers; must be 16.
    pub tile_rows: usize,
    /// Reduction length.
    pub columns: usize,
}

/// A `Q4_K` matrix in the tile-major device layout.
#[derive(Clone, Copy, Debug)]
pub struct Q4KWeight<'a> {
    /// Tile-major packed 4-bit codes.
    pub codes: Buffer<'a, u8>,
    /// Tile-major per-group `d * scale`.
    pub factors: Buffer<'a, f32>,
    /// Tile-major per-group `dmin * min`.
    pub biases: Buffer<'a, f32>,
}
