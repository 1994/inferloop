//! Shapes and tile sets that tuning identities and eager device work cover, shared by every
//! kernel source so the host names one type whichever source is built.

/// Axes of a projection shape: rows, outputs and columns.
pub const GEMM_RANK: usize = 3;
/// Key axes a recurrent identity covers: key heads, value heads, channels and lanes.
pub const RECURRENT_RANK: usize = 4;

/// Tiles an FP4 product identity covers.
#[derive(Clone, Copy, Debug)]
pub struct Fp4Search<'a> {
    pub group: usize,
    pub codes_tile: [usize; 2],
    pub scales_tile: [usize; 2],
    pub tile_m: &'a [usize],
    pub tile_n: &'a [usize],
}

/// Tiles a pooled dense identity covers.
#[derive(Clone, Copy, Debug)]
pub struct DenseSearch<'a> {
    pub tile_rows: usize,
    pub tile_columns: usize,
    pub shared_rows: usize,
    pub native_tile: [usize; 2],
    pub wide_rows: usize,
    /// Wider native tiles; the qualified family lists none, so its identity stays frozen.
    pub wide: &'a [usize],
}

/// Tiles a fused `SwiGLU` identity covers.
#[derive(Clone, Copy, Debug)]
pub struct SwigluSearch<'a> {
    pub group: usize,
    pub codes_tile: [usize; 2],
    pub scales_tile: [usize; 2],
    pub tiles: &'a [[usize; 2]],
}

/// Geometry and arithmetic an attention identity covers.
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub lanes: usize,
    pub capacity: usize,
    pub block: usize,
    pub window: usize,
    pub scales: [f32; 2],
    pub metadata_fields: usize,
}

/// Tile geometry of one eager projection.
#[derive(Clone, Copy, Debug)]
pub struct MatvecGeometry {
    pub rows: usize,
    pub columns: usize,
    pub tile_rows: usize,
    pub tile_columns: usize,
}

/// Rows and tile widths of the probe.
#[derive(Clone, Copy, Debug)]
pub struct ProbeGeometry {
    pub rows: usize,
    pub columns: usize,
    pub code_tile: usize,
    pub scale_tile: usize,
}
