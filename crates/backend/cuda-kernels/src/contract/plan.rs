//! Geometry passed to a kernel launch; owns no device handles.

/// Which positions a rotary launch reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// One sequence at one position.
    Single,
    /// One sequence with per-axis positions.
    SingleAxes,
    /// A prompt: each lane of `heads` rows at its own position.
    Prompt,
    /// A prompt with per-axis positions.
    PromptAxes,
    /// Pooled rows, each lane's position `fields` apart in the metadata.
    Pooled { fields: usize },
}

/// Rotary update of `rows` rows of `head_dim`, over the first `2 * half` elements of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopePlan {
    pub mode: RopeMode,
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub half: usize,
}

/// Row-independent projection: `output[row, n] = sum_k weight[n, k] * input[row, k]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolGemvPlan {
    /// Independent activation rows.
    pub rows: usize,
    /// Output elements per row and weight rows.
    pub outputs: usize,
    /// Reduction length and weight columns.
    pub columns: usize,
    /// Output elements covered by one tile; only cuTile uses it.
    pub tile_rows: usize,
    /// Reduction tile the kernel loads per step; only cuTile uses it.
    pub tile_columns: usize,
}

/// How a normalization launch partitions its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormLayout {
    /// One row per block, over the next power of two of the dimension.
    Block,
    /// `tile` rows per block; the kernel masks a final partial tile.
    Rows { tile: usize },
}

/// RMS normalization of `rows` rows of `dimension`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormPlan {
    pub layout: NormLayout,
    pub rows: usize,
    pub dimension: usize,
    /// Multiply by the gate activation as well; [`NormLayout`] is unchanged by it.
    pub gated: bool,
    pub epsilon: f32,
    pub offset: f32,
    /// Measured compiler occupancy; `None` leaves the compiler's choice.
    pub occupancy: Option<i32>,
}

/// Which rows a draft fusion normalizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionShape {
    /// One embedding row and one hidden row, both flat.
    Single,
    /// `lanes` embedding rows and as many hidden rows, each `dimension` wide.
    Lanes { lanes: usize },
}

/// Normalization of a draft's embedding and hidden rows into `[2 * lanes, dimension]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionNormPlan {
    pub shape: FusionShape,
    pub dimension: usize,
    pub epsilon: f32,
    pub offset: f32,
}

/// Normalization that also produces what the next projection quantizes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantNormPlan {
    pub rows: usize,
    pub dimension: usize,
    pub epsilon: f32,
    pub offset: f32,
}
