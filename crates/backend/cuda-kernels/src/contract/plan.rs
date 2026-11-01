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
