//! Geometry passed to a kernel launch; owns no device handles.

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
