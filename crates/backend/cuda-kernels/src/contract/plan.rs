//! Geometry passed to a kernel launch; owns no device handles.

/// One row-major matrix-vector projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearPlan {
    /// Output rows and weight rows.
    pub rows: usize,
    /// Reduction length and weight columns.
    pub columns: usize,
    /// Reduction tile the kernel loads per step.
    pub block: usize,
    /// Output rows covered by one tile.
    pub tile_rows: usize,
}
