//! Default projection policies; kernel modules retain only layout and ABI constraints.

/// Vendor BF16 projection wins from 2048x8192 weights upwards in prompt measurements.
const MIN_DELEGATED_WEIGHT_ELEMENTS: usize = 8 * 1024 * 1024;
/// Narrowing tile chosen from prompt measurements (1024 elements: 5.2 us versus 8.1 us at 8192).
pub const DENSE_CAST_TILE: usize = 1024;
/// Output width of the small dense pair whose N16 tile was measured.
const NARROW_DENSE_ROWS: usize = 48;
/// Reduction widths must stay tileable by the shared 64-column loading contract.
const NARROW_DENSE_COLUMN_TILE: usize = 64;

pub const fn delegate_dense(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(k) > 0 && n.saturating_mul(k) >= MIN_DELEGATED_WEIGHT_ELEMENTS
}

/// The fused draft projection has been qualified on this architecture.
pub fn fused_draft(architecture: &str) -> bool {
    architecture == "sm_120"
}

/// Whether a small dense projection can use the measured N16 output tile.
///
/// This is the kernel's actual structural contract: a 48-row projection with a
/// tileable reduction width. The prompt width is not part of the kernel geometry and
/// must not turn a legal small projection into a shape-exact special case.
pub fn narrow_dense_layout(architecture: &str, rows: usize, columns: usize) -> bool {
    architecture == "sm_120"
        && rows == NARROW_DENSE_ROWS
        && columns.is_multiple_of(NARROW_DENSE_COLUMN_TILE)
}
