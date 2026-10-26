//! Default projection policies; kernel modules retain only layout and ABI constraints.

/// Vendor BF16 projection wins from 2048x8192 weights upwards in prompt measurements.
const MIN_DELEGATED_WEIGHT_ELEMENTS: usize = 8 * 1024 * 1024;
/// Narrowing tile chosen from prompt measurements (1024 elements: 5.2 us versus 8.1 us at 8192).
pub const DENSE_CAST_TILE: usize = 1024;
/// Maximum output width handled by the 16-column prompt tile.
const NARROW_DENSE_MAX_ROWS: usize = 64;
/// Reduction widths must stay tileable by the shared 64-column loading contract.
const NARROW_DENSE_COLUMN_TILE: usize = 64;

pub const fn delegate_dense(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(k) > 0 && n.saturating_mul(k) >= MIN_DELEGATED_WEIGHT_ELEMENTS
}

/// The fused draft projection has been qualified on this architecture.
pub fn fused_draft(architecture: &str) -> bool {
    architecture == "sm_120"
}

/// Whether a projection can use the N16 output tile under the kernel's ABI.
///
/// Role recognition is separate: only a Delta beta/alpha producer may select this
/// tile, so an unrelated same-shaped Linear remains on the normal path. Outputs above
/// 64 columns use the 64-column tile; 32/40/48/64 cardinalities are all legal inputs.
pub fn narrow_dense_layout(architecture: &str, rows: usize, columns: usize) -> bool {
    architecture == "sm_120"
        && (1..NARROW_DENSE_MAX_ROWS).contains(&rows)
        && columns.is_multiple_of(NARROW_DENSE_COLUMN_TILE)
}
