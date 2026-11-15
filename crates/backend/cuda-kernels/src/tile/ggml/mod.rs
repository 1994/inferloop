//! GGML/GGUF-specific kernels.
//!
//! Kept separate from the native projection family: these kernels only run when a GGUF package
//! supplies GGML block layouts, and they are not part of `Kernels`.
mod q4_k_gemv_kernels;
pub use q4_k_gemv_kernels::q4_k;

use crate::contract::buffer::Buffer;
use crate::contract::ggml::{Q4KPlan, Q4KWeight};
use crate::contract::recording::Recording;
use crate::tile::{backend_error, borrowed, capture_scope};
use cutile::prelude::*;
use infer_core::{Error, Result};

/// Rows a `Q4_K` tile covers.
const TILE_ROWS: usize = 16;
/// Values in one `Q4_K` super-block.
const BLOCK_VALUES: usize = 256;
/// Code bytes in one `Q4_K` group.
const GROUP_LANES: usize = 32;

/// Record a `Q4_K` GEMV.
/// # Errors
/// Rejects a plan or weight buffer that does not match the tiled layout, or a launch the
/// recording refuses.
pub fn q4_k_row(
    ordinal: usize,
    recording: &Recording<'_>,
    plan: &Q4KPlan,
    out: Buffer<'_, f32>,
    x: Buffer<'_, f32>,
    weight: Q4KWeight<'_>,
) -> Result<()> {
    let scope = capture_scope(recording)?;
    let columns = plan.columns;
    let outputs = out.len();
    if columns == 0
        || !columns.is_multiple_of(BLOCK_VALUES)
        || plan.tile_rows != TILE_ROWS
        || !outputs.is_multiple_of(TILE_ROWS)
        || x.len() != columns
    {
        return Err(Error::unsupported("Q4_K GEMV shape"));
    }
    let tiles = outputs / TILE_ROWS;
    let groups = columns / (GROUP_LANES * 2);
    let scales = columns / GROUP_LANES;
    let code_len = tiles * groups * GROUP_LANES * TILE_ROWS;
    let scale_len = tiles * scales * TILE_ROWS;
    if weight.codes.len() != code_len
        || weight.factors.len() != scale_len
        || weight.biases.len() != scale_len
    {
        return Err(Error::unsupported("Q4_K GEMV layout"));
    }
    recording.retain(out);
    recording.retain(x);
    recording.retain(weight.codes);
    recording.retain(weight.factors);
    recording.retain(weight.biases);
    let mut out = borrowed(out, ordinal, &[outputs])?;
    let x = borrowed(x, ordinal, &[columns])?;
    let codes = borrowed(weight.codes, ordinal, &[code_len])?;
    let factors = borrowed(weight.factors, ordinal, &[scale_len])?;
    let biases = borrowed(weight.biases, ordinal, &[scale_len])?;
    let tile = (&mut out).partition([TILE_ROWS]);
    let generics = vec![
        columns.to_string(),
        (columns / (BLOCK_VALUES * 4)).to_string(),
    ];
    scope
        .record(q4_k::gemv(tile, &x, &codes, &factors, &biases).generics(generics))
        .map(drop)
        .map_err(backend_error)
}
