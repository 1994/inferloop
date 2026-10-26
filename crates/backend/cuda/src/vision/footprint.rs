//! Checked device-byte footprint of one media prefill.
//!
//! Encoding is sequential per media item, and the encoded rows are retained together only after
//! every item has been encoded. The quote is therefore the maximum of the largest single-item
//! encode peak and the total retained encoded-row tensor. Resident vision weights are excluded;
//! they are load-time owners, not request allocations.
use super::program::{attention_tiles, mlp_width, patch_width};
use super::weights::half_tile;
use infer_core::{Error, Result};
use infer_ir::MediaPlan;
use infer_models::vision::{merge_unit, merged_width, row_tile};
use infer_spi::ModalityEncoder;

const F32_BYTES: u64 = 4;
const TOKEN_TILE: usize = 32;
/// Qualified wide attention head width after packing compact 96-wide heads.
const WIDE_HEAD_WIDTH: usize = 128;
/// Q, K, V and attended temporary wide buffers.
const COMPACT_PACK_BUFFERS: usize = 4;
/// Patch embeddings, gathered positions and the added output each hold one hidden matrix.
const PATCH_EMBED_HIDDEN_COPIES: usize = 3;
/// Q, K, V, rotated Q, rotated K and the attended output each hold one projection matrix.
const ATTENTION_PROJECTION_COPIES: usize = 6;
/// Residual add uploads both operands and the output.
const RESIDUAL_ACTIVATIONS: usize = 3;

fn bytes(elements: usize) -> Result<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|n| n.checked_mul(F32_BYTES))
        .ok_or_else(|| Error::invalid("vision media footprint overflow"))
}

fn padded(rows: usize, tile: usize) -> Result<usize> {
    rows.checked_next_multiple_of(tile)
        .ok_or_else(|| Error::invalid("vision media row padding overflow"))
}

fn one_item_peak(encoder: &ModalityEncoder, grid: (usize, usize, usize)) -> Result<u64> {
    let (temporal, height, width) = grid;
    let patches = temporal
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| Error::invalid("vision media grid overflow"))?;
    if patches == 0 {
        return Err(Error::invalid("vision media grid is empty"));
    }
    let hidden = encoder.hidden_size;
    let patch = patch_width(encoder)?;
    let half = half_tile(encoder)?;
    let head_width = half
        .checked_mul(2)
        .ok_or_else(|| Error::invalid("vision half-head overflow"))?;
    let projections = encoder
        .heads
        .checked_mul(head_width)
        .ok_or_else(|| Error::invalid("vision projection overflow"))?;
    let mlp = mlp_width(encoder)?;
    let token_padded = padded(patches, TOKEN_TILE)?;
    let attention_rows = padded(patches, attention_tiles(patches)[0])?;
    let tile = row_tile(encoder)?;

    let patch_embed = bytes(
        token_padded
            .checked_mul(
                patch
                    .checked_add(
                        hidden
                            .checked_mul(PATCH_EMBED_HIDDEN_COPIES)
                            .ok_or_else(|| Error::invalid("vision hidden overflow"))?,
                    )
                    .ok_or_else(|| Error::invalid("vision patch footprint overflow"))?,
            )
            .ok_or_else(|| Error::invalid("vision patch footprint overflow"))?,
    )?;
    let attention = bytes(
        attention_rows
            .checked_mul(
                hidden
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(head_width.checked_mul(2)?))
                    .and_then(|n| {
                        n.checked_add(projections.checked_mul(ATTENTION_PROJECTION_COPIES)?)
                    })
                    .ok_or_else(|| Error::invalid("vision attention footprint overflow"))?,
            )
            .ok_or_else(|| Error::invalid("vision attention footprint overflow"))?,
    )?;
    let layernorm = bytes(
        patches
            .checked_mul(tile)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::invalid("vision norm footprint overflow"))?,
    )?;
    let fc1 = bytes(
        token_padded
            .checked_mul(
                hidden
                    .checked_add(mlp)
                    .ok_or_else(|| Error::invalid("vision MLP footprint overflow"))?,
            )
            .ok_or_else(|| Error::invalid("vision MLP footprint overflow"))?,
    )?;
    let fc2 = fc1;
    let gelu = bytes(
        token_padded
            .checked_mul(mlp)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::invalid("vision GELU footprint overflow"))?,
    )?;
    let residual = bytes(
        token_padded
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(RESIDUAL_ACTIVATIONS))
            .ok_or_else(|| Error::invalid("vision residual footprint overflow"))?,
    )?;
    let compact_pack = compact_pack_bytes(encoder, attention_rows)?;
    let attention_pack = attention
        .checked_add(compact_pack)
        .ok_or_else(|| Error::invalid("vision attention pack footprint overflow"))?;
    let block = [attention_pack, layernorm, fc1, fc2, gelu, residual]
        .into_iter()
        .max()
        .ok_or_else(|| Error::invalid("vision block footprint empty"))?;

    let merger_peak = merger_peak(encoder, patches, hidden)?;
    Ok(patch_embed.max(block).max(merger_peak))
}

fn compact_pack_bytes(encoder: &ModalityEncoder, attention_rows: usize) -> Result<u64> {
    // Compact attention packs Q/K/V into 128-wide temporary buffers and keeps the attended
    // output wide before packing back. Price those four buffers explicitly; this is
    // deliberately conservative for the 128 fallback.
    let wide_projection = encoder
        .heads
        .checked_mul(WIDE_HEAD_WIDTH)
        .ok_or_else(|| Error::invalid("vision wide projection overflow"))?;
    bytes(
        attention_rows
            .checked_mul(wide_projection)
            .and_then(|n| n.checked_mul(COMPACT_PACK_BUFFERS))
            .ok_or_else(|| Error::invalid("vision compact pack overflow"))?,
    )
}

fn merger_peak(encoder: &ModalityEncoder, patches: usize, hidden: usize) -> Result<u64> {
    let merged = merged_width(encoder)?;
    let unit = merge_unit(encoder)?;
    if !patches.is_multiple_of(unit) {
        return Err(Error::invalid("vision merger geometry"));
    }
    let rows = patches / unit;
    let tile = row_tile(encoder)?;
    let merge_rows = padded(rows, TOKEN_TILE)?;
    let merger_norm = bytes(
        patches
            .checked_mul(tile)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::invalid("vision merger norm overflow"))?,
    )?;
    let merger_fc1 = bytes(
        merge_rows
            .checked_mul(
                hidden
                    .checked_add(merged)
                    .and_then(|n| n.checked_add(merged))
                    .ok_or_else(|| Error::invalid("vision merger footprint overflow"))?,
            )
            .ok_or_else(|| Error::invalid("vision merger footprint overflow"))?,
    )?;
    let merger_gelu = bytes(
        merge_rows
            .checked_mul(merged)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::invalid("vision merger GELU overflow"))?,
    )?;
    let merger_fc2 = bytes(
        merge_rows
            .checked_mul(
                merged
                    .checked_add(encoder.out_hidden_size)
                    .ok_or_else(|| Error::invalid("vision merger output overflow"))?,
            )
            .ok_or_else(|| Error::invalid("vision merger output overflow"))?,
    )?;
    [merger_norm, merger_fc1, merger_gelu, merger_fc2]
        .into_iter()
        .max()
        .ok_or_else(|| Error::invalid("vision merger footprint empty"))
}

/// Peak request-owned bytes and retained encoded rows for a prepared media plan.
///
/// # Errors
/// Rejects empty/overflowing grids or grid/encoder geometry mismatches.
pub fn footprint(encoder: &ModalityEncoder, media: &MediaPlan) -> Result<u64> {
    if media.items.is_empty() {
        return Err(Error::invalid("media plan has no items"));
    }
    let mut peak = 0_u64;
    let mut retained = 0_u64;
    for item in &media.items {
        peak = peak.max(one_item_peak(encoder, item.grid)?);
        let patches = item
            .grid
            .0
            .checked_mul(item.grid.1)
            .and_then(|n| n.checked_mul(item.grid.2))
            .ok_or_else(|| Error::invalid("vision media grid overflow"))?;
        let rows = patches
            .checked_div(merge_unit(encoder)?)
            .ok_or_else(|| Error::invalid("vision merge unit"))?;
        retained = retained
            .checked_add(bytes(
                rows.checked_mul(encoder.out_hidden_size)
                    .ok_or_else(|| Error::invalid("vision retained row overflow"))?,
            )?)
            .ok_or_else(|| Error::invalid("vision retained footprint overflow"))?;
    }
    peak.max(retained)
        .checked_add(0)
        .ok_or_else(|| Error::invalid("vision media footprint"))
}
