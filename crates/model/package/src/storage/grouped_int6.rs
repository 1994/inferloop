//! Kernel-facing `Q6_K` layout and its host oracle.
//!
//! A 210-byte GGUF block is repacked into one byte per 6-bit code plus one F32 factor per
//! 16-value group. The tile-major order keeps a `[256, 16]` super-block contiguous.
use infer_core::{Error, Result};
use infer_format::half_to_f32;

/// Values in one `Q6_K` super-block.
const QK_K: usize = 256;
/// Stored bytes of one GGUF `Q6_K` block.
const BLOCK_BYTES: usize = 210;
/// Rows one tiled super-block covers.
pub const Q6K_TILE_ROWS: usize = 16;
/// 16-value groups in one super-block.
const GROUPS: usize = 16;
/// Values in one group.
const GROUP_VALUES: usize = 16;
/// Half-super-block values.
const HALF_VALUES: usize = 128;
/// Bytes of low nibbles per half.
const HALF_QL: usize = 64;
/// Offset of high-bit plane.
const QH_OFFSET: usize = HALF_VALUES;
/// Offset of the signed scales.
const SCALES_OFFSET: usize = 192;
/// Bytes of signed scales.
const SCALE_BYTES: usize = 16;
/// Offset of the F16 `d`.
const D_OFFSET: usize = 208;
/// Lanes in one high-bit half.
const LANES: usize = 32;
/// Low nibble mask.
const NIBBLE_MASK: u8 = 0xF;
/// Bits per nibble.
const NIBBLE_BITS: u32 = 4;
/// Two high bits per lane.
const QH_MASK: u8 = 3;
/// Mask of one 6-bit code.
const CODE_MASK: i32 = 0x3F;
/// Shift selecting the fourth code's high bits.
const HIGH_SHIFT: u32 = 6;
/// Codes produced by one lane.
const CODES_PER_LANE: usize = 4;
/// Output offsets for the four codes of one lane.
const OUTPUT_OFFSETS: [usize; CODES_PER_LANE] = [0, LANES, 64, 96];
/// Scale offsets for the four codes of one lane.
const SCALE_OFFSETS: [usize; CODES_PER_LANE] = [0, 2, 4, 6];

/// One `Q6_K` matrix in the tiled device layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Q6KTiled {
    rows: usize,
    columns: usize,
    codes: Vec<u8>,
    factors: Vec<f32>,
}

impl Q6KTiled {
    /// Repack a row-major `[rows, columns]` `Q6_K` matrix, zero-padding a partial last tile.
    /// # Errors
    /// Rejects a shape that is not a whole number of super-blocks or a payload whose length
    /// disagrees with the shape.
    pub fn from_blocks(rows: usize, columns: usize, blocks: &[u8]) -> Result<Self> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(QK_K) {
            return Err(Error::invalid("Q6_K matrix shape"));
        }
        let blocks_per_row = columns / QK_K;
        let block_count = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| Error::invalid("Q6_K matrix size overflow"))?;
        let expected = block_count
            .checked_mul(BLOCK_BYTES)
            .ok_or_else(|| Error::invalid("Q6_K matrix size overflow"))?;
        if blocks.len() != expected {
            return Err(Error::invalid("Q6_K payload length"));
        }
        let tiles = rows.div_ceil(Q6K_TILE_ROWS);
        let code_len = tiles
            .checked_mul(blocks_per_row)
            .and_then(|value| value.checked_mul(QK_K * Q6K_TILE_ROWS))
            .ok_or_else(|| Error::invalid("Q6_K code size overflow"))?;
        let factor_len = tiles
            .checked_mul(blocks_per_row)
            .and_then(|value| value.checked_mul(GROUPS * Q6K_TILE_ROWS))
            .ok_or_else(|| Error::invalid("Q6_K factor size overflow"))?;
        let mut tiled = Self {
            rows,
            columns,
            codes: vec![0; code_len],
            factors: vec![0.0; factor_len],
        };
        for tile in 0..tiles {
            for block_index in 0..blocks_per_row {
                // Source rows are not contiguous across a tile, so resolve each row explicitly.
                for row in 0..Q6K_TILE_ROWS {
                    let source_row = tile * Q6K_TILE_ROWS + row;
                    if source_row >= rows {
                        continue;
                    }
                    let offset = (source_row * blocks_per_row + block_index) * BLOCK_BYTES;
                    let data = &blocks[offset..offset + BLOCK_BYTES];
                    let d = half_to_f32(u16::from_le_bytes([data[D_OFFSET], data[D_OFFSET + 1]]));
                    for half in 0..2 {
                        let ql = &data[half * HALF_QL..][..HALF_QL];
                        let qh = &data[QH_OFFSET + half * LANES..][..LANES];
                        let scales =
                            &data[SCALES_OFFSET + half * SCALE_BYTES / 2..][..SCALE_BYTES / 2];
                        let base = half * HALF_VALUES;
                        for lane in 0..LANES {
                            let q: [i32; CODES_PER_LANE] = [
                                i32::from(ql[lane] & NIBBLE_MASK)
                                    | (i32::from(qh[lane] & QH_MASK) << NIBBLE_BITS),
                                i32::from(ql[lane + LANES] & NIBBLE_MASK)
                                    | (i32::from((qh[lane] >> 2) & QH_MASK) << NIBBLE_BITS),
                                i32::from(ql[lane] >> NIBBLE_BITS)
                                    | (i32::from((qh[lane] >> NIBBLE_BITS) & QH_MASK)
                                        << NIBBLE_BITS),
                                i32::from(ql[lane + LANES] >> NIBBLE_BITS)
                                    | (i32::from((qh[lane] >> HIGH_SHIFT) & QH_MASK)
                                        << NIBBLE_BITS),
                            ];
                            let scale = lane / GROUP_VALUES;
                            for index in 0..CODES_PER_LANE {
                                let position = base + lane + OUTPUT_OFFSETS[index];
                                let code = u8::try_from(q[index] & CODE_MASK)
                                    .map_err(|_| Error::invalid("Q6_K code exceeds u8"))?;
                                tiled.codes[(tile * blocks_per_row + block_index)
                                    * QK_K
                                    * Q6K_TILE_ROWS
                                    + position * Q6K_TILE_ROWS
                                    + row] = code;
                                let scale_index = scale + SCALE_OFFSETS[index];
                                let factor = d * f32::from(
                                    i8::try_from(scales[scale_index])
                                        .map_err(|_| Error::invalid("Q6_K scale"))?,
                                );
                                tiled.factors[(tile * blocks_per_row + block_index)
                                    * GROUPS
                                    * Q6K_TILE_ROWS
                                    + (position / GROUP_VALUES) * Q6K_TILE_ROWS
                                    + row] = factor;
                            }
                        }
                    }
                }
            }
        }
        Ok(tiled)
    }

    /// Matrix rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Matrix columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Tile-major 6-bit codes.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Tile-major per-group factors.
    #[must_use]
    pub fn factors(&self) -> &[f32] {
        &self.factors
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gguf_q6k_layout.rs"]
mod tests;
