//! Kernel-facing `Q4_K` layout and its host GEMV oracle.
//!
//! GGUF's 144-byte super-block is a load format. The device layout keeps the 4-bit codes but
//! expands the 6-bit scale/min table into one byte per 32-value group and stores `d`/`dmin` as
//! F32, so a kernel can index them without unpacking bits. The host oracle computes the same
//! grouped accumulation the device kernel must reproduce.
use infer_core::{Error, Result};
use infer_format::half_to_f32;

/// Values in one `Q4_K` super-block.
const QK_K: usize = 256;
/// Stored bytes of one GGUF `Q4_K` super-block.
const BLOCK_BYTES: usize = 144;
/// Bytes of `d` and `dmin`.
const HEADER_BYTES: usize = 4;
/// Offset of the packed 6-bit scale/min table.
const SCALE_OFFSET: usize = HEADER_BYTES;
/// Bytes of the packed 6-bit scale/min table.
const SCALE_BYTES: usize = 12;
/// Offset of the packed 4-bit codes.
const CODE_OFFSET: usize = SCALE_OFFSET + SCALE_BYTES;
/// Bytes of the packed 4-bit codes.
const CODE_BYTES: usize = 128;
/// 64-value groups in one super-block.
const GROUPS: usize = 4;
/// Values in one 32-lane half-group.
const LANES: usize = 32;
/// Values in one group.
const GROUP_VALUES: usize = 64;
/// Scale/min pairs in one super-block.
const SCALES: usize = 8;
/// A nibble mask.
const NIBBLE_MASK: u8 = 0xf;
/// Six bits of a packed scale/min entry.
const K_SCALE_MASK: u8 = 63;
/// Shift selecting the two high bits packed into a scale byte.
const K_SCALE_HIGH_SHIFT: u32 = 6;
/// Bits per nibble.
const NIBBLE_BITS: u32 = 4;

/// One `Q4_K` matrix in the layout a device GEMV receives.
#[derive(Debug, Clone, PartialEq)]
pub struct Q4KDevice {
    rows: usize,
    columns: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    mins: Vec<u8>,
    d: Vec<f32>,
    dmin: Vec<f32>,
}

impl Q4KDevice {
    /// Repack a row-major `[rows, columns]` `Q4_K` matrix.
    /// # Errors
    /// Rejects a shape that is not a whole number of super-blocks, or a payload whose length
    /// disagrees with the shape.
    pub fn from_blocks(rows: usize, columns: usize, blocks: &[u8]) -> Result<Self> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(QK_K) {
            return Err(Error::invalid("Q4_K matrix shape"));
        }
        let blocks_per_row = columns / QK_K;
        let block_count = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| Error::invalid("Q4_K matrix size overflow"))?;
        let expected = block_count
            .checked_mul(BLOCK_BYTES)
            .ok_or_else(|| Error::invalid("Q4_K matrix size overflow"))?;
        if blocks.len() != expected {
            return Err(Error::invalid("Q4_K payload length"));
        }
        let code_len = block_count
            .checked_mul(CODE_BYTES)
            .ok_or_else(|| Error::invalid("Q4_K code size overflow"))?;
        let scale_len = block_count
            .checked_mul(SCALES)
            .ok_or_else(|| Error::invalid("Q4_K scale size overflow"))?;
        let mut device = Self {
            rows,
            columns,
            codes: vec![0u8; code_len],
            scales: vec![0u8; scale_len],
            mins: vec![0u8; scale_len],
            d: vec![0.0; block_count],
            dmin: vec![0.0; block_count],
        };
        for row in 0..rows {
            for block_index in 0..blocks_per_row {
                let offset = (row * blocks_per_row + block_index) * BLOCK_BYTES;
                let block = &blocks[offset..offset + BLOCK_BYTES];
                let index = block_index * rows + row;
                device.d[index] = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
                device.dmin[index] = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
                for lane in 0..CODE_BYTES {
                    let target = (block_index * CODE_BYTES + lane) * rows + row;
                    device.codes[target] = block[CODE_OFFSET + lane];
                }
                let packed = &block[SCALE_OFFSET..SCALE_OFFSET + SCALE_BYTES];
                for scale in 0..SCALES {
                    let (value, min) = scale_min(scale, packed);
                    let target = (block_index * SCALES + scale) * rows + row;
                    device.scales[target] = value;
                    device.mins[target] = min;
                }
            }
        }
        Ok(device)
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

    /// Expanded per-group scales, `[columns / 32, rows]` (K-major).
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    /// Expanded per-group minimum codes, `[columns / 32, rows]` (K-major).
    #[must_use]
    pub fn mins(&self) -> &[u8] {
        &self.mins
    }

    /// Per-super-block `d`, `[columns / 256, rows]` (K-major).
    #[must_use]
    pub fn d(&self) -> &[f32] {
        &self.d
    }

    /// Per-super-block `dmin`, `[columns / 256, rows]` (K-major).
    #[must_use]
    pub fn dmin(&self) -> &[f32] {
        &self.dmin
    }

    /// Packed 4-bit codes, `[columns / 2, rows]` (K-major), in the GGUF group order.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// The host oracle for one GEMV row: `out = sum_k x[k] * dequant(code[k])`.
    ///
    /// The accumulation is grouped exactly like the device kernel: per 32-lane half-group it
    /// accumulates `code * x` and `x`, then applies `d * scale` and `dmin * min`. Both sums use
    /// F32 FMA.
    /// # Errors
    /// Rejects a row outside the matrix or an activation of a different width.
    pub fn row_dot(&self, row: usize, x: &[f32]) -> Result<f32> {
        if row >= self.rows || x.len() != self.columns {
            return Err(Error::invalid("Q4_K GEMV shape"));
        }
        let blocks = self.columns / QK_K;
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let d = self.d[block * self.rows + row];
            let dmin = self.dmin[block * self.rows + row];
            for group in 0..GROUPS {
                for half in 0..2 {
                    let scale_index = block * SCALES + group * 2 + half;
                    let scale = f32::from(self.scales[scale_index * self.rows + row]);
                    let min = f32::from(self.mins[scale_index * self.rows + row]);
                    let mut code_dot = 0.0f32;
                    let mut x_sum = 0.0f32;
                    for lane in 0..LANES {
                        let column = block * QK_K + group * GROUP_VALUES + half * LANES + lane;
                        let byte = self.codes
                            [(block * CODE_BYTES + group * LANES + lane) * self.rows + row];
                        let code = if half == 0 {
                            byte & NIBBLE_MASK
                        } else {
                            byte >> NIBBLE_BITS
                        };
                        code_dot = f32::from(code).mul_add(x[column], code_dot);
                        x_sum += x[column];
                    }
                    sum = (d * scale).mul_add(code_dot, sum);
                    sum = (-dmin * min).mul_add(x_sum, sum);
                }
            }
        }
        Ok(sum)
    }
}

/// Rows a tiled `Q4_K` block covers.
pub const Q4K_TILE_ROWS: usize = 16;
/// `Q4_K` matrix packed for a 16-row GEMV tile.
///
/// Every 64-value group of one output tile is one contiguous `[32, 16]` run: 32 bytes of nibbles
/// across 16 rows. Scales, minima and super-block factors use the same tile-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Q4KTiled {
    rows: usize,
    columns: usize,
    tiles: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    mins: Vec<u8>,
    d: Vec<f32>,
    dmin: Vec<f32>,
    factors: Vec<f32>,
    biases: Vec<f32>,
}

impl Q4KTiled {
    /// Repack a row-major matrix for the 16-row device kernel, zero-padding a partial last tile.
    /// # Errors
    /// Rejects a matrix that cannot be tiled or whose checked lengths overflow.
    pub fn from_device(device: &Q4KDevice) -> Result<Self> {
        let rows = device.rows;
        let columns = device.columns;
        let tiles = rows.div_ceil(Q4K_TILE_ROWS);
        let groups = columns / (LANES * 2);
        let scale_count = columns / LANES;
        let blocks = columns / QK_K;
        let code_tile = LANES * Q4K_TILE_ROWS;
        let codes_len = tiles
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(code_tile))
            .ok_or_else(|| Error::invalid("Q4_K tiled code size overflow"))?;
        let scale_len = tiles
            .checked_mul(scale_count)
            .and_then(|value| value.checked_mul(Q4K_TILE_ROWS))
            .ok_or_else(|| Error::invalid("Q4_K tiled scale size overflow"))?;
        let block_len = tiles
            .checked_mul(blocks)
            .and_then(|value| value.checked_mul(Q4K_TILE_ROWS))
            .ok_or_else(|| Error::invalid("Q4_K tiled block size overflow"))?;
        let mut packed = Self {
            rows,
            columns,
            tiles,
            codes: vec![0u8; codes_len],
            scales: vec![0u8; scale_len],
            mins: vec![0u8; scale_len],
            d: vec![0.0; block_len],
            dmin: vec![0.0; block_len],
            factors: vec![0.0; scale_len],
            biases: vec![0.0; scale_len],
        };
        for tile in 0..tiles {
            for group in 0..groups {
                for lane in 0..LANES {
                    for row in 0..Q4K_TILE_ROWS {
                        let source_row = tile * Q4K_TILE_ROWS + row;
                        if source_row >= rows {
                            continue;
                        }
                        let source = device.codes[(group * LANES + lane) * rows + source_row];
                        let target =
                            (tile * groups + group) * code_tile + lane * Q4K_TILE_ROWS + row;
                        packed.codes[target] = source;
                    }
                }
            }
            for scale in 0..scale_count {
                for row in 0..Q4K_TILE_ROWS {
                    let source_row = tile * Q4K_TILE_ROWS + row;
                    if source_row >= rows {
                        continue;
                    }
                    packed.scales[(tile * scale_count + scale) * Q4K_TILE_ROWS + row] =
                        device.scales[scale * rows + source_row];
                    packed.mins[(tile * scale_count + scale) * Q4K_TILE_ROWS + row] =
                        device.mins[scale * rows + source_row];
                }
            }
            for block in 0..blocks {
                for row in 0..Q4K_TILE_ROWS {
                    let source_row = tile * Q4K_TILE_ROWS + row;
                    if source_row >= rows {
                        continue;
                    }
                    packed.d[(tile * blocks + block) * Q4K_TILE_ROWS + row] =
                        device.d[block * rows + source_row];
                    packed.dmin[(tile * blocks + block) * Q4K_TILE_ROWS + row] =
                        device.dmin[block * rows + source_row];
                }
            }
            for scale in 0..scale_count {
                let block = scale / SCALES;
                for row in 0..Q4K_TILE_ROWS {
                    let source_row = tile * Q4K_TILE_ROWS + row;
                    if source_row >= rows {
                        continue;
                    }
                    let index = (tile * scale_count + scale) * Q4K_TILE_ROWS + row;
                    packed.factors[index] = device.d[block * rows + source_row]
                        * f32::from(device.scales[scale * rows + source_row]);
                    packed.biases[index] = device.dmin[block * rows + source_row]
                        * f32::from(device.mins[scale * rows + source_row]);
                }
            }
        }
        Ok(packed)
    }

    /// Matrix rows, including the zero-padded tail of the last tile.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Matrix columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }
    /// Number of 16-row tiles.
    #[must_use]
    pub const fn tiles(&self) -> usize {
        self.tiles
    }
    /// Tile-major 4-bit codes.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }
    /// Tile-major scales.
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    /// Tile-major minima.
    #[must_use]
    pub fn mins(&self) -> &[u8] {
        &self.mins
    }
    /// Tile-major `d`.
    #[must_use]
    pub fn d(&self) -> &[f32] {
        &self.d
    }
    /// Tile-major `dmin`.
    #[must_use]
    pub fn dmin(&self) -> &[f32] {
        &self.dmin
    }
    /// Tile-major per-group `d * scale`.
    #[must_use]
    pub fn factors(&self) -> &[f32] {
        &self.factors
    }
    /// Tile-major per-group `dmin * min`.
    #[must_use]
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }
    /// The codes, factors and biases a device upload takes, moved out instead of copied.
    #[must_use]
    pub fn into_device_parts(self) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        (self.codes, self.factors, self.biases)
    }
}

/// Decode one 6-bit scale/min pair from the packed table.
const fn scale_min(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < HEADER_BYTES {
        (
            scales[index] & K_SCALE_MASK,
            scales[index + HEADER_BYTES] & K_SCALE_MASK,
        )
    } else {
        (
            (scales[index + HEADER_BYTES] & NIBBLE_MASK)
                | ((scales[index - HEADER_BYTES] >> K_SCALE_HIGH_SHIFT) << NIBBLE_BITS),
            (scales[index + HEADER_BYTES] >> NIBBLE_BITS)
                | ((scales[index] >> K_SCALE_HIGH_SHIFT) << NIBBLE_BITS),
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gguf_layout.rs"]
mod tests;
