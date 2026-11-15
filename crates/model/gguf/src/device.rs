//! Kernel-facing `Q4_K` layout and its host GEMV oracle.
//!
//! GGUF's 144-byte super-block is a load format. The device layout keeps the 4-bit codes but
//! expands the 6-bit scale/min table into one byte per 32-value group and stores `d`/`dmin` as
//! F32, so a kernel can index them without unpacking bits. The host oracle computes the same
//! grouped accumulation the device kernel must reproduce.
use crate::quant::half_to_f32;
use infer_core::{Error, Result};

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
            codes: Vec::with_capacity(code_len),
            scales: Vec::with_capacity(scale_len),
            mins: Vec::with_capacity(scale_len),
            d: Vec::with_capacity(block_count),
            dmin: Vec::with_capacity(block_count),
        };
        for block in blocks.as_chunks::<BLOCK_BYTES>().0 {
            device
                .codes
                .extend_from_slice(&block[CODE_OFFSET..CODE_OFFSET + CODE_BYTES]);
            device
                .d
                .push(half_to_f32(u16::from_le_bytes([block[0], block[1]])));
            device
                .dmin
                .push(half_to_f32(u16::from_le_bytes([block[2], block[3]])));
            let packed = &block[SCALE_OFFSET..SCALE_OFFSET + SCALE_BYTES];
            for index in 0..SCALES {
                let (scale, min) = scale_min(index, packed);
                device.scales.push(scale);
                device.mins.push(min);
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

    /// Expanded per-group scales, `[rows, columns / 32]`.
    #[must_use]
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    /// Expanded per-group minimum codes, `[rows, columns / 32]`.
    #[must_use]
    pub fn mins(&self) -> &[u8] {
        &self.mins
    }

    /// Per-super-block `d`, `[rows, columns / 256]`.
    #[must_use]
    pub fn d(&self) -> &[f32] {
        &self.d
    }

    /// Per-super-block `dmin`, `[rows, columns / 256]`.
    #[must_use]
    pub fn dmin(&self) -> &[f32] {
        &self.dmin
    }

    /// Packed 4-bit codes, `[rows, columns / 2]`, in the GGUF group order.
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
        let groups = self.columns / LANES;
        let mut sum = 0.0f32;
        for block in 0..blocks {
            let d = self.d[row * blocks + block];
            let dmin = self.dmin[row * blocks + block];
            for group in 0..GROUPS {
                for half in 0..2 {
                    let scale_index = block * SCALES + group * 2 + half;
                    let scale = f32::from(self.scales[row * groups + scale_index]);
                    let min = f32::from(self.mins[row * groups + scale_index]);
                    let mut code_dot = 0.0f32;
                    let mut x_sum = 0.0f32;
                    for lane in 0..LANES {
                        let column = block * QK_K + group * GROUP_VALUES + half * LANES + lane;
                        let byte = self.codes
                            [row * self.columns / 2 + block * CODE_BYTES + group * LANES + lane];
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
