//! One host packer for every grouped-int format.
//!
//! GGML blocks are decoded span by span through the shared decoder and scattered into the
//! tile-major layout [`GroupedIntLayout`] describes. Formats differ only in [`CodePacking`] and
//! whether they carry biases, so a new format adds no branch here.
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    CodePacking, GROUPED_INT_TILE_ROWS, GroupedIntLayout, NIBBLE_PAIR_STRIDE,
};
use infer_ir::{GROUPED_INT_SPAN, GroupedIntFormat};
use infer_models::{GroupedSpan, decode_span, span_bytes};

/// Values in one nibble-packed group: each code byte holds a pair `NIBBLE_PAIR_STRIDE` apart.
const NIBBLE_GROUP: usize = 2 * NIBBLE_PAIR_STRIDE;
/// Width of one nibble.
const NIBBLE_SHIFT: u32 = 4;
/// Low bits of a code the nibble plane holds.
const NIBBLE_MASK: u8 = 0x0f;
/// Rows one code word carries, one per byte.
const WORD_ROWS: usize = 4;
/// Bytes of one code word.
const WORD_BYTES: usize = 4;
/// Bits of one byte.
const BYTE_BITS: usize = 8;

/// A grouped-int matrix in the tile-major layout the CUDA kernels read.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedIntTiled {
    format: GroupedIntFormat,
    rows: usize,
    columns: usize,
    codes: Vec<u8>,
    factors: Vec<f32>,
    biases: Option<Vec<f32>>,
}

impl GroupedIntTiled {
    /// Pack row-major GGML blocks of `format`.
    /// # Errors
    /// Rejects a shape the layout cannot tile, a payload whose length disagrees with the shape,
    /// or a malformed block.
    pub fn from_blocks(
        format: GroupedIntFormat,
        rows: usize,
        columns: usize,
        blocks: &[u8],
    ) -> Result<Self> {
        let layout = GroupedIntLayout::new(format, rows, columns)?;
        let stride = span_bytes(format)?;
        let spans = columns / GROUPED_INT_SPAN;
        let expected = rows
            .checked_mul(spans)
            .and_then(|count| count.checked_mul(stride))
            .ok_or_else(|| Error::invalid("grouped-int payload size overflow"))?;
        if blocks.len() != expected {
            return Err(Error::invalid(format!(
                "{} payload is {} bytes, expected {expected}",
                format.label(),
                blocks.len()
            )));
        }
        let mut packed = Self {
            format,
            rows,
            columns,
            codes: vec![0; layout.codes],
            factors: vec![0.0; layout.terms],
            biases: format.has_biases().then(|| vec![0.0; layout.terms]),
        };
        let mut span = GroupedSpan::default();
        for (index, stored) in blocks.chunks_exact(stride).enumerate() {
            decode_span(format, stored, &mut span)?;
            packed.scatter(index / spans, index % spans, &span);
        }
        Ok(packed)
    }

    /// Write one decoded span of `row` into its tile.
    fn scatter(&mut self, row: usize, span_index: usize, span: &GroupedSpan) {
        let spans = self.columns / GROUPED_INT_SPAN;
        let tile = row / GROUPED_INT_TILE_ROWS;
        let lane_row = row % GROUPED_INT_TILE_ROWS;
        let tile_span = tile * spans + span_index;
        let groups = self.format.span_groups();
        for group in 0..groups {
            let term = (tile_span * groups + group) * GROUPED_INT_TILE_ROWS + lane_row;
            self.factors[term] = span.factors[group];
            if let Some(biases) = &mut self.biases {
                biases[term] = span.biases[group];
            }
        }
        let packing = CodePacking::of(self.format);
        if packing == CodePacking::Byte {
            let base = tile_span * GROUPED_INT_SPAN * GROUPED_INT_TILE_ROWS;
            for (value, code) in span.codes.iter().enumerate() {
                self.codes[base + value * GROUPED_INT_TILE_ROWS + lane_row] = *code;
            }
            return;
        }
        let base = tile_span * GROUPED_INT_SPAN / 2 * GROUPED_INT_TILE_ROWS;
        for (group, values) in span.codes.as_chunks::<NIBBLE_GROUP>().0.iter().enumerate() {
            let (low, high) = values.split_at(NIBBLE_PAIR_STRIDE);
            for (lane, (low, high)) in low.iter().zip(high).enumerate() {
                let byte = (group * NIBBLE_PAIR_STRIDE + lane) * GROUPED_INT_TILE_ROWS;
                self.codes[base + byte + lane_row] =
                    (low & NIBBLE_MASK) | ((high & NIBBLE_MASK) << NIBBLE_SHIFT);
            }
        }
        if let CodePacking::NibbleHigh { high_bits } = packing {
            self.scatter_high(tile_span, lane_row, high_bits as usize, &span.codes);
        }
    }

    /// Write the bits above the nibbles of one span's `codes` into the high plane, as
    /// [`CodePacking::NibbleHigh`] lays them out.
    fn scatter_high(&mut self, tile_span: usize, lane_row: usize, high_bits: usize, codes: &[u8]) {
        let plane = self.rows * self.columns / 2;
        let wide = NIBBLE_PAIR_STRIDE * high_bits;
        let (word_column, row_bit) = (lane_row / WORD_ROWS, lane_row % WORD_ROWS * high_bits);
        for (value, code) in codes.iter().enumerate() {
            let (slot, lane) = (value / wide, value % wide);
            let word = (tile_span * wide + lane) * WORD_ROWS + word_column;
            let top = code >> NIBBLE_SHIFT;
            let bit = slot * WORD_ROWS * high_bits + row_bit;
            let byte = plane + word * WORD_BYTES + bit / BYTE_BITS;
            // A row's bits never straddle a byte: `high_bits` divides the bit offset.
            self.codes[byte] |= top << (bit % BYTE_BITS);
        }
    }

    /// Block family.
    #[must_use]
    pub const fn format(&self) -> GroupedIntFormat {
        self.format
    }

    /// Logical rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Logical columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Upload shape of the code buffer.
    #[must_use]
    pub const fn code_shape(&self) -> [usize; 2] {
        code_shape(self.format, self.rows, self.columns)
    }

    /// Tile-major codes.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Tile-major per-group factors.
    #[must_use]
    pub fn factors(&self) -> &[f32] {
        &self.factors
    }

    /// Tile-major per-group biases, present exactly when the format has them.
    #[must_use]
    pub fn biases(&self) -> Option<&[f32]> {
        self.biases.as_deref()
    }

    /// The codes, factors and biases a device upload takes, moved out instead of copied.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, Vec<f32>, Option<Vec<f32>>) {
        (self.codes, self.factors, self.biases)
    }
}

/// `[rows, code columns]` of an `[rows, columns]` matrix of `format`.
#[must_use]
pub const fn code_shape(format: GroupedIntFormat, rows: usize, columns: usize) -> [usize; 2] {
    [rows, CodePacking::of(format).code_bytes(columns)]
}

#[cfg(test)]
#[path = "../../tests/unit/grouped_int_tiled.rs"]
mod tests;
