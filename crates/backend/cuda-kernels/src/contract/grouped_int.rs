//! Grouped integer weight contracts.
//!
//! Every [`GroupedIntFormat`] shares one tile-major layout: rows are tiled by
//! [`GROUPED_INT_TILE_ROWS`], and each 256-value span of a tile stores its codes followed by one
//! factor (and bias) per group; a [`CodePacking::NibbleHigh`] matrix stores the bits above the
//! nibbles in a second plane after the whole matrix's nibbles. Formats differ only in
//! [`CodePacking`] and their [`infer_ir::CodeOffset`], so the host packer and the kernels branch
//! on those two properties, never on a format name.
use crate::contract::buffer::Buffer;
use infer_core::{Error, Result};
use infer_ir::{GROUPED_INT_SPAN, GroupedIntFormat};

/// Output rows one grouped-int tile covers.
pub const GROUPED_INT_TILE_ROWS: usize = 16;
/// Activation rows one grouped-int rows-kernel CTA multiplies per decoded weight tile. On the
/// RTX 5090, 64 and 128 rows ran 1.4-3x slower over 128-2048 rows and 16 rows slower still.
pub const GROUPED_INT_GEMM_ROWS: usize = 32;
/// Widest code a nibble packing holds.
const NIBBLE_BITS: u32 = 4;
/// Width of a byte-per-code packing.
const BYTE_BITS: u32 = 8;
/// Code bits one device byte holds.
const BYTE_WIDTH: usize = 8;
/// Values one nibble-packed 64-value group holds in each of its 32 code bytes.
pub const NIBBLE_PAIR_STRIDE: usize = 32;

/// How codes occupy device bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodePacking {
    /// Two codes per byte: lane `l` of each 64-value group holds value `l` in the low nibble and
    /// value `l + 32` in the high nibble.
    Nibble,
    /// The low four bits of every code as in [`Self::Nibble`], and the `high_bits` above them in
    /// a plane after the whole matrix's nibbles. Each 256-value span of a tile holds
    /// `[32 * high_bits, 4]` little-endian `u32` words: with `slots = 8 / high_bits`, bit
    /// `4 * high_bits * k + high_bits * b + i` of word `(j, w)` is bit `4 + i` of value
    /// `32 * high_bits * k + j` in row `4w + b`.
    NibbleHigh {
        /// Code bits above the nibble: 1 or 2.
        high_bits: u32,
    },
    /// One code per byte.
    Byte,
}

impl CodePacking {
    /// The packing the kernels use for `format`.
    #[must_use]
    pub const fn of(format: GroupedIntFormat) -> Self {
        let bits = format.code_bits();
        if bits <= NIBBLE_BITS {
            Self::Nibble
        } else if bits < BYTE_BITS {
            Self::NibbleHigh {
                high_bits: bits - NIBBLE_BITS,
            }
        } else {
            Self::Byte
        }
    }

    /// Device bits of one code.
    #[must_use]
    pub const fn code_bits(self) -> usize {
        match self {
            Self::Nibble => NIBBLE_BITS as usize,
            Self::NibbleHigh { high_bits } => (NIBBLE_BITS + high_bits) as usize,
            Self::Byte => BYTE_BITS as usize,
        }
    }

    /// Code bytes of `values` codes; `values` is a whole number of spans.
    #[must_use]
    pub const fn code_bytes(self, values: usize) -> usize {
        values / BYTE_WIDTH * self.code_bits()
    }

    /// Codes in `bytes` code bytes; the inverse of [`Self::code_bytes`].
    #[must_use]
    pub const fn code_values(self, bytes: usize) -> usize {
        bytes * BYTE_WIDTH / self.code_bits()
    }

    /// Bytes of the nibble plane that leads the codes of `values` values, when the packing splits
    /// its codes into two planes.
    #[must_use]
    pub const fn nibble_plane(self, values: usize) -> Option<usize> {
        match self {
            Self::NibbleHigh { .. } => Some(values / 2),
            Self::Nibble | Self::Byte => None,
        }
    }
}

/// Element counts of one grouped-int matrix in the tile-major layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedIntLayout {
    /// Code bytes.
    pub codes: usize,
    /// Factors, and biases when the format has them.
    pub terms: usize,
}

impl GroupedIntLayout {
    /// The layout of an `[outputs, columns]` matrix.
    /// # Errors
    /// Rejects outputs that do not fill whole tiles, columns that are not whole spans, or sizes
    /// that overflow.
    pub fn new(format: GroupedIntFormat, outputs: usize, columns: usize) -> Result<Self> {
        if outputs == 0
            || columns == 0
            || !outputs.is_multiple_of(GROUPED_INT_TILE_ROWS)
            || !columns.is_multiple_of(GROUPED_INT_SPAN)
        {
            return Err(Error::unsupported(format!(
                "{} matrix [{outputs}, {columns}] needs {GROUPED_INT_TILE_ROWS}-row tiles and \
                 {GROUPED_INT_SPAN}-value spans",
                format.label()
            )));
        }
        let values = outputs
            .checked_mul(columns)
            .ok_or_else(|| Error::invalid("grouped-int matrix size overflow"))?;
        Ok(Self {
            codes: CodePacking::of(format).code_bytes(values),
            terms: values / format.group_values(),
        })
    }
}

/// Geometry of one grouped-integer GEMV launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedIntPlan {
    /// Output rows one CTA covers; currently 16.
    pub tile_rows: usize,
    /// Reduction length.
    pub columns: usize,
}

/// A grouped-integer matrix in the tile-major device layout.
#[derive(Clone, Copy, Debug)]
pub struct GroupedIntWeight<'a> {
    /// Block family; selects the packing and offset.
    pub format: GroupedIntFormat,
    /// Tile-major codes.
    pub codes: Buffer<'a, u8>,
    /// Tile-major per-group factors.
    pub factors: Buffer<'a, f32>,
    /// Tile-major per-group biases, present exactly when the format has them.
    pub biases: Option<Buffer<'a, f32>>,
}
