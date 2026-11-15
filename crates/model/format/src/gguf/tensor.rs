//! GGML tensor types and their stored block geometry.
use infer_core::{Error, Result};

/// Elements in one k-quant super-block.
const QK_K: u64 = 256;
/// Elements in one legacy 4/5/8-bit quant block.
const QK_LEGACY: u64 = 32;
/// Stored bytes of one `Q2_K` block.
const Q2_K_BYTES: u64 = 84;
/// Stored bytes of one `Q3_K` block.
const Q3_K_BYTES: u64 = 110;
/// Stored bytes of one `Q4_K` block.
const Q4_K_BYTES: u64 = 144;
/// Stored bytes of one `Q5_K` block.
const Q5_K_BYTES: u64 = 176;
/// Stored bytes of one `Q6_K` block.
const Q6_K_BYTES: u64 = 210;
/// Stored bytes of one `Q8_K` block.
const Q8_K_BYTES: u64 = 292;
/// Bytes per F32 value.
const F32_BYTES: u64 = 4;
/// Bytes per 16-bit value.
const F16_BYTES: u64 = 2;
/// Bytes per 64-bit value.
const F64_BYTES: u64 = 8;
/// Bytes per I8 value.
const I8_BYTES: u64 = 1;
/// Stored bytes of one `Q4_0` block.
const Q4_0_BYTES: u64 = 18;
/// Stored bytes of one `Q4_1` block.
const Q4_1_BYTES: u64 = 20;
/// Stored bytes of one `Q5_0` block.
const Q5_0_BYTES: u64 = 22;
/// Stored bytes of one `Q5_1` block.
const Q5_1_BYTES: u64 = 24;
/// Stored bytes of one `Q8_0` block.
const Q8_0_BYTES: u64 = 34;
/// Stored bytes of one `Q8_1` block.
const Q8_1_BYTES: u64 = 36;

/// A tensor type id from the GGUF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
    /// A type this crate does not name, kept so a file can still be inspected.
    Other(u32),
}

/// `(type, id, block elements, block bytes)`, sorted by id. A zero byte count means the block
/// geometry is unknown to this crate.
const TYPE_TABLE: &[(GgufType, u32, u64, u64)] = &[
    (GgufType::F32, 0, 1, F32_BYTES),
    (GgufType::F16, 1, 1, F16_BYTES),
    (GgufType::Q4_0, 2, QK_LEGACY, Q4_0_BYTES),
    (GgufType::Q4_1, 3, QK_LEGACY, Q4_1_BYTES),
    (GgufType::Q5_0, 6, QK_LEGACY, Q5_0_BYTES),
    (GgufType::Q5_1, 7, QK_LEGACY, Q5_1_BYTES),
    (GgufType::Q8_0, 8, QK_LEGACY, Q8_0_BYTES),
    (GgufType::Q8_1, 9, QK_LEGACY, Q8_1_BYTES),
    (GgufType::Q2K, 10, QK_K, Q2_K_BYTES),
    (GgufType::Q3K, 11, QK_K, Q3_K_BYTES),
    (GgufType::Q4K, 12, QK_K, Q4_K_BYTES),
    (GgufType::Q5K, 13, QK_K, Q5_K_BYTES),
    (GgufType::Q6K, 14, QK_K, Q6_K_BYTES),
    (GgufType::Q8K, 15, QK_K, Q8_K_BYTES),
    (GgufType::I8, 24, 1, I8_BYTES),
    (GgufType::I16, 25, 1, F16_BYTES),
    (GgufType::I32, 26, 1, F32_BYTES),
    (GgufType::I64, 27, 1, F64_BYTES),
    (GgufType::F64, 28, 1, F64_BYTES),
    (GgufType::Bf16, 30, 1, F16_BYTES),
];

impl GgufType {
    /// The type for a stored GGUF type id.
    #[must_use]
    pub fn from_id(id: u32) -> Self {
        TYPE_TABLE
            .iter()
            .find(|entry| entry.1 == id)
            .map_or(Self::Other(id), |entry| entry.0)
    }

    /// The stored type id.
    #[must_use]
    pub fn id(self) -> u32 {
        TYPE_TABLE
            .iter()
            .find(|entry| entry.0 == self)
            .map_or_else(|| self.raw_id(), |entry| entry.1)
    }

    const fn raw_id(self) -> u32 {
        match self {
            Self::Other(id) => id,
            _ => u32::MAX,
        }
    }

    /// Stable lower-case name used by GGUF tooling and diagnosis.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Q8K => "q8_k",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::Bf16 => "bf16",
            Self::Other(_) => "unknown",
        }
    }

    /// `(elements, bytes)` of one stored block, when the type's geometry is known.
    ///
    /// `None` means this crate cannot compute a stored size for the type; callers must reject the
    /// tensor rather than guess. Decoding additionally supports only a subset.
    #[must_use]
    pub fn block_layout(self) -> Option<(u64, u64)> {
        TYPE_TABLE
            .iter()
            .find(|entry| entry.0 == self && entry.3 != 0)
            .map(|entry| (entry.2, entry.3))
    }

    /// Whether the type stores quantized blocks.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::Q8K
        )
    }

    /// Stored bytes for `elements` values.
    /// # Errors
    /// Returns an unsupported error for an unknown type and an invalid-input error when the
    /// element count is not a whole number of blocks or the byte count overflows.
    pub fn byte_len(self, elements: u64) -> Result<u64> {
        let (block_elements, block_bytes) = self.block_layout().ok_or_else(|| {
            Error::unsupported(format!(
                "GGUF tensor type {} has no known block size",
                self.id()
            ))
        })?;
        if !elements.is_multiple_of(block_elements) {
            return Err(Error::invalid(format!(
                "GGUF tensor has {elements} elements, not a whole number of {} blocks",
                self.name()
            )));
        }
        (elements / block_elements)
            .checked_mul(block_bytes)
            .ok_or_else(|| Error::invalid("GGUF tensor byte size overflow"))
    }
}

/// The row-major shape for GGML dimensions, which are stored fastest-axis first.
/// # Errors
/// Returns an invalid-input error when a dimension exceeds the address space.
pub fn row_major_shape(dims: &[u64]) -> Result<Vec<usize>> {
    dims.iter()
        .rev()
        .map(|dim| {
            usize::try_from(*dim)
                .map_err(|_| Error::invalid("GGUF tensor dimension exceeds address space"))
        })
        .collect()
}

/// Element count for GGML dimensions.
/// # Errors
/// Returns an invalid-input error when the product overflows.
pub fn elements(dims: &[u64]) -> Result<u64> {
    dims.iter().try_fold(1u64, |total, dim| {
        total
            .checked_mul(*dim)
            .ok_or_else(|| Error::invalid("GGUF tensor element count overflow"))
    })
}
