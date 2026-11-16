//! One host layout for every grouped-int block family.
//!
//! Each family keeps its own packing, but callers use this enum for shape, codes, factors and
//! optional biases, so a new family adds one variant and its constructor instead of another
//! branch in the loader and projection binding.
use infer_core::Result;
use infer_models::{GroupedIntKind, Q4KDevice, Q4KTiled, Q6KTiled};

/// A grouped-int matrix in its kernel-facing host layout.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupedIntTiled {
    /// `Q4_K`.
    Q4(Q4KTiled),
    /// `Q6_K`.
    Q6(Q6KTiled),
}

impl GroupedIntTiled {
    /// Repack GGUF blocks with the family named by `kind`.
    /// # Errors
    /// Rejects a malformed shape or payload, or an unsupported family.
    pub fn from_blocks(
        kind: GroupedIntKind,
        rows: usize,
        columns: usize,
        blocks: &[u8],
    ) -> Result<Self> {
        Ok(match kind {
            GroupedIntKind::Q4K => Self::Q4(Q4KTiled::from_device(&Q4KDevice::from_blocks(
                rows, columns, blocks,
            )?)?),
            GroupedIntKind::Q6K => Self::Q6(Q6KTiled::from_blocks(rows, columns, blocks)?),
        })
    }

    /// Family tag.
    #[must_use]
    pub const fn kind(&self) -> GroupedIntKind {
        match self {
            Self::Q4(_) => GroupedIntKind::Q4K,
            Self::Q6(_) => GroupedIntKind::Q6K,
        }
    }

    /// Logical rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        match self {
            Self::Q4(layout) => layout.rows(),
            Self::Q6(layout) => layout.rows(),
        }
    }

    /// Logical columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        match self {
            Self::Q4(layout) => layout.columns(),
            Self::Q6(layout) => layout.columns(),
        }
    }

    /// Kernel-facing code bytes.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        match self {
            Self::Q4(layout) => layout.codes(),
            Self::Q6(layout) => layout.codes(),
        }
    }

    /// Per-group factors.
    #[must_use]
    pub fn factors(&self) -> &[f32] {
        match self {
            Self::Q4(layout) => layout.factors(),
            Self::Q6(layout) => layout.factors(),
        }
    }

    /// Per-group biases, empty for families without a min term.
    #[must_use]
    pub fn biases(&self) -> &[f32] {
        match self {
            Self::Q4(layout) => layout.biases(),
            Self::Q6(_) => &[],
        }
    }

    /// Upload shape of the code buffer.
    #[must_use]
    pub const fn code_shape(&self) -> [usize; 2] {
        self.kind().code_shape(self.rows(), self.columns())
    }
}
