//! Kernel strategy independent of model names, storage and device ownership.
use crate::constants::DEFAULT_TILE_COLUMNS;
use crate::constants::DEFAULT_TILE_ROWS;
use infer_core::{Error, Result};
use serde::Serialize;

/// Tile row choices supported by the decode GEMV kernels.
const SUPPORTED_TILE_ROWS: [usize; 5] = [1, 2, 4, 8, 16];
/// Tile column bounds supported by the decode GEMV kernels.
const MIN_TILE_COLUMNS: usize = 64;
/// Tile column bounds supported by the decode GEMV kernels.
const MAX_TILE_COLUMNS: usize = 32768;
/// Bounded candidate set for decode tiling search; the original tile stays first.
const DECODE_TILE_CANDIDATES: [(usize, usize); 15] = [
    (4, 128),
    (1, 128),
    (1, 256),
    (1, 512),
    (2, 128),
    (2, 256),
    (2, 512),
    (4, 256),
    (4, 512),
    (4, 1024),
    (8, 128),
    (8, 256),
    (8, 512),
    (16, 128),
    (16, 256),
];
/// Wide candidates extend the search to one power-of-two full-width tile per row choice.
const WIDE_TILE_ROWS: [usize; 3] = [1, 2, 4];

/// Validated tile dimensions for a row-major decode projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LinearTiling {
    rows: usize,
    columns: usize,
}

impl LinearTiling {
    /// # Errors
    /// Rejects unsupported tile sizes before specializing device code.
    pub fn new(rows: usize, columns: usize) -> Result<Self> {
        if !SUPPORTED_TILE_ROWS.contains(&rows)
            || !(MIN_TILE_COLUMNS..=MAX_TILE_COLUMNS).contains(&columns)
            || !columns.is_power_of_two()
        {
            return Err(Error::invalid("unsupported CUDA linear tile"));
        }
        Ok(Self { rows, columns })
    }

    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> usize {
        self.columns
    }
}

/// Strategy boundary: model adapters supply dimensions, policies supply launch choices.
pub trait LinearStrategy {
    fn candidates(&self, columns: usize) -> Vec<LinearTiling>;
}

impl LinearStrategy for LinearTiling {
    fn candidates(&self, _columns: usize) -> Vec<LinearTiling> {
        vec![*self]
    }
}

/// Bounded search; the original 4 × 128 kernel remains the baseline.
pub struct DecodeSearch;

impl LinearStrategy for DecodeSearch {
    fn candidates(&self, columns: usize) -> Vec<LinearTiling> {
        let wide = columns
            .clamp(MIN_TILE_COLUMNS, MAX_TILE_COLUMNS)
            .next_power_of_two();
        DECODE_TILE_CANDIDATES
            .into_iter()
            .chain(WIDE_TILE_ROWS.map(|rows| (rows, wide)))
            .map(|(rows, columns)| LinearTiling { rows, columns })
            .collect()
    }
}

/// Conservative graph tile used when no measurement covers a projection.
///
/// Tile size is a property of the machine, so this module never chooses one from a device name,
/// architecture or model shape. Device-specific winners come from the measured tuning table or
/// the bounded autotuner; every other case gets the same baseline those measurements compare
/// against, which keeps a new GPU a measurement task instead of a code change.
/// # Errors
/// Returns an error if the built-in default violates the supported dimensions.
pub fn default_tiling() -> Result<LinearTiling> {
    LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS)
}

#[cfg(test)]
mod tests;
