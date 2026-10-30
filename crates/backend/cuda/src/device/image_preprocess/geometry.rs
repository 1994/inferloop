//! Validated dimensions for the fixed-layout RGB8 image kernels.
use infer_core::{Error, Result};

/// Channels in an interleaved RGB8 image.
pub(super) const CHANNELS: usize = 3;
/// Pixels along one square patch edge in the qualified kernel.
const PATCH: usize = 16;
/// Frames replicated into each image patch.
const TEMPORAL: usize = 2;
/// Adjacent patches merged along each spatial axis.
const MERGE: usize = 2;
/// Patch rows required by the vision projection's token tile.
const PATCH_TILE: usize = 32;
/// Flattened channel, temporal and spatial elements in one patch.
pub(super) const PATCH_WIDTH: usize = CHANNELS * TEMPORAL * PATCH * PATCH;

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub batch: i32,
    pub source_height: i32,
    pub source_width: i32,
    pub target_height: i32,
    pub target_width: i32,
    pub padded_patches: i32,
    pub input_elements: usize,
    pub horizontal_elements: usize,
    pub output_elements: usize,
}

impl Geometry {
    /// Validate the integer ABI and every address range before device allocation or launch.
    pub fn new(batch: usize, source: [usize; 2], target: [usize; 2]) -> Result<Self> {
        if batch == 0
            || source.contains(&0)
            || target.contains(&0)
            || !target.iter().all(|axis| axis.is_multiple_of(PATCH * MERGE))
        {
            return Err(Error::invalid("unsupported RGB image geometry"));
        }
        let input_elements = elements(&[batch, source[0], source[1], CHANNELS])?;
        let horizontal_elements = elements(&[batch, source[0], target[1], CHANNELS])?;
        let patches = elements(&[target[0] / PATCH, target[1] / PATCH])?;
        let padded_patches = patches
            .checked_next_multiple_of(PATCH_TILE)
            .ok_or_else(|| Error::invalid("image patch padding overflow"))?;
        let output_elements = elements(&[batch, padded_patches, CHANNELS, TEMPORAL, PATCH, PATCH])?;
        // cuTile's one-dimensional tensor metadata uses signed 32-bit element counts.
        for count in [input_elements, horizontal_elements, output_elements] {
            checked_i32(count)?;
        }
        Ok(Self {
            batch: checked_i32(batch)?,
            source_height: checked_i32(source[0])?,
            source_width: checked_i32(source[1])?,
            target_height: checked_i32(target[0])?,
            target_width: checked_i32(target[1])?,
            padded_patches: checked_i32(padded_patches)?,
            input_elements,
            horizontal_elements,
            output_elements,
        })
    }

    pub const fn is_identity(self) -> bool {
        self.source_height == self.target_height && self.source_width == self.target_width
    }
}

fn elements(dimensions: &[usize]) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| Error::invalid("image tensor dimensions overflow"))
}

fn checked_i32(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::invalid("image tensor exceeds CUDA integer ABI"))
}

#[cfg(test)]
#[path = "../../../tests/unit/vision_image_geometry.rs"]
mod tests;
