//! Device-side interpolation of the learned vision position table.
#![expect(
    unsafe_code,
    reason = "Construction validates every table index, which the position taps operand requires"
)]
use crate::device::{CudaDevice, device_error};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::PositionTaps;
use infer_cuda_kernels::tile::runtime::Allocation;
use std::sync::Arc;

#[cfg(test)]
#[path = "../../tests/unit/vision_position.rs"]
mod tests;

/// Bilinear interpolation takes two taps on each image axis.
const TAPS: usize = 4;
/// Rows in one position-gather tile.
const TOKEN_TILE: usize = 32;
/// Channels in one position-gather tile.
const COLUMN_TILE: usize = 64;

/// Validated gather indices and interpolation weights, in tap-major order.
pub struct DeviceTaps {
    indices: Allocation<i32>,
    weights: Allocation<f32>,
    table: Allocation<bf16>,
    hidden: usize,
}

impl DeviceTaps {
    /// Validate and upload the bounded tap data for one interpolation.
    ///
    /// # Errors
    /// Rejects invalid shapes, out-of-range indices or failed CUDA transfers.
    pub fn new(
        device: &CudaDevice,
        tap_indices: &[[u32; TAPS]],
        tap_weights: &[[f32; TAPS]],
        table: &Arc<Tensor<bf16>>,
        rows: usize,
        hidden: usize,
        padded: usize,
    ) -> Result<Self> {
        let elements = rows
            .checked_mul(hidden)
            .ok_or_else(|| Error::invalid("vision position table overflow"))?;
        let bytes = elements
            .checked_mul(size_of::<bf16>())
            .ok_or_else(|| Error::invalid("vision position table overflow"))?;
        padded
            .checked_mul(hidden)
            .and_then(|value| value.checked_mul(size_of::<f32>()))
            .ok_or_else(|| Error::invalid("vision position activation overflow"))?;
        let table_shape = [
            i32::try_from(rows).map_err(device_error)?,
            i32::try_from(hidden).map_err(device_error)?,
        ];
        i32::try_from(padded).map_err(device_error)?;
        if rows == 0
            || hidden == 0
            || !hidden.is_multiple_of(COLUMN_TILE)
            || !padded.is_multiple_of(TOKEN_TILE)
            || tap_indices.is_empty()
            || tap_indices.len() != tap_weights.len()
            || tap_indices.len() > padded
            || table.num_bytes() != bytes
            || table.shape() != table_shape
            || i32::try_from(elements).is_err()
        {
            return Err(Error::invalid("vision position gather shape"));
        }
        let len = TAPS
            .checked_mul(padded)
            .ok_or_else(|| Error::invalid("vision position taps overflow"))?;
        let mut indices = vec![0_i32; len];
        let mut weights = vec![0.0_f32; len];
        for (patch, (item_indices, item_weights)) in tap_indices.iter().zip(tap_weights).enumerate()
        {
            for tap in 0..TAPS {
                let index = usize::try_from(item_indices[tap]).map_err(device_error)?;
                if index >= rows {
                    return Err(Error::invalid("vision position index out of table"));
                }
                indices[tap * padded + patch] = i32::try_from(index).map_err(device_error)?;
                weights[tap * padded + patch] = item_weights[tap];
            }
        }
        Ok(Self {
            indices: device.runtime().upload(indices)?,
            weights: device.runtime().upload(weights)?,
            table: Allocation::adopt(Arc::clone(table))?,
            hidden,
        })
    }

    /// The taps as the gather addresses them; a recording graph keeps their memory alive.
    #[must_use]
    pub fn taps(&self) -> PositionTaps<'_> {
        // SAFETY: construction checks each index against the table's rows and pads with row zero.
        unsafe {
            PositionTaps::new(
                self.table.operand(),
                self.indices.operand(),
                self.weights.operand(),
                self.hidden,
            )
        }
    }
}
