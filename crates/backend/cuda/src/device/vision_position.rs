//! Device-side interpolation of the learned vision position table.
#![expect(
    unsafe_code,
    reason = "The position gather validates every table index and owns all device buffers through graph completion"
)]
use crate::device::{CudaDevice, device_error};
use cutile::cuda_async::cuda_graph::Scope;
use cutile::cuda_async::error::DeviceError;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use std::result::Result as StdResult;
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
    indices: Arc<Tensor<i32>>,
    weights: Arc<Tensor<f32>>,
    table: Arc<Tensor<bf16>>,
    hidden: usize,
    activation_shape: [i32; 2],
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
        let activation_shape = [i32::try_from(padded).map_err(device_error)?, table_shape[1]];
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
            indices: device.upload(indices, &[TAPS, padded])?,
            weights: device.upload(weights, &[TAPS, padded])?,
            table: table.clone(),
            hidden,
            activation_shape,
        })
    }

    /// Record a gather and add on the supplied graph, retaining its device tensors.
    ///
    /// # Errors
    /// Rejects mismatched activation shapes or graph-recording failures.
    pub fn record(
        &self,
        scope: &Scope,
        embedded: &Tensor<f32>,
        output: &mut Tensor<f32>,
    ) -> StdResult<(), DeviceError> {
        if embedded.shape() != self.activation_shape || output.shape() != self.activation_shape {
            return Err(DeviceError::Launch(
                "vision position activation shape".into(),
            ));
        }
        // SAFETY: construction checks each index and the full table span fits i32; the padded
        // entries address row zero. Shapes tile by 32x64, and the graph retains the table, taps,
        // input and output until the recorded work completes on the owning stream.
        scope.record(unsafe {
            kernels::add_positions(
                output.partition([TOKEN_TILE, COLUMN_TILE]),
                embedded,
                &self.table,
                &self.indices,
                &self.weights,
            )
            .generics(vec![self.hidden.to_string()])
        })?;
        Ok(())
    }
}

pub use infer_cuda_kernels::tile::vision_position as kernels;
