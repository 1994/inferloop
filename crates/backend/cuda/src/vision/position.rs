//! Device-side interpolation of the learned vision position table.
#![expect(
    unsafe_code,
    reason = "The position gather validates every table index and owns all device buffers through graph completion"
)]
use super::program::{COLUMN_TILE, TOKEN_TILE};
use crate::device::{CudaDevice, device_error};
use cutile::cuda_async::cuda_graph::Scope;
use cutile::cuda_async::error::DeviceError;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_models::vision::PositionTaps;
use std::sync::Arc;

#[cfg(test)]
#[path = "../../tests/unit/vision_position.rs"]
mod tests;

/// Bilinear interpolation takes two taps on each image axis.
const TAPS: usize = 4;

/// Validated gather indices and interpolation weights, in tap-major order.
pub(super) struct DeviceTaps {
    indices: Arc<Tensor<i32>>,
    weights: Arc<Tensor<f32>>,
    table: Arc<Tensor<bf16>>,
    hidden: usize,
    activation_shape: [i32; 2],
}

impl DeviceTaps {
    pub(super) fn new(
        device: &CudaDevice,
        taps: &PositionTaps,
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
            || taps.indices.is_empty()
            || taps.indices.len() != taps.weights.len()
            || taps.indices.len() > padded
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
        for (patch, (item_indices, item_weights)) in
            taps.indices.iter().zip(&taps.weights).enumerate()
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

    pub(super) fn record(
        &self,
        scope: &Scope,
        embedded: &Tensor<f32>,
        output: &mut Tensor<f32>,
    ) -> std::result::Result<(), DeviceError> {
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

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// Add a four-tap interpolation to a patch projection without a host round trip.
    #[cutile::entry()]
    unsafe fn add_positions<const H: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        embedded: &Tensor<f32, { [-1, H] }>,
        table: &Tensor<bf16, { [-1, H] }>,
        indices: &Tensor<i32, { [4, -1] }>,
        weights: &Tensor<f32, { [4, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let base: PointerTile<*const bf16, { [] }> = pointer_to_tile(table.as_ptr());
        let base: PointerTile<*const bf16, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*const bf16, { [32, 64] }> = base.broadcast(shape![32, 64]);
        let columns: Tile<i32, { [64] }> = iota(shape![64]) + (pid.1 * 64).broadcast(shape![64]);
        let columns: Tile<i32, { [1, 64] }> = columns.reshape(shape![1, 64]);
        let columns: Tile<i32, { [32, 64] }> = columns.broadcast(shape![32, 64]);
        let mut sum: Tile<f32, { [32, 64] }> = 0.0f32.broadcast(shape![32, 64]);
        for tap in 0i32..4i32 {
            let rows: Tile<i32, { [1, 32] }> = indices.partition(shape![1, 32]).load([tap, pid.0]);
            let rows: Tile<i32, { [32, 1] }> = rows.reshape(shape![32, 1]);
            let rows: Tile<i32, { [32, 64] }> = rows.broadcast(shape![32, 64]);
            let offsets: Tile<i32, { [32, 64] }> = rows * H.broadcast(shape![32, 64]) + columns;
            let addresses: PointerTile<*const bf16, { [32, 64] }> = addptr_tile(base, offsets);
            // SAFETY: the host validates every row and column address; all padded rows target
            // table row zero, and the source tensor remains live through graph completion.
            let (values, _): (Tile<bf16, { [32, 64] }>, Token) = unsafe {
                load_ptr_tko(
                    addresses,
                    ordering::Relaxed,
                    Some(scope::Device),
                    None,
                    None,
                    None,
                    Latency::<0>,
                )
            };
            let weights: Tile<f32, { [1, 32] }> =
                weights.partition(shape![1, 32]).load([tap, pid.0]);
            let weights: Tile<f32, { [32, 1] }> = weights.reshape(shape![32, 1]);
            let weights: Tile<f32, { [32, 64] }> = weights.broadcast(shape![32, 64]);
            let values: Tile<f32, { [32, 64] }> = convert_tile(values);
            sum = fma(weights, values, sum, rounding::NearestEven, ftz::Disabled);
        }
        let embedded: Tile<f32, { [32, 64] }> =
            embedded.partition(shape![32, 64]).load([pid.0, pid.1]);
        out.store(embedded + sum);
    }
}
