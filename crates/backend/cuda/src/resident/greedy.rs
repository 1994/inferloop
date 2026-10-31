//! Exact finite F32 greedy selection; only token IDs and validation flags leave the GPU.
use crate::device::Readbacks;
use crate::device::{CudaDevice, device_error};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{Error, Result};
use std::result::Result as StdResult;

/// Candidates to measure per device and row/vocabulary geometry before production selection.
pub const TILE_SEARCH: [usize; 4] = [256, 512, 1024, 2048];

pub struct Greedy {
    rows: usize,
    vocabulary: usize,
    tile: usize,
    blocks: usize,
    keys: Tensor<u32>,
    indices: Tensor<i32>,
    invalid: Tensor<i32>,
    pub tokens: Tensor<i32>,
    readbacks: Readbacks<i32>,
}
impl Greedy {
    pub fn new(device: &CudaDevice, rows: usize, vocabulary: usize, tile: usize) -> Result<Self> {
        if rows == 0 || vocabulary == 0 || !TILE_SEARCH.contains(&tile) {
            return Err(Error::invalid("greedy reduction geometry"));
        }
        i32::try_from(vocabulary).map_err(device_error)?;
        let blocks = vocabulary.div_ceil(tile);
        i32::try_from(blocks.next_power_of_two()).map_err(device_error)?;
        Ok(Self {
            readbacks: Readbacks::allocated(device, rows * 2)?,
            rows,
            vocabulary,
            tile,
            blocks,
            keys: api::zeros::<u32>(&[rows, blocks])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            indices: api::zeros::<i32>(&[rows, blocks])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            invalid: api::zeros::<i32>(&[rows, blocks])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            tokens: api::zeros::<i32>(&[rows, 2])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }
    pub fn record(&mut self, scope: &Scope, logits: &Tensor<f32>) -> StdResult<(), DeviceError> {
        self.record_partials(scope, logits)?;
        scope.record(
            kernels::finish(
                (&mut self.tokens).partition([1, 2]),
                &self.keys,
                &self.indices,
                &self.invalid,
            )
            .generics(vec![
                self.blocks.to_string(),
                self.blocks.next_power_of_two().to_string(),
                "0".into(),
                self.rows.to_string(),
            ]),
        )?;
        Ok(())
    }

    pub(crate) fn record_into(
        &mut self,
        scope: &Scope,
        logits: &Tensor<f32>,
        output: &mut Tensor<i32>,
        offset: usize,
    ) -> StdResult<(), DeviceError> {
        if !output.size().is_multiple_of(2)
            || offset
                .checked_add(self.rows)
                .is_none_or(|end| end > output.size() / 2)
        {
            return Err(tensor_error("greedy output row range").into());
        }
        self.record_partials(scope, logits)?;
        scope.record(
            kernels::finish(
                output.partition([1, 2]),
                &self.keys,
                &self.indices,
                &self.invalid,
            )
            .generics(vec![
                self.blocks.to_string(),
                self.blocks.next_power_of_two().to_string(),
                offset.to_string(),
                self.rows.to_string(),
            ]),
        )?;
        Ok(())
    }

    fn record_partials(
        &mut self,
        scope: &Scope,
        logits: &Tensor<f32>,
    ) -> StdResult<(), DeviceError> {
        let input = logits.view(&[self.rows, self.vocabulary])?;
        scope.record(
            kernels::partials(
                (&mut self.keys).partition([1, 1]),
                (&mut self.indices).partition([1, 1]),
                (&mut self.invalid).partition([1, 1]),
                &input,
            )
            .generics(vec![self.vocabulary.to_string(), self.tile.to_string()]),
        )?;
        Ok(())
    }
    /// Canonical frontend identities include both reductions and their exact geometry.
    pub fn fingerprint(&mut self, device: &CudaDevice, logits: &Tensor<f32>) -> Result<String> {
        let input = logits
            .view(&[self.rows, self.vocabulary])
            .map_err(device_error)?;
        let first = kernels::partials(
            (&mut self.keys).partition([1, 1]),
            (&mut self.indices).partition([1, 1]),
            (&mut self.invalid).partition([1, 1]),
            &input,
        )
        .generics(vec![self.vocabulary.to_string(), self.tile.to_string()])
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?;
        let second = kernels::finish(
            (&mut self.tokens).partition([1, 2]),
            &self.keys,
            &self.indices,
            &self.invalid,
        )
        .generics(vec![
            self.blocks.to_string(),
            self.blocks.next_power_of_two().to_string(),
            "0".into(),
            self.rows.to_string(),
        ])
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?;
        Ok(format!("{first}:{second}"))
    }

    pub fn read(&self, device: &CudaDevice) -> Result<Vec<u32>> {
        self.decode_values(&device.read_borrowed(&self.tokens)?)
    }

    pub(crate) async fn read_async(&mut self, device: &CudaDevice) -> Result<Vec<u32>> {
        let values = device.read_async(&self.tokens, &mut self.readbacks).await?;
        self.decode_values(&values)
    }

    fn decode_values(&self, values: &[i32]) -> Result<Vec<u32>> {
        values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|row| Self::decode(row, self.vocabulary))
            .collect()
    }

    pub(crate) fn decode(row: &[i32], vocabulary: usize) -> Result<u32> {
        if row.len() != 2 || row[1] != 0 || row[0] < 0 {
            return Err(Error::invalid("invalid sampling logits"));
        }
        let token = u32::try_from(row[0]).map_err(device_error)?;
        if usize::try_from(token).map_err(device_error)? >= vocabulary {
            return Err(Error::invariant("greedy token outside vocabulary"));
        }
        Ok(token)
    }
}

pub use infer_cuda_kernels::tile::sampling::greedy as kernels;

#[cfg(test)]
#[path = "../../tests/unit/resident_greedy.rs"]
mod tests;
