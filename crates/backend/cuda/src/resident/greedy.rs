//! Exact finite F32 greedy selection; only token IDs and validation flags leave the GPU.
use super::buffers::{Operand, buffer};
use super::capture::error;
use crate::device::Readbacks;
use crate::device::{CudaDevice, device_error};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{Buffer, GreedyPartials, GreedyPlan, Kernels, Recording};
use infer_cuda_kernels::tile::identity;
use infer_cuda_kernels::tile::runtime::Allocation;
use std::sync::Arc;

/// Values each selected row writes: the token and its validation flag.
const TOKEN_FIELDS: usize = 2;

/// Candidates to measure per device and row/vocabulary geometry before production selection.
pub const TILE_SEARCH: [usize; 4] = [256, 512, 1024, 2048];

pub struct Greedy {
    rows: usize,
    vocabulary: usize,
    tile: usize,
    blocks: usize,
    keys: Allocation<u32>,
    indices: Allocation<i32>,
    invalid: Allocation<i32>,
    pub tokens: Allocation<i32>,
    readbacks: Readbacks<i32>,
    kernels: Arc<Kernels>,
}
impl Greedy {
    pub fn new(device: &CudaDevice, rows: usize, vocabulary: usize, tile: usize) -> Result<Self> {
        if rows == 0 || vocabulary == 0 || !TILE_SEARCH.contains(&tile) {
            return Err(Error::invalid("greedy reduction geometry"));
        }
        i32::try_from(vocabulary).map_err(device_error)?;
        let blocks = vocabulary.div_ceil(tile);
        i32::try_from(blocks.next_power_of_two()).map_err(device_error)?;
        let runtime = device.runtime();
        Ok(Self {
            kernels: device.kernels(),
            readbacks: Readbacks::allocated(device, rows * 2)?,
            rows,
            vocabulary,
            tile,
            blocks,
            keys: runtime.zeros(rows * blocks)?,
            indices: runtime.zeros(rows * blocks)?,
            invalid: runtime.zeros(rows * blocks)?,
            tokens: runtime.zeros(rows * TOKEN_FIELDS)?,
        })
    }
    pub fn record<L: Operand<Marker = f32> + ?Sized>(
        &self,
        scope: &Recording<'_>,
        logits: &L,
    ) -> Result<()> {
        self.record_partials(scope, logits)?;
        self.record_finish(scope, buffer(&self.tokens)?, 0)
    }

    pub(crate) fn record_into<L: Operand<Marker = f32> + ?Sized>(
        &self,
        scope: &Recording<'_>,
        logits: &L,
        output: &Allocation<i32>,
        offset: usize,
    ) -> Result<()> {
        if !output.len().is_multiple_of(TOKEN_FIELDS)
            || offset
                .checked_add(self.rows)
                .is_none_or(|end| end > output.len() / TOKEN_FIELDS)
        {
            return Err(device_error("greedy output row range"));
        }
        self.record_partials(scope, logits)?;
        self.record_finish(scope, buffer(output)?, offset)
    }

    const fn plan(&self) -> GreedyPlan {
        GreedyPlan {
            rows: self.rows,
            vocabulary: self.vocabulary,
            tile: self.tile,
            blocks: self.blocks,
        }
    }

    fn partials(&self) -> Result<GreedyPartials<'_>> {
        Ok(GreedyPartials {
            keys: buffer(&self.keys)?,
            indices: buffer(&self.indices)?,
            invalid: buffer(&self.invalid)?,
        })
    }

    fn record_finish(
        &self,
        scope: &Recording<'_>,
        output: Buffer<'_, i32>,
        offset: usize,
    ) -> Result<()> {
        self.kernels
            .sampling
            .greedy_finish(scope, &self.plan(), offset, output, self.partials()?)
            .map_err(error)
    }

    fn record_partials<L: Operand<Marker = f32> + ?Sized>(
        &self,
        scope: &Recording<'_>,
        logits: &L,
    ) -> Result<()> {
        self.kernels
            .sampling
            .greedy_partials(scope, &self.plan(), self.partials()?, buffer(logits)?)
            .map_err(error)
    }
    /// Canonical frontend identities include both reductions and their exact geometry.
    pub fn fingerprint(&self, device: &CudaDevice) -> Result<String> {
        identity::greedy_reductions(&device.stream, &self.plan())
    }

    pub fn read(&self, device: &CudaDevice) -> Result<Vec<u32>> {
        self.decode_values(&device.runtime().read(&self.tokens)?)
    }

    pub(crate) async fn read_async(&mut self, device: &CudaDevice) -> Result<Vec<u32>> {
        let values = device
            .read_allocation(&self.tokens, &mut self.readbacks)
            .await?;
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

#[cfg(test)]
#[path = "../../tests/unit/resident_greedy.rs"]
mod tests;
