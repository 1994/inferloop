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

#[cutile::module]
mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_0,
        Tile_1, Tile_2, andi, bitcast, cat, eq_tile, get_tile_block_id, iota, lt_tile, reduce_max,
        reduce_min, select, xori,
    };

    // Flip the sign bit for nonnegative values and all bits for negative values.
    const SIGN_SHIFT: u32 = 31;
    const MAGNITUDE_MASK: u32 = 0x7fff_ffff;
    const NONFINITE_BITS: u32 = 0x7f80_0000;
    const UNUSED_INDEX: i32 = 2_147_483_647;

    #[cutile::entry()]
    pub fn partials<const V: i32, const B: i32>(
        best_keys: &mut Tensor<u32, { [1, 1] }>,
        indices: &mut Tensor<i32, { [1, 1] }>,
        invalid: &mut Tensor<i32, { [1, 1] }>,
        logits: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let ids: Tile<i32, { [B] }> = iota(shape![B]) + (pid.1 * B).broadcast(shape![B]);
        let values = logits
            .partition(shape![1, B])
            .load([pid.0, pid.1])
            .reshape(shape![B]);
        let bits: Tile<u32, { [B] }> = bitcast(values);
        let sign_bit = 1u32.broadcast(shape![B]) << SIGN_SHIFT.broadcast(shape![B]);
        let all_bits = 0u32.broadcast(shape![B]) - 1u32.broadcast(shape![B]);
        let positive = lt_tile(bits, sign_bit);
        let keys = select(positive, xori(bits, sign_bit), xori(bits, all_bits));
        let present = lt_tile(ids, V.broadcast(shape![B]));
        let masked = select(present, keys, 0u32.broadcast(shape![B]));
        let best: Tile<u32, { [] }> = reduce_max(masked, 0i32);
        let tied = eq_tile(keys, best.reshape(shape![1]).broadcast(shape![B]));
        let candidates = select(
            present,
            select(tied, ids, UNUSED_INDEX.broadcast(shape![B])),
            UNUSED_INDEX.broadcast(shape![B]),
        );
        let token: Tile<i32, { [] }> = reduce_min(candidates, 0i32);
        let abs = andi(bits, MAGNITUDE_MASK.broadcast(shape![B]));
        let finite = lt_tile(abs, NONFINITE_BITS.broadcast(shape![B]));
        let bad = select(
            present,
            select(finite, 0i32.broadcast(shape![B]), 1i32.broadcast(shape![B])),
            0i32.broadcast(shape![B]),
        );
        let flag: Tile<i32, { [] }> = reduce_max(bad, 0i32);
        best_keys.store(best.reshape(shape![1, 1]));
        indices.store(token.reshape(shape![1, 1]));
        invalid.store(flag.reshape(shape![1, 1]));
    }

    #[cutile::entry()]
    pub fn finish<const N: i32, const B: i32, const START: i32, const ROWS: i32>(
        output: &mut Tensor<i32, { [1, 2] }>,
        keys: &Tensor<u32, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1, -1] }>,
        invalid: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        if pid.0 >= START && pid.0 < START + ROWS {
            // Modulo preserves the active-row index and expresses its bounded extent
            // to the launch checker when START=0 and the shared output is wider.
            let row = (pid.0 - START) % ROWS;
            let ids: Tile<i32, { [B] }> = iota(shape![B]);
            let present = lt_tile(ids, N.broadcast(shape![B]));
            let values = keys
                .partition(shape![1, B])
                .load([row, 0i32])
                .reshape(shape![B]);
            let masked = select(present, values, 0u32.broadcast(shape![B]));
            let best: Tile<u32, { [] }> = reduce_max(masked, 0i32);
            let candidates = indices
                .partition(shape![1, B])
                .load([row, 0i32])
                .reshape(shape![B]);
            let tied = eq_tile(values, best.reshape(shape![1]).broadcast(shape![B]));
            let candidates = select(
                present,
                select(tied, candidates, UNUSED_INDEX.broadcast(shape![B])),
                UNUSED_INDEX.broadcast(shape![B]),
            );
            let token: Tile<i32, { [] }> = reduce_min(candidates, 0i32);
            let flags = invalid
                .partition(shape![1, B])
                .load([row, 0i32])
                .reshape(shape![B]);
            let flag: Tile<i32, { [] }> = reduce_max(flags, 0i32);
            let result: Tile<i32, { [1, 2] }> = cat(
                token.reshape(shape![1, 1]),
                flag.reshape(shape![1, 1]),
                1i32,
            );
            output.store(result);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_greedy.rs"]
mod tests;
