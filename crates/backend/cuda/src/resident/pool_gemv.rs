//! Row-independent F32 GEMV with the original private accumulation and reduction tree.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        ElementType, Reshape, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_1, Tile_2, constant,
        convert_tile, get_tile_block_id, reduce_sum,
    };
    #[cutile::entry()]
    fn dense<E: ElementType, const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [1, BN] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![1, BK]);
        let wp = weight.partition(shape![BN, BK]);
        let mut sum: Tile<f32, { [BN, BK] }> = constant(0.0f32, shape![BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let a = xp.load([pid.0, k]);
            let b: Tile<f32, { [BN, BK] }> = convert_tile(wp.load([pid.1, k]));
            sum = sum + b * a.broadcast(shape![BN, BK]);
        }
        let reduced: Tile<f32, { [BN] }> = reduce_sum(sum, 1i32);
        out.store(reduced.reshape(shape![1, BN]));
    }
}

use crate::constants::CB_DECODE_SLOTS;
use crate::{mlp::ProjectionWeight, strategy::LinearTiling};
use cutile::half::bf16;
use cutile::prelude::*;
#[cfg(test)]
use std::cell::Cell;
const MAX_SMALL_OUTPUTS: usize = 128;
const MIN_SMALL_COLUMNS: usize = 4096;
const MAX_POOL_ROWS: usize = CB_DECODE_SLOTS;
pub(super) fn record(
    scope: &Scope,
    weight: &ProjectionWeight,
    input: &TensorView<'_, f32>,
    output: &mut Tensor<f32>,
    columns: usize,
    tile: LinearTiling,
) -> Result<bool, DeviceError> {
    let rows =
        usize::try_from(input.shape()[0]).map_err(|e| DeviceError::Internal(e.to_string()))?;
    let outputs =
        usize::try_from(output.shape()[1]).map_err(|e| DeviceError::Internal(e.to_string()))?;
    if rows == 0
        || rows > MAX_POOL_ROWS
        || outputs > MAX_SMALL_OUTPUTS
        || columns < MIN_SMALL_COLUMNS
        || !enabled()
    {
        return Ok(false);
    }
    let ProjectionWeight::Dense(weight) = weight else {
        return Ok(false);
    };
    scope.record(
        kernels::dense(output.partition([1, tile.rows()]), input, weight).generics(vec![
            bf16::DTYPE.as_str().into(),
            tile.rows().to_string(),
            tile.columns().to_string(),
            columns.to_string(),
        ]),
    )?;
    #[cfg(test)]
    CAPTURES.with(|n| n.set(n.get() + 1));
    Ok(true)
}
#[cfg(not(test))]
const fn enabled() -> bool {
    true
}
#[cfg(test)]
std::thread_local! {
    static ENABLED:Cell<bool> = const {Cell::new(true)};
    static CAPTURES:Cell<usize> = const {Cell::new(0)};
}
#[cfg(test)]
fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.with(Cell::get)
}
#[cfg(test)]
pub fn with_pool_gemv<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.with(|n| n.set(self.0));
        }
    }
    let _restore = Restore(ENABLED.with(|n| n.replace(enabled)));
    run()
}
