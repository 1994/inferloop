//! Row-independent F32 GEMV with the original private accumulation and reduction tree.
pub use infer_cuda_kernels::tile::projection::pool_gemv as kernels;

use super::capture::error;
use crate::tuning::pool_dense::{self, Choice};
use crate::{mlp::ProjectionWeight, strategy::LinearTiling};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_cuda_kernels::contract::{DevicePtr, PoolGemvPlan, Projection, Recording};
use infer_cuda_kernels::tile::projection::TileProjection;
#[cfg(test)]
use std::cell::Cell;
use std::sync::Arc;
#[expect(
    clippy::too_many_arguments,
    reason = "The caller's view does not expose its address, so it is passed alongside"
)]
pub(super) fn record(
    scope: &Scope,
    weight: &ProjectionWeight,
    input: &TensorView<'_, f32>,
    input_pointer: u64,
    output: &mut Tensor<f32>,
    columns: usize,
    tile: LinearTiling,
    choice: Choice,
) -> Result<bool, DeviceError> {
    let rows =
        usize::try_from(input.shape()[0]).map_err(|e| DeviceError::Internal(e.to_string()))?;
    let outputs =
        usize::try_from(output.shape()[1]).map_err(|e| DeviceError::Internal(e.to_string()))?;
    if !pool_dense::supported([rows, outputs, columns]) || !enabled() {
        return Ok(false);
    }
    #[cfg(test)]
    let choice = pool_dense::recorded_choice(choice);
    if choice == Choice::Native {
        return Ok(false);
    }
    let ProjectionWeight::Dense(weight) = weight else {
        return Ok(false);
    };
    if choice == Choice::Gemv {
        record_gemv(scope, weight, input, input_pointer, output, columns, tile)?;
    } else {
        choice.record(scope, weight, input, output, columns, tile)?;
    }
    #[cfg(test)]
    CAPTURES.with(|n| n.set(n.get() + 1));
    Ok(true)
}
/// The row-independent kernel goes through the family contract, as every source will.
#[expect(
    unsafe_code,
    reason = "The contract takes raw addresses of tensors the resident program owns"
)]
fn record_gemv(
    scope: &Scope,
    weight: &Arc<Tensor<bf16>>,
    input: &TensorView<'_, f32>,
    input_pointer: u64,
    output: &mut Tensor<f32>,
    columns: usize,
    tile: LinearTiling,
) -> Result<(), DeviceError> {
    let width = i32::try_from(columns).map_err(error)?;
    if input.strides() != [width, 1] || !weight.is_contiguous() || !output.is_contiguous() {
        return Err(error("pool GEMV operands must be contiguous"));
    }
    let plan = PoolGemvPlan {
        rows: usize::try_from(input.shape()[0]).map_err(error)?,
        outputs: usize::try_from(output.shape()[1]).map_err(error)?,
        columns,
        tile_rows: tile.rows(),
        tile_columns: tile.columns(),
    };
    // SAFETY: strides were checked above, and the resident program owns the three tensors until
    // its captured graphs are destroyed. The arena hands out distinct input and output slots.
    unsafe {
        TileProjection::new(output.device_id()).pool_gemv_bf16(
            &Recording::Scope(scope),
            &plan,
            DevicePtr(output.device_pointer().cu_deviceptr()),
            DevicePtr(input_pointer),
            DevicePtr(weight.device_pointer().cu_deviceptr()),
        )
    }
    .map_err(error)
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
