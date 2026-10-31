//! Row-independent F32 GEMV with the original private accumulation and reduction tree.
pub use infer_cuda_kernels::tile::projection::pool_gemv as kernels;

use crate::tuning::pool_dense::{self, Choice};
use crate::{mlp::ProjectionWeight, strategy::LinearTiling};
use cutile::prelude::*;
#[cfg(test)]
use std::cell::Cell;
pub(super) fn record(
    scope: &Scope,
    weight: &ProjectionWeight,
    input: &TensorView<'_, f32>,
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
    choice.record(scope, weight, input, output, columns, tile)?;
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
