//! Row-independent F32 GEMV with the original private accumulation and reduction tree.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::projection::pool_gemv as kernels;

use super::buffers::{RowMajor, buffer};
use super::capture::error;
use super::projection_lowering::Format;
use crate::tuning::pool_dense::{self, Choice};
use crate::tuning::projection_budget::ProjectionShape;
use crate::{mlp::ProjectionWeight, strategy::LinearTiling};
use infer_core::Result;
use infer_cuda_kernels::contract::{Buffer, Kernels, PoolGemvPlan, Recording};
#[cfg(test)]
use std::cell::Cell;
/// The pooled dense choice that records for a `shape` projection of `weight`, or `None` when
/// the native path records instead.
pub(super) fn lowered(weight: Format, shape: ProjectionShape, choice: Choice) -> Option<Choice> {
    if !pool_dense::searchable(shape) || !enabled() {
        return None;
    }
    #[cfg(test)]
    let choice = pool_dense::recorded_choice(choice);
    (choice != Choice::Native && weight == Format::Dense).then_some(choice)
}

/// Record a choice [`lowered`] returned.
pub(super) fn record(
    (kernels, scope): (&Kernels, &Recording<'_>),
    weight: &ProjectionWeight,
    input: Buffer<'_, f32>,
    output: RowMajor<'_, f32>,
    columns: usize,
    tile: LinearTiling,
    choice: Choice,
) -> Result<()> {
    let rows = input.len() / columns.max(1);
    let outputs = output.columns;
    let ProjectionWeight::Dense(weight) = weight else {
        return Err(error("pooled dense weight"));
    };
    if choice == Choice::Gemv {
        kernels
            .projection
            .pool_gemv_bf16(
                scope,
                &PoolGemvPlan {
                    rows,
                    outputs,
                    columns,
                    tile_rows: tile.rows(),
                    tile_columns: tile.columns(),
                },
                output.buffer,
                input,
                buffer(weight)?,
            )
            .map_err(error)?;
    } else {
        choice.record((kernels, scope), weight, input, output, columns, tile)?;
    }
    #[cfg(test)]
    CAPTURES.with(|n| n.set(n.get() + 1));
    Ok(())
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

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_pool_gemv_cell.rs"]
pub(super) mod cell_tests;
