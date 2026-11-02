//! Exact token FP8 quantization with parallel conversion for large aligned rows.
#[cfg(test)]
pub use infer_cuda_kernels::tile::projection::fp8_quant_reduction as kernels;

use infer_cuda_kernels::contract::Fp8QuantKind;
#[cfg(test)]
use std::cell::Cell;
const SERIAL_COLUMNS: usize = 4096;
const CONVERSION_COLUMNS: usize = 512;

pub(super) const fn parallel_columns(columns: usize) -> bool {
    columns > SERIAL_COLUMNS && columns.is_multiple_of(CONVERSION_COLUMNS)
}

/// The conversion a row of `columns` takes; `maximum_ready` consumes a maximum the immediately
/// preceding normalization produced. Large rows partition conversion independently from the row
/// maximum, and both stages reuse the existing quantized activation and scale buffers.
#[cfg_attr(
    not(test),
    expect(
        clippy::missing_const_for_fn,
        reason = "The test build reads a thread-local override"
    )
)]
pub(super) fn kind(columns: usize, maximum_ready: bool) -> Fp8QuantKind {
    if !maximum_ready && (!parallel_columns(columns) || !enabled()) {
        return Fp8QuantKind::Whole;
    }
    #[cfg(test)]
    if !maximum_ready {
        CAPTURES.with(|value| value.set(value.get() + 1));
    }
    Fp8QuantKind::Parallel {
        conversion_columns: CONVERSION_COLUMNS,
        maximum_ready,
    }
}
#[cfg(not(test))]
const fn enabled() -> bool {
    true
}
#[cfg(test)]
std::thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
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
pub fn with_parallel_conversion<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(ENABLED.with(|value| value.replace(enabled)));
    run()
}

#[cfg(test)]
#[path = "../../tests/unit/resident_fp8_quant_reduction.rs"]
mod tests;
