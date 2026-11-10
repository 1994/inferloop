//! RMS normalization primes only the immediately following fused NVFP4 gate/up pass.
use crate::constants::NVFP4_QUANT_CODES_TILE;
#[cfg(test)]
use std::cell::Cell;

/// Full quantizer windows retain the canonical RMS reduction layout. Partial windows use
/// separate normalization: the fused compiler layout can otherwise change F32 rounding.
const QUANT_COLUMNS: usize = NVFP4_QUANT_CODES_TILE[1] * 2;

/// Bound the fused register tile to the measured normalization reduction range.
const MAX_PADDED_COLUMNS: usize = 8192;

pub const fn eligible_columns(columns: usize) -> bool {
    columns > super::norm_lowering::NORM_ROWS_MAX_DIMENSION
        && columns <= MAX_PADDED_COLUMNS
        && columns.is_multiple_of(QUANT_COLUMNS)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Binding {
    pub pointer: u64,
    pub rows: usize,
    pub columns: usize,
    pub scale_bits: u32,
    pub codes: u64,
    pub scales: u64,
}
#[derive(Clone, Copy)]
pub(super) struct Pending {
    pub next: usize,
    pub binding: Binding,
}

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
    static CONSUMPTIONS: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub(super) fn separate() -> bool {
    DISABLED.get()
}
#[cfg(test)]
pub(super) fn record_capture() {
    CAPTURES.set(CAPTURES.get() + 1);
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}
#[cfg(test)]
pub fn consumptions() -> usize {
    CONSUMPTIONS.get()
}
#[cfg(test)]
pub(super) fn record_consumption(ready: bool) {
    if ready {
        CONSUMPTIONS.set(CONSUMPTIONS.get() + 1);
    }
}
#[cfg(test)]
pub fn with_separate<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let _restore = Restore(DISABLED.replace(true));
    run()
}
