//! Small batched projections using tensor cores with a two-term BF16 input.
//! Retaining the input residual avoids reducing F32 activations to one BF16 value.
#[cfg(test)]
pub use infer_cuda_kernels::tile::projection::small_gemm as kernels;
#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
#[path = "../../tests/unit/resident_small_gemm.rs"]
mod tests;

/// Internal A/B selection; failed candidates cannot be enabled in a production build.
#[cfg(test)]
pub(super) fn precise_dense_enabled() -> bool {
    !pool_f32_enabled() && PRECISE_DENSE.with(Cell::get)
}

#[cfg(test)]
thread_local! {
    static PRECISE_DENSE: Cell<bool> = const { Cell::new(false) };
}
#[cfg(test)]
pub fn with_precise_dense<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRECISE_DENSE.with(|cell| cell.set(self.0));
        }
    }
    let restore = Restore(PRECISE_DENSE.with(|cell| cell.replace(enabled)));
    let result = run();
    drop(restore);
    result
}

/// Internal four-slot experiment; never dispatched by a production build.
#[cfg(test)]
pub(super) fn pool_f32_enabled() -> bool {
    POOL_F32.with(Cell::get)
}
#[cfg(test)]
thread_local! { static POOL_F32: Cell<bool> = const { Cell::new(false) }; }
#[cfg(test)]
pub fn with_pool_f32<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            POOL_F32.with(|cell| cell.set(self.0));
        }
    }
    let restore = Restore(POOL_F32.with(|cell| cell.replace(enabled)));
    let result = run();
    drop(restore);
    result
}

#[cfg(test)]
thread_local! { static POOL_RECORDS: Cell<usize> = const { Cell::new(0) }; }
#[cfg(test)]
pub(super) fn note_pool_record() {
    POOL_RECORDS.with(|count| count.set(count.get().saturating_add(1)));
}
#[cfg(test)]
pub fn pool_f32_capture_records() -> usize {
    POOL_RECORDS.with(Cell::get)
}
