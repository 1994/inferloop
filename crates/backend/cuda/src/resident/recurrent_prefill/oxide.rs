//! Whether a prompt recurrence takes the pinned SIMT build, which the kernel crate launches.
use crate::device::CudaDevice;
use crate::tuning::recurrent::exact::select as exact_select;
use crate::tuning::resident_gemm::ResidentTuningSession;
use infer_core::Result;
use infer_cuda_kernels::contract::{Recurrent, RecurrentShape};
#[cfg(test)]
use std::cell::Cell;

/// Prepare the SIMT build for `shape` when the target ships it and exact tuning admits it.
///
/// # Errors
/// Reports a failed tuning lookup or a build that does not load.
pub(super) fn select(
    device: &CudaDevice,
    recurrent: &dyn Recurrent,
    shape: RecurrentShape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<bool> {
    #[cfg(test)]
    if DISABLED.get() {
        return Ok(false);
    }
    let key = [shape.key_heads, shape.value_heads, shape.dim, shape.lanes];
    Ok(device.target().has_simt_artifacts()
        && recurrent.prepare_simt_delta(shape)?
        && exact_select(device, key, autotune, session))
}

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}
/// Count one SIMT recording, for tests that check which build a capture took.
#[cfg(test)]
pub(super) fn recorded() {
    CAPTURES.set(CAPTURES.get() + 1);
}

#[cfg(test)]
pub fn with_cutile<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let guard = Reset(DISABLED.replace(true));
    let result = f();
    drop(guard);
    result
}
