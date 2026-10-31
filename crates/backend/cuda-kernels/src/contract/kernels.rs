//! Operator-family traits implemented by each kernel source.
#![expect(
    unsafe_code,
    reason = "Launch methods take raw device addresses until the host owns contract buffers"
)]

use crate::contract::plan::PoolGemvPlan;
use crate::contract::ptr::DevicePtr;
use crate::contract::recording::Recording;
use infer_core::Result;

/// Row-major projection kernels.
///
/// Methods are split by weight dtype instead of a generic parameter so the trait stays
/// object-safe and unsupported dtypes are rejected before a launch.
pub trait Projection: Send + Sync {
    /// Record one BF16 row-independent projection; no work completes before the graph runs.
    ///
    /// # Safety
    /// `out`, `x` and `w` must address contiguous row-major allocations of the planned shapes on
    /// the recording's device: `[rows, outputs]`, `[rows, columns]` and `[outputs, columns]`.
    /// `out` must not alias an input, and all three must stay valid until every replay has finished.
    /// # Errors
    /// Fails when the recording form does not belong to this implementation, or the launch is
    /// rejected.
    unsafe fn pool_gemv_bf16(
        &self,
        recording: &Recording<'_>,
        plan: &PoolGemvPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()>;
}
