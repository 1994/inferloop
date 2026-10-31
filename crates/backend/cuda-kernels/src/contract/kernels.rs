//! Operator-family traits implemented by each kernel profile.
#![expect(
    unsafe_code,
    reason = "Launch methods take raw device addresses until the host owns contract buffers"
)]

use crate::contract::plan::LinearPlan;
use crate::contract::ptr::DevicePtr;
use infer_core::Result;

/// Row-major projection kernels.
///
/// Methods are split by weight dtype instead of a generic parameter so the trait stays
/// object-safe and unsupported dtypes are rejected before a launch.
///
/// A launch is ordered on the implementation's stream. It may return before the work completes, so
/// the only completion guarantee is the next synchronization of that stream.
///
/// # Safety
/// Every method requires that `out`, `x` and `w` address live, contiguous, row-major allocations
/// of the planned shape on this device, that `out` does not alias an input, and that all three stay
/// valid with no conflicting access until the stream is synchronized.
pub trait Projection: Send + Sync {
    /// # Safety
    /// See the trait contract.
    /// # Errors
    /// Returns a backend error when the geometry is invalid or the launch fails.
    unsafe fn dense_bf16(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()>;
    /// # Safety
    /// See the trait contract.
    /// # Errors
    /// Returns a backend error when the geometry is invalid or the launch fails.
    unsafe fn dense_f32(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()>;
    /// # Safety
    /// See the trait contract.
    /// # Errors
    /// Returns a backend error when the geometry is invalid or the launch fails.
    unsafe fn dense_fp8(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()>;
}
