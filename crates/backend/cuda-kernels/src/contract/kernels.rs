//! Operator-family traits implemented by each kernel profile.

use crate::contract::plan::LinearPlan;
use crate::contract::ptr::DevicePtr;
use infer_core::Result;

/// Row-major projection kernels.
///
/// Methods are split by weight dtype instead of a generic parameter so the trait stays
/// object-safe and unsupported dtypes are rejected before a launch.
pub trait Projection: Send + Sync {
    /// # Errors
    /// Returns a backend error when the geometry is invalid or the launch fails.
    fn dense_bf16(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()>;
    /// # Errors
    /// Returns a backend error when the geometry is invalid or the launch fails.
    fn dense_f32(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()>;
    /// # Errors
    /// Returns a backend error when the geometry is invalid or the launch fails.
    fn dense_fp8(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()>;
}
