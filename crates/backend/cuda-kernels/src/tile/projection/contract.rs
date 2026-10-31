//! cuTile implementation of the projection contract.
//!
//! The host owns every allocation; this module wraps its device pointers in borrowed cuTile
//! tensors for exactly one launch and synchronizes the shared stream before returning.
#![expect(
    unsafe_code,
    reason = "Wraps host-owned device pointers in borrowed cuTile tensors for exactly one launch"
)]

use super::linear;
use crate::contract::kernels::Projection;
use crate::contract::plan::LinearPlan;
use crate::contract::ptr::DevicePtr;
use cuda_core::{DType, Device, Stream};
use cutile::core::{ElementType, bf16, f8e4m3fn};
use cutile::prelude::*;
use infer_core::{Error, ErrorCode, Result};
use std::sync::Arc;

/// Projection kernels compiled with cuTile.
pub struct TileProjection {
    device: Arc<Device>,
    stream: Arc<Stream>,
}

impl TileProjection {
    /// Build the implementation for one device and its launch stream.
    #[must_use]
    pub const fn new(device: Arc<Device>, stream: Arc<Stream>) -> Self {
        Self { device, stream }
    }

    /// Build the implementation as a shared contract object.
    #[must_use]
    pub fn shared(device: Arc<Device>, stream: Arc<Stream>) -> Arc<dyn Projection> {
        Arc::new(Self::new(device, stream))
    }

    /// # Safety
    /// The caller upholds the `Projection` address contract.
    unsafe fn launch<E: ElementType + DType>(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
        dtype: &str,
    ) -> Result<()> {
        let rows = i32::try_from(plan.rows).map_err(|_| geometry("rows"))?;
        let columns = i32::try_from(plan.columns).map_err(|_| geometry("columns"))?;
        let ordinal = self.device.ordinal();
        // SAFETY: the caller guarantees a live allocation of the planned shape; the borrowed
        // tensors are dropped after the stream synchronizes below.
        let out_tensor =
            unsafe { Tensor::<f32>::borrow_raw_parts(out.0, ordinal, vec![rows], vec![1]) };
        // SAFETY: same contract as `out_tensor`; `x` is a contiguous activation vector.
        let x_tensor =
            unsafe { Tensor::<f32>::borrow_raw_parts(x.0, ordinal, vec![columns], vec![1]) };
        // SAFETY: same contract as `out_tensor`; `w` is a contiguous row-major weight matrix.
        let w_tensor = unsafe {
            Tensor::<E>::borrow_raw_parts(w.0, ordinal, vec![rows, columns], vec![columns, 1])
        };
        let out_partitioned = out_tensor.partition([plan.tile_rows]);
        linear::dense(out_partitioned, x_tensor, w_tensor)
            .generics(vec![
                dtype.into(),
                plan.tile_rows.to_string(),
                plan.block.to_string(),
                plan.columns.to_string(),
            ])
            .first()
            .unpartition()
            .sync_on(&self.stream)
            .map_err(|error| Error::new(ErrorCode::Backend, error.to_string()))?;
        Ok(())
    }
}

impl Projection for TileProjection {
    unsafe fn dense_bf16(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()> {
        // SAFETY: the caller upholds the `Projection` address contract.
        unsafe { self.launch::<bf16>(plan, out, x, w, "bf16") }
    }

    unsafe fn dense_f32(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()> {
        // SAFETY: the caller upholds the `Projection` address contract.
        unsafe { self.launch::<f32>(plan, out, x, w, "f32") }
    }

    unsafe fn dense_fp8(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()> {
        // SAFETY: the caller upholds the `Projection` address contract.
        unsafe { self.launch::<f8e4m3fn>(plan, out, x, w, "f8e4m3fn") }
    }
}

fn geometry(field: &str) -> Error {
    Error::invalid(format!("projection {field} exceeds i32"))
}
