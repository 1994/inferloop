//! AOT implementation of the projection family; kernels live in `projection/*.cu`.
#![expect(
    unsafe_code,
    reason = "Implements the unsafe launch contract and forwards its obligations to the driver"
)]

use super::artifact::Manifest;
use super::driver::{Device, Function};
use crate::contract::kernels::Projection;
use crate::contract::plan::PoolGemvPlan;
use crate::contract::ptr::DevicePtr;
use crate::contract::recording::Recording;
use infer_core::{Error, ErrorCode, Result};
use std::ffi::{CStr, c_void};
use std::fs;
use std::path::Path;

/// Manifest name of the only entry this implementation launches.
const ENTRY_NAME: &str = "pool_gemv_bf16";
/// Device symbol of [`ENTRY_NAME`].
const ENTRY: &CStr = c"pool_gemv_bf16";
/// One warp per output element.
const WARP_THREADS: u32 = 32;

/// Projection kernels loaded from an AOT artifact set.
pub struct AotProjection {
    function: Function,
}

impl AotProjection {
    /// Load the projection entry from `root`, for the architecture `device` reports.
    /// # Errors
    /// Fails when the manifest is missing, invalid or built for another architecture, a digest
    /// differs, or the cubin/module/function cannot be loaded.
    pub fn load(device: &Device, root: &Path) -> Result<Self> {
        let bytes = fs::read(root.join("manifest.json")).map_err(|error| {
            Error::new(
                ErrorCode::Backend,
                format!("AOT manifest {}: {error}", root.display()),
            )
        })?;
        let manifest = Manifest::parse(&bytes)?;
        manifest.select(device.arch()?)?;
        manifest.verify(root)?;
        let entry = manifest
            .entry(ENTRY_NAME)
            .ok_or_else(|| Error::unsupported("AOT projection entry missing"))?;
        let cubin = fs::read(root.join(&entry.cubin)).map_err(|error| {
            Error::new(
                ErrorCode::Backend,
                format!("AOT cubin {}: {error}", entry.cubin),
            )
        })?;
        let module = device.load_module(&cubin)?;
        Ok(Self {
            function: module.function(ENTRY)?,
        })
    }
}

impl Projection for AotProjection {
    unsafe fn pool_gemv_bf16(
        &self,
        recording: &Recording<'_>,
        plan: &PoolGemvPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()> {
        let rows = u32::try_from(plan.rows).map_err(|_| geometry("rows"))?;
        let elements = u32::try_from(plan.outputs).map_err(|_| geometry("outputs"))?;
        let mut outputs = i32::try_from(plan.outputs).map_err(|_| geometry("outputs"))?;
        let mut columns = i32::try_from(plan.columns).map_err(|_| geometry("columns"))?;
        let (mut x_pointer, mut w_pointer, mut out_pointer) = (x.0, w.0, out.0);
        let mut parameters = [
            (&raw mut x_pointer).cast::<c_void>(),
            (&raw mut w_pointer).cast::<c_void>(),
            (&raw mut out_pointer).cast::<c_void>(),
            (&raw mut columns).cast::<c_void>(),
            (&raw mut outputs).cast::<c_void>(),
        ];
        // SAFETY: the five parameters match `pool_gemv_bf16` and live through this call; the
        // caller upholds the device-address contract of `Projection`.
        unsafe {
            self.function.launch(
                recording,
                (elements, rows, 1),
                (WARP_THREADS, 1, 1),
                &mut parameters,
            )
        }
    }
}

fn geometry(field: &str) -> Error {
    Error::invalid(format!("projection {field} exceeds i32"))
}
