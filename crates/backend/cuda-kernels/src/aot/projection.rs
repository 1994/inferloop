//! AOT implementation of the projection contract.
//!
//! The provider loads one artifact set, verifies every digest, and keeps the module alive for the
//! process. Only the f32 entry ships today; bf16/fp8 return `Unsupported` instead of silently
//! falling back to another profile.
#![expect(
    unsafe_code,
    reason = "Implements the unsafe launch contract and forwards its obligations to the driver"
)]

use super::artifact::Manifest;
use super::driver::{Device, Function};
use crate::contract::kernels::Projection;
use crate::contract::plan::LinearPlan;
use crate::contract::ptr::DevicePtr;
use crate::contract::target::Target;
use infer_core::{Error, ErrorCode, Result};
use std::ffi::{CStr, c_void};
use std::fs;
use std::path::Path;

/// Manifest name of the only entry this implementation launches.
const ENTRY_NAME: &str = "projection_dense_f32";
/// Device symbol of [`ENTRY_NAME`].
const ENTRY: &CStr = c"projection_dense_f32";
/// One warp per output row.
const WARP_THREADS: u32 = 32;

/// Projection kernels loaded from an AOT artifact set.
pub struct AotProjection {
    function: Function,
}

impl AotProjection {
    /// Load the `projection_dense_f32` entry from `root` for `target`.
    /// # Errors
    /// Fails when the manifest is missing or invalid, the target does not match, a digest differs,
    /// or the cubin/module/function cannot be loaded.
    pub fn load(device: &Device, root: &Path, target: Target) -> Result<Self> {
        let bytes = fs::read(root.join("manifest.json")).map_err(|error| {
            Error::new(
                ErrorCode::Backend,
                format!("AOT manifest {}: {error}", root.display()),
            )
        })?;
        let manifest = Manifest::parse(&bytes)?;
        manifest.select(target)?;
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
    unsafe fn dense_bf16(
        &self,
        _plan: &LinearPlan,
        _out: DevicePtr,
        _x: DevicePtr,
        _w: DevicePtr,
    ) -> Result<()> {
        Err(Error::unsupported("AOT projection provides f32 only"))
    }

    unsafe fn dense_f32(
        &self,
        plan: &LinearPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()> {
        let rows =
            u32::try_from(plan.rows).map_err(|_| Error::invalid("projection rows exceed u32"))?;
        let columns = i32::try_from(plan.columns)
            .map_err(|_| Error::invalid("projection columns exceed i32"))?;
        let mut x_pointer = x.0;
        let mut w_pointer = w.0;
        let mut out_pointer = out.0;
        let mut columns_argument = columns;
        let mut parameters = [
            (&raw mut x_pointer).cast::<c_void>(),
            (&raw mut w_pointer).cast::<c_void>(),
            (&raw mut out_pointer).cast::<c_void>(),
            (&raw mut columns_argument).cast::<c_void>(),
        ];
        // SAFETY: the four parameters match `projection_dense_f32` and live through this call; the
        // caller upholds the device-address contract of `Projection`.
        unsafe {
            self.function
                .launch((rows, 1, 1), (WARP_THREADS, 1, 1), &mut parameters)
        }
    }

    unsafe fn dense_fp8(
        &self,
        _plan: &LinearPlan,
        _out: DevicePtr,
        _x: DevicePtr,
        _w: DevicePtr,
    ) -> Result<()> {
        Err(Error::unsupported("AOT projection provides f32 only"))
    }
}
