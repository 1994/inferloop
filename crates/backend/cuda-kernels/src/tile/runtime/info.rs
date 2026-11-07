//! Facts the driver reports about a device and itself, as host profiles and policies read them.
#![expect(
    unsafe_code,
    reason = "Read-only driver queries with live output pointers on a device the stream keeps alive"
)]
use super::Device;
use crate::tile::backend_error;
use cuda_core::sys::{
    CUdevice_attribute, CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE as CLOCK_RATE,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH as BUS_WIDTH,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as L2_CACHE,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE as MEMORY_CLOCK,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as MULTIPROCESSORS,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE as WARP_SIZE, CUresult,
    cuDeviceGetAttribute, cuDriverGetVersion, cuMemGetInfo_v2, cudaError_enum_CUDA_SUCCESS,
};
use infer_core::Result;
use std::sync::Arc;

/// A device attribute the host reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attribute {
    ComputeMajor,
    ComputeMinor,
    Multiprocessors,
    WarpSize,
    /// Peak SM clock, in kHz.
    ClockKhz,
    /// Peak memory clock, in kHz.
    MemoryClockKhz,
    MemoryBusWidthBits,
    L2CacheBytes,
}

impl Attribute {
    const fn driver(self) -> CUdevice_attribute {
        match self {
            Self::ComputeMajor => MAJOR,
            Self::ComputeMinor => MINOR,
            Self::Multiprocessors => MULTIPROCESSORS,
            Self::WarpSize => WARP_SIZE,
            Self::ClockKhz => CLOCK_RATE,
            Self::MemoryClockKhz => MEMORY_CLOCK,
            Self::MemoryBusWidthBits => BUS_WIDTH,
            Self::L2CacheBytes => L2_CACHE,
        }
    }
}

impl Device {
    /// `attribute` of this device.
    /// # Errors
    /// Returns the driver's failure, or a negative value.
    pub fn attribute(&self, attribute: Attribute) -> Result<u32> {
        let mut value = 0;
        // SAFETY: the stream keeps its device alive and `value` is a live output.
        let status = unsafe {
            cuDeviceGetAttribute(
                &raw mut value,
                attribute.driver(),
                self.stream().device().cu_device(),
            )
        };
        driver(status)?;
        u32::try_from(value).map_err(backend_error)
    }

    /// The device's marketing name.
    /// # Errors
    /// Returns the driver's failure to report it.
    pub fn name(&self) -> Result<String> {
        self.stream().device().name().map_err(backend_error)
    }

    /// Make this device's context current on the calling thread, for driver calls made outside
    /// the kernel families.
    /// # Errors
    /// Returns the driver's failure to bind it.
    pub fn bind_to_thread(&self) -> Result<()> {
        self.stream()
            .device()
            .bind_to_thread()
            .map_err(backend_error)
    }

    /// Wait until every stream of this device has finished its work.
    /// # Errors
    /// Returns the driver's failure to synchronize.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: the device is live while the stream holds it; this only waits for its work.
        unsafe { self.stream().device().synchronize() }.map_err(backend_error)
    }

    /// Whether `other` enqueues on the same stream as this device.
    #[must_use]
    pub fn shares_stream(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.stream(), other.stream())
    }

    /// Free and total device memory, in bytes, as the driver counts them.
    /// # Errors
    /// Returns the driver's failure to bind the context or report memory.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.stream()
            .device()
            .bind_to_thread()
            .map_err(backend_error)?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: the device's context is current on this thread and both outputs are live.
        driver(unsafe { cuMemGetInfo_v2(&raw mut free, &raw mut total) })?;
        Ok((free as u64, total as u64))
    }
}

/// The installed driver's CUDA API version.
/// # Errors
/// Returns the driver's failure to report it.
pub fn driver_version() -> Result<u32> {
    let mut version = 0;
    // SAFETY: `version` is a live output; the query needs no current context.
    driver(unsafe { cuDriverGetVersion(&raw mut version) })?;
    u32::try_from(version).map_err(backend_error)
}

fn driver(status: CUresult) -> Result<()> {
    if status == cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(backend_error(format!("CUDA driver status {status}")))
    }
}
