//! Safe CUDA memory and kernel launch boundary.
use crate::constants::KV_CACHE_RANK;
use crate::constants::MATVEC_TILE_COLUMNS;
use crate::constants::MATVEC_TILE_ROWS;
use crate::kernels::linear::dense;
use crate::strategy::LinearTiling;
use crate::target::CudaTarget;
use crate::tuning::memory::arena_budget;
use cuda_core::sys::CUdevice_attribute;
use cuda_core::sys::CUresult;
use cuda_core::sys::cuDeviceGetAttribute;
use cuda_core::sys::cuDriverGetVersion;
use cuda_core::sys::cuMemGetInfo_v2;
use cuda_core::sys::cuMemcpyDtoDAsync_v2;
use cuda_core::sys::cuMemcpyDtoH_v2;
use cuda_core::sys::cuMemcpyHtoDAsync_v2;
use cuda_core::sys::cudaError_enum_CUDA_SUCCESS;
use cuda_core::sys::{CUDA_MEMCPY3D, CUmemorytype_enum_CU_MEMORYTYPE_DEVICE, cuMemcpy3DAsync_v2};
use cuda_core::sys::{
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE as CORE_CLOCK,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as COMPUTE_MAJOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as COMPUTE_MINOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as MULTIPROCESSORS,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE as WARP_SIZE,
};
use cuda_core::{DeviceCopy, PinnedHostBuffer};
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::jit_cache::enable_default;
use cutile::prelude::*;
use infer_core::{Error as InferError, ErrorCode, Result};
use serde::Serialize;
use std::fmt::Display;
use std::ptr::{null, null_mut};
use std::result::Result as StdResult;
use std::sync::{Arc, OnceLock};

/// One MiB in bytes, for human-readable identities.
const MIB: u64 = 1024 * 1024;
/// Device memory reserved for driver and module bookkeeping, as a fraction of total memory.
const HEADROOM_DIVISOR: u64 = 32;
/// Lower bound of that reservation, for small devices.
const MIN_DEVICE_HEADROOM: u64 = 256 * MIB;
/// Upper bound of that reservation, so huge devices keep state space usable.
const MAX_DEVICE_HEADROOM: u64 = 1024 * MIB;
/// Hertz per kilohertz, for the driver's kHz clock attributes.
const HERTZ_PER_KILOHERTZ: u64 = 1000;
/// Bits per byte, for the driver's bit-width attributes.
const BITS_PER_BYTE: u32 = 8;
/// Double data rate: two transfers per memory clock.
const TRANSFERS_PER_CLOCK: u64 = 2;

/// Hardware facts queried from the driver at open time, plus the policy values derived from them.
///
/// Nothing here is maintained per GPU model: every field is read from the device, so a new card
/// needs no code or table change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceProfile {
    /// Driver-reported device name.
    pub name: String,
    /// Compute capability as `sm_<major><minor>`.
    pub architecture: String,
    /// Streaming multiprocessors.
    pub multiprocessors: u32,
    /// Total device memory.
    pub total_memory_bytes: u64,
    /// Memory clock in kHz.
    pub memory_clock_khz: u32,
    /// Global memory bus width in bits.
    pub memory_bus_width_bits: u32,
    /// L2 cache size in bytes.
    pub l2_cache_bytes: u32,
}

/// Driver-reported SM capability, separate from any kernel or artifact qualification policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComputeCapability {
    pub(crate) major: u32,
    pub(crate) minor: u32,
}

impl ComputeCapability {
    /// Tensor-core generation for architectures qualified by this backend.
    pub(crate) const fn tensor_core_generation(self) -> Option<u16> {
        /// Compute capability major for the qualified fourth-generation tensor cores.
        const HOPPER_SM_MAJOR: u32 = 9;
        /// Compute capability major for the qualified fifth-generation tensor cores.
        const BLACKWELL_SM_MAJOR: u32 = 12;
        /// Tensor-core generation reported for the qualified Hopper target.
        const HOPPER_TENSOR_CORE_GENERATION: u16 = 4;
        /// Tensor-core generation reported for the qualified Blackwell target.
        const BLACKWELL_TENSOR_CORE_GENERATION: u16 = 5;
        match (self.major, self.minor) {
            (HOPPER_SM_MAJOR, 0) => Some(HOPPER_TENSOR_CORE_GENERATION),
            (BLACKWELL_SM_MAJOR, 0) => Some(BLACKWELL_TENSOR_CORE_GENERATION),
            _ => None,
        }
    }
}

fn compute_capability_with(
    mut query: impl FnMut(CUdevice_attribute) -> Result<u32>,
) -> Result<ComputeCapability> {
    Ok(ComputeCapability {
        major: query(COMPUTE_MAJOR)?,
        minor: query(COMPUTE_MINOR)?,
    })
}
impl DeviceProfile {
    /// Stable identity for machine-local artifacts such as the kernel tuning table. Two
    /// different devices cannot share name, architecture and memory size at once.
    #[must_use]
    pub fn identity(&self) -> String {
        format!(
            "{}|{}|{}MiB",
            self.name,
            self.architecture,
            self.total_memory_bytes / MIB
        )
    }
    /// Theoretical DRAM bandwidth, the roofline denominator for bandwidth-bound decode steps.
    #[must_use]
    pub fn memory_bandwidth_bytes_per_second(&self) -> u64 {
        memory_bandwidth(self.memory_clock_khz, self.memory_bus_width_bits)
    }
    /// Driver and module slack kept out of the request-state budget.
    #[must_use]
    pub fn device_headroom_bytes(&self) -> u64 {
        device_headroom(self.total_memory_bytes)
    }
    /// Graph capture and allocator slack counted inside a sequence budget.
    #[must_use]
    pub fn graph_headroom_bytes(&self) -> u64 {
        graph_headroom(self.total_memory_bytes)
    }
    /// Per-request activation arena and projection workspace budget.
    #[must_use]
    pub fn arena_budget_bytes(&self) -> u64 {
        arena_budget(self.total_memory_bytes)
    }
    /// Verification checkpoint budget retained across speculation lanes.
    #[must_use]
    pub fn checkpoint_budget_bytes(&self) -> u64 {
        checkpoint_budget(self.total_memory_bytes)
    }
    /// Resident sequences this device is expected to admit before the byte budget rejects more.
    #[must_use]
    pub fn resident_states(&self) -> usize {
        resident_states(self.total_memory_bytes)
    }
}

/// Double-data-rate bandwidth from the memory clock and bus width.
fn memory_bandwidth(clock_khz: u32, bus_width_bits: u32) -> u64 {
    u64::from(clock_khz)
        * HERTZ_PER_KILOHERTZ
        * u64::from(bus_width_bits / BITS_PER_BYTE)
        * TRANSFERS_PER_CLOCK
}

/// Fractional headroom clamped so small devices keep a floor and large ones stay bounded.
fn device_headroom(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / HEADROOM_DIVISOR).clamp(MIN_DEVICE_HEADROOM, MAX_DEVICE_HEADROOM)
}

/// Graph and allocator slack reserved inside every sequence budget.
const GRAPH_HEADROOM_DIVISOR: u64 = 128;
/// Lower bound of that reservation.
const MIN_GRAPH_HEADROOM: u64 = 64 * MIB;
/// Upper bound of that reservation.
const MAX_GRAPH_HEADROOM: u64 = 512 * MIB;
/// Retained verification checkpoints across all speculation lanes.
const CHECKPOINT_DIVISOR: u64 = 16;
/// Lower bound of the checkpoint budget.
const MIN_CHECKPOINT_BUDGET: u64 = 512 * MIB;
/// Upper bound of the checkpoint budget.
const MAX_CHECKPOINT_BUDGET: u64 = 4 * 1024 * MIB;
/// Device memory charged per reserved sequence when bounding concurrency.
const STATE_BYTES_PER_SLOT: u64 = 512 * MIB;
/// Concurrency bounds applied after dividing device memory by the per-slot charge.
const MIN_RESIDENT_STATES: usize = 4;
/// Upper bound on resident sequences regardless of device size.
const MAX_RESIDENT_STATES: usize = 64;

/// Fraction of device memory reserved for graph capture and allocator bookkeeping.
fn graph_headroom(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / GRAPH_HEADROOM_DIVISOR).clamp(MIN_GRAPH_HEADROOM, MAX_GRAPH_HEADROOM)
}

/// Retained verification checkpoint budget.
fn checkpoint_budget(total_memory_bytes: u64) -> u64 {
    (total_memory_bytes / CHECKPOINT_DIVISOR).clamp(MIN_CHECKPOINT_BUDGET, MAX_CHECKPOINT_BUDGET)
}

/// Resident state slots a device of this size can be expected to admit.
fn resident_states(total_memory_bytes: u64) -> usize {
    usize::try_from(total_memory_bytes / STATE_BYTES_PER_SLOT)
        .unwrap_or(MAX_RESIDENT_STATES)
        .clamp(MIN_RESIDENT_STATES, MAX_RESIDENT_STATES)
}
pub(crate) mod asynchronous;
pub(crate) mod library_names;
#[cfg(test)]
mod nvjpeg_probe;
pub(crate) mod platform;
mod readback;
mod upload;
#[cfg(test)]
pub(crate) use readback::queue_test_gate;
pub(crate) use upload::PinnedUpload;
// Exercised by its own device test today; the projection path adopts it in a following change.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "adopted by the projection path in a following change"
    )
)]
pub(crate) mod cublaslt;
pub(crate) mod vendor_fp8;
pub(crate) use readback::{ReadbackSource, Readbacks};

#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) stream: Arc<cuda_core::Stream>,
    target: CudaTarget,
    profile: Arc<OnceLock<DeviceProfile>>,
    ordinal: usize,
}

pub(crate) fn device_error(error: impl Display) -> InferError {
    InferError::new(ErrorCode::Backend, error.to_string())
}

impl CudaDevice {
    /// Enable cuTile's persistent compiled-kernel cache for subsequent captures.
    /// # Errors
    /// Returns cache initialization errors.
    pub fn enable_kernel_cache() -> Result<()> {
        enable_default().map_err(device_error)
    }
    /// # Errors
    /// Fails when the requested CUDA device or stream is unavailable.
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = Device::new(ordinal).map_err(device_error)?;
        let target = query_target(&device)?;
        Ok(Self {
            stream: device.new_stream().map_err(device_error)?,
            target,
            profile: Arc::new(OnceLock::new()),
            ordinal,
        })
    }

    /// Ordinal this instance was created for, which identifies the device a handle belongs to.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// # Errors
    /// Returns a CUDA error when the device name cannot be queried.
    pub fn name(&self) -> Result<String> {
        self.stream.device().name().map_err(device_error)
    }

    #[must_use]
    pub const fn target(&self) -> &CudaTarget {
        &self.target
    }

    /// # Errors
    /// Returns a CUDA error if upload or reshape fails.
    pub fn upload<T: DType>(&self, values: Vec<T>, shape: &[usize]) -> Result<Arc<Tensor<T>>> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(&self.stream)
            .map_err(device_error)?
            .reshape(shape)
            .map(Arc::new)
            .map_err(device_error)
    }

    /// # Errors
    /// Returns a shape or device error for an invalid matrix-vector product.
    pub fn matvec<T: DType>(
        &self,
        x: Arc<Tensor<f32>>,
        w: Arc<Tensor<T>>,
    ) -> Result<Arc<Tensor<f32>>> {
        self.matvec_tiled(
            x,
            w,
            LinearTiling::new(MATVEC_TILE_ROWS, MATVEC_TILE_COLUMNS)?,
        )
    }

    /// # Errors
    /// Returns a shape or device error for an invalid projection.
    pub fn matvec_tiled<T: DType>(
        &self,
        x: Arc<Tensor<f32>>,
        w: Arc<Tensor<T>>,
        tiling: LinearTiling,
    ) -> Result<Arc<Tensor<f32>>> {
        let (rows, columns) = dimensions(&x, &w)?;
        let block = columns.min(tiling.columns()).next_power_of_two();
        let out = api::zeros::<f32>(&[rows]).partition([tiling.rows()]);
        dense(out, x, w)
            .generics(vec![
                T::DTYPE.as_str().into(),
                tiling.rows().to_string(),
                block.to_string(),
                columns.to_string(),
            ])
            .first()
            .unpartition()
            .sync_on(&self.stream)
            .map(Arc::new)
            .map_err(device_error)
    }

    /// # Errors
    /// Returns a CUDA transfer error.
    pub fn read<T: DType>(&self, tensor: &Arc<Tensor<T>>) -> Result<Vec<T>> {
        tensor
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(device_error)
    }

    #[expect(
        unsafe_code,
        reason = "Read-only shared storage alias held across the device future; the immutable source borrow excludes host mutation"
    )]
    pub(crate) async fn read_async<T: DType + DeviceCopy>(
        &self,
        tensor: &Tensor<T>,
        readbacks: &mut Readbacks<T>,
    ) -> Result<Vec<T>> {
        // SAFETY: the immutable borrow lasts through the read; the alias owns shared storage,
        // registered device leases retain it until the future fences the ordered D2H.
        let source = unsafe { tensor.into_shared_alias() };
        readbacks.read_async(self, source).await
    }

    /// Enqueue a host slice into an existing device tensor on the device stream.
    ///
    /// The driver stages pageable host memory before returning, so the caller may drop `values`
    /// immediately; the device-side write stays ordered before later work on the same stream.
    ///
    /// # Errors
    /// Rejects a slice longer than the tensor or a driver failure when the copy cannot be enqueued.
    #[expect(
        unsafe_code,
        reason = "Audited host-to-device copy from a live slice into a live tensor of known length"
    )]
    #[cfg(test)]
    pub(crate) fn copy_host_slice<T: DType>(
        &self,
        dst: &mut Tensor<T>,
        values: &[T],
    ) -> Result<()> {
        if values.len() > dst.size() {
            return Err(InferError::invalid("host copy range outside tensor"));
        }
        let bytes = values
            .len()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| InferError::invalid("host copy size overflow"))?;
        let stream = self.stream.cu_stream();
        // SAFETY: the checked length keeps the write inside the tensor, the source slice is live
        // for the duration of the call, and the stream belongs to this device.
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(
                dst.device_pointer().cu_deviceptr(),
                values.as_ptr().cast(),
                bytes,
                stream,
            )
        };
        driver_status(status)
    }

    /// Copy a resident tensor back without taking ownership, so a borrowed snapshot can be inspected.
    ///
    /// # Errors
    /// Returns a backend error when the device is unbound or the copy fails.
    #[expect(
        unsafe_code,
        reason = "Audited device-to-host copy from a live tensor of known length into a live host slice"
    )]
    pub fn read_borrowed<T: DType>(&self, tensor: &Tensor<T>) -> Result<Vec<T>> {
        self.reclaim_barrier()?;
        let mut host = vec![T::zero(); tensor.size()];
        let bytes = tensor
            .size()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| device_error("readback size overflow"))?;
        // SAFETY: the device context is current, the tensor owns `size()` T elements and the
        // destination is a live host slice of the same length.
        let status = unsafe {
            cuMemcpyDtoH_v2(
                host.as_mut_ptr().cast(),
                tensor.device_pointer().cu_deviceptr(),
                bytes,
            )
        };
        driver_status(status)?;
        Ok(host)
    }
}

#[expect(
    unsafe_code,
    reason = "Audited read-only CUDA attribute query; the live owned Device supplies a valid driver handle"
)]
fn query_target(device: &Device) -> Result<CudaTarget> {
    // SAFETY: Device::new initialized CUDA and owns this live device handle.
    let name =
        unsafe { cuda_core::get_device_sm_name(device.cu_device()) }.map_err(device_error)?;
    Ok(CudaTarget::from_sm_name(&name))
}

fn head_copy_shape<T: DType>(tensor: &Tensor<T>) -> Result<[usize; KV_CACHE_RANK]> {
    let [heads, capacity, dim] = tensor.shape() else {
        return Err(InferError::invalid("head copy requires rank-three tensors"));
    };
    let [heads, capacity, dim] =
        [*heads, *capacity, *dim].map(|n| usize::try_from(n).map_err(device_error));
    let shape = [heads?, capacity?, dim?];
    let size = shape
        .iter()
        .try_fold(1usize, |count, &n| count.checked_mul(n))
        .ok_or_else(|| InferError::invalid("head copy shape overflow"))?;
    if size != tensor.size() {
        return Err(InferError::invalid("head copy tensor extent"));
    }
    size.checked_mul(size_of::<T>())
        .ok_or_else(|| InferError::invalid("head copy byte overflow"))?;
    Ok(shape)
}

fn dimensions<T: DType>(x: &Tensor<f32>, w: &Tensor<T>) -> Result<(usize, usize)> {
    if w.shape().len() != 2 || x.shape().len() != 1 || x.shape()[0] != w.shape()[1] {
        return Err(InferError::invalid("CUDA matrix-vector dimensions"));
    }
    let rows = usize::try_from(w.shape()[0]).map_err(|_| InferError::invalid("CUDA rows"))?;
    let columns = usize::try_from(w.shape()[1]).map_err(|_| InferError::invalid("CUDA columns"))?;
    if rows == 0 || columns == 0 {
        return Err(InferError::invalid("empty CUDA matrix"));
    }
    Ok((rows, columns))
}

#[expect(
    unsafe_code,
    reason = "Audited driver queries and stream drain with an owned device and a bound context"
)]
impl CudaDevice {
    fn attribute(&self, attribute: CUdevice_attribute) -> Result<u32> {
        let mut value = 0;
        // SAFETY: the owned stream retains this device; value is a live output pointer.
        let status = unsafe {
            cuDeviceGetAttribute(&raw mut value, attribute, self.stream.device().cu_device())
        };
        driver_status(status)?;
        u32::try_from(value).map_err(device_error)
    }

    pub(crate) fn compute_capability(&self) -> Result<ComputeCapability> {
        compute_capability_with(|attribute| self.attribute(attribute))
    }

    pub(crate) fn multiprocessor_count(&self) -> Result<u32> {
        self.attribute(MULTIPROCESSORS)
    }

    pub(crate) fn warp_size(&self) -> Result<u32> {
        self.attribute(WARP_SIZE)
    }

    pub(crate) fn core_clock_khz(&self) -> Result<u32> {
        self.attribute(CORE_CLOCK)
    }

    /// CUDA driver API version, used to invalidate measured kernel decisions.
    /// # Errors
    /// Returns a driver error when the installed driver cannot report its version.
    pub(crate) fn driver_version() -> Result<u32> {
        let mut version = 0;
        // SAFETY: version is a live output pointer; this query requires no current context.
        let status = unsafe { cuDriverGetVersion(&raw mut version) };
        driver_status(status)?;
        u32::try_from(version).map_err(device_error)
    }

    /// Cached hardware facts and derived policy for this device.
    /// # Errors
    /// Returns a driver error if any device attribute or memory size cannot be queried.
    pub fn profile(&self) -> Result<&DeviceProfile> {
        if let Some(profile) = self.profile.get() {
            return Ok(profile);
        }
        let queried = self.query_profile()?;
        let _ = self.profile.set(queried);
        self.profile
            .get()
            .ok_or_else(|| device_error("device profile cache"))
    }

    /// Query the driver for hardware facts and derived policy.
    /// # Errors
    /// Returns a driver error if any device attribute or memory size cannot be queried.
    fn query_profile(&self) -> Result<DeviceProfile> {
        use cuda_core::sys::{
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH as BUS,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as L2,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE as MEMORY_CLOCK,
            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as SMS,
        };
        Ok(DeviceProfile {
            name: self.name()?,
            architecture: format!("sm_{}{}", self.attribute(MAJOR)?, self.attribute(MINOR)?),
            multiprocessors: self.attribute(SMS)?,
            total_memory_bytes: self.memory_info()?.1,
            memory_clock_khz: self.attribute(MEMORY_CLOCK)?,
            memory_bus_width_bits: self.attribute(BUS)?,
            l2_cache_bytes: self.attribute(L2)?,
        })
    }

    /// Copy `elements` values between resident tensors of one dtype on this device's stream.
    ///
    /// Prefix caching snapshots and restores KV state with this: one stream-ordered device copy,
    /// no host round trip.
    /// # Errors
    /// Returns a driver error when the copy cannot be enqueued.
    pub fn copy_d2d<T: DType>(
        &self,
        dst: &mut Tensor<T>,
        src: &Tensor<T>,
        elements: usize,
    ) -> Result<()> {
        self.copy_d2d_at(dst, src, 0, 0, elements)
    }

    /// Copy the first `rows` positions of head-major `[heads, capacity, dim]` tensors.
    /// Owners retain both buffers through the next stream fence. Capacities may differ;
    /// the active prefix moves in one asynchronous driver submission without touching tails.
    /// # Errors
    /// Rejects incompatible layouts, ranges, devices, size overflow or driver failures.
    pub(crate) fn copy_head_prefix<T: DType>(
        &self,
        dst: &mut Tensor<T>,
        src: &Tensor<T>,
        rows: usize,
    ) -> Result<()> {
        let d = head_copy_shape(dst)?;
        let s = head_copy_shape(src)?;
        if d[0] != s[0]
            || d[2] != s[2]
            || rows > d[1]
            || rows > s[1]
            || src.device_id() != self.ordinal
            || dst.device_id() != self.ordinal
        {
            return Err(InferError::invalid("head prefix copy layout or range"));
        }
        if rows == 0 || s[0] == 0 || s[2] == 0 {
            return Ok(());
        }
        if dst.device_pointer().cu_deviceptr() == src.device_pointer().cu_deviceptr() {
            return if d == s {
                Ok(())
            } else {
                Err(InferError::invalid("aliased head copy layout"))
            };
        }
        if rows == s[1] && d == s {
            return self.copy_d2d(dst, src, src.size());
        }
        let pitch = s[2]
            .checked_mul(size_of::<T>())
            .ok_or_else(|| InferError::invalid("head copy pitch overflow"))?;
        let copy = CUDA_MEMCPY3D {
            srcXInBytes: 0,
            srcY: 0,
            srcZ: 0,
            srcLOD: 0,
            srcMemoryType: CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
            srcHost: null(),
            srcDevice: src.device_pointer().cu_deviceptr(),
            srcArray: null_mut(),
            reserved0: null_mut(),
            srcPitch: pitch,
            srcHeight: s[1],
            dstXInBytes: 0,
            dstY: 0,
            dstZ: 0,
            dstLOD: 0,
            dstMemoryType: CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
            dstHost: null_mut(),
            dstDevice: dst.device_pointer().cu_deviceptr(),
            dstArray: null_mut(),
            reserved1: null_mut(),
            dstPitch: pitch,
            dstHeight: d[1],
            WidthInBytes: pitch,
            Height: rows,
            Depth: s[0],
        };
        // SAFETY: checked shapes bound every [head, row, column] to the owned tensors.
        // The fully initialized descriptor uses typed, aligned device storage and NULL unused
        // host/array/reserved fields. Independent programs/snapshots own disjoint buffers;
        // identical aliases returned above. Callers keep both owners through the stream fence,
        // and the owned stream retains the current device context through this enqueue.
        let status = unsafe { cuMemcpy3DAsync_v2(&raw const copy, self.stream.cu_stream()) };
        driver_status(status)
    }

    /// Ranged variant: snapshots copy only the token rows a prefix actually covers.
    /// # Errors
    /// Rejects ranges outside either tensor or a driver failure when the copy cannot be enqueued.
    pub fn copy_d2d_at<T: DType>(
        &self,
        dst: &mut Tensor<T>,
        src: &Tensor<T>,
        dst_offset: usize,
        src_offset: usize,
        elements: usize,
    ) -> Result<()> {
        let fits = |tensor: &Tensor<T>, offset: usize| {
            offset
                .checked_add(elements)
                .is_some_and(|end| end <= tensor.size())
        };
        if !fits(dst, dst_offset) || !fits(src, src_offset) {
            return Err(InferError::invalid("device copy range outside tensor"));
        }
        let bytes = elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| InferError::invalid("device copy size overflow"))?;
        let offset_bytes = |offset: usize| {
            offset
                .checked_mul(size_of::<T>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| InferError::invalid("device copy offset overflow"))
        };
        let dst_bytes = offset_bytes(dst_offset)?;
        let src_bytes = offset_bytes(src_offset)?;
        let stream = self.stream.cu_stream();
        // SAFETY: the checked ranges keep both accesses inside live device allocations, and the
        // stream belongs to this device; callers drain it before reusing the buffers.
        let status = unsafe {
            cuMemcpyDtoDAsync_v2(
                dst.device_pointer().cu_deviceptr() + dst_bytes,
                src.device_pointer().cu_deviceptr() + src_bytes,
                bytes,
                stream,
            )
        };
        driver_status(status)
    }

    /// Async H2D copy out of a pinned staging buffer, enqueued on the device stream and
    /// never synchronized here: the caller keeps the pinned bytes untouched until a later
    /// stream barrier (a readback or the owning pool's drop barrier) establishes completion.
    /// # Errors
    /// Rejects copies outside either buffer or a driver failure when the copy cannot be enqueued.
    pub(crate) fn copy_h2d_pinned<T: DType + DeviceCopy>(
        &self,
        dst: &mut Tensor<T>,
        src: &PinnedHostBuffer<T>,
        elements: usize,
    ) -> Result<()> {
        if elements > dst.size() || elements > src.len() {
            return Err(InferError::invalid("pinned upload range outside buffer"));
        }
        let bytes = elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| InferError::invalid("pinned upload size overflow"))?;
        let stream = self.stream.cu_stream();
        // SAFETY: the checked lengths keep both ends inside live allocations, the pinned
        // source is not rewritten until the caller's next stream barrier, and the stream
        // belongs to this device.
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(
                dst.device_pointer().cu_deviceptr(),
                src.as_ptr().cast(),
                bytes,
                stream,
            )
        };
        driver_status(status)
    }

    /// # Errors
    /// Returns a driver error if available device memory cannot be queried.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: the owned device's context is current and both outputs are live.
        let status = unsafe { cuMemGetInfo_v2(&raw mut free, &raw mut total) };
        driver_status(status)?;
        Ok((free as u64, total as u64))
    }

    /// Bytes available to this backend, including unused blocks in its CUDA allocation pool.
    /// Driver free memory alone excludes pool reservations even after their tensors are freed.
    /// Live allocations remain charged; other CUDA contexts' pools are never counted.
    /// # Errors
    /// Returns a driver error if memory or allocation-pool accounting cannot be queried.
    pub fn allocatable_memory_bytes(&self) -> Result<u64> {
        let (free, total) = self.memory_info()?;
        Ok(free
            .saturating_add(self.pool_reclaimable_bytes()?)
            .min(total))
    }

    /// Device bytes the allocator can hand out again without asking the driver.
    ///
    /// Tensors are allocated with `cuMemAllocAsync` from the device's memory pool, and a freed
    /// block stays reserved in that pool. `cuMemGetInfo` counts reserved-but-unused blocks as
    /// used, so a pre-flight admission check against it under-reports what a retry can actually
    /// obtain; this is the difference.
    pub(crate) fn pool_reclaimable_bytes(&self) -> Result<u64> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        let pool = self
            .stream
            .device()
            .default_mem_pool()
            .map_err(device_error)?;
        let stats = pool.mem_stats().map_err(device_error)?;
        Ok(stats.reserved_current.saturating_sub(stats.used_current))
    }

    // cuda-async frees tensors on a separate deallocator stream. Join it before
    // using driver free-memory queries to admit allocations after cache eviction.
    pub(crate) fn reclaim_barrier(&self) -> Result<()> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        // Tensors come from the device memory pool, and a block returned to the pool keeps
        // counting as used in `cuMemGetInfo` until the driver trims it. Neither raising the
        // release threshold nor `cuMemPoolTrimTo` (unbound in cuda-bindings 0.4) gets the memory
        // back in time, so a failed slot-pool attempt is only visible through
        // `pool_reclaimable_bytes`; nothing here pretends to make the query honest.
        // SAFETY: the owned device's context is current; this only waits for its work.
        unsafe { self.stream.device().synchronize() }.map_err(device_error)
    }

    /// Waits until all work queued on the device has completed.
    ///
    /// Diagnostics that sample driver memory need a quiescent device; this is the
    /// audited barrier already used before admission queries.
    ///
    /// # Errors
    /// Returns a driver error when the device context cannot be bound or synchronized.
    pub fn synchronize(&self) -> Result<()> {
        self.reclaim_barrier()
    }

    pub(crate) fn drain(&self) -> Result<()> {
        self.stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        // SAFETY: the owned stream remains live and its parent context is current.
        unsafe { self.stream.synchronize() }.map_err(device_error)
    }
}

fn driver_status(status: CUresult) -> Result<()> {
    if status == cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(device_error(format!("CUDA driver status {status}")))
    }
}

/// Discard only the host return value of a graph-safe operation. Resource leases
/// remain owned by its execution context until the enclosing replay completes.
pub(crate) struct GraphUpdate<N>(pub N);

#[expect(
    unsafe_code,
    reason = "Audited forwarding of the caller's live execution context to a graph-safe operation"
)]
impl<N: GraphNode> DeviceOp for GraphUpdate<N> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> StdResult<(), DeviceError> {
        // SAFETY: preserve the caller's context and submission lifetime unchanged.
        unsafe { self.0.execute(context) }.map(|_| ())
    }
}

impl<N: GraphNode> GraphNode for GraphUpdate<N> {}

impl<N: GraphNode> IntoFuture for GraphUpdate<N> {
    type Output = StdResult<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/device.rs"]
mod tests;

/// Native memory clearing lives at the audited device submission boundary.
pub(crate) mod clear {
    use cuda_core::sys::cuMemsetD8Async;
    use cuda_core::sys::cudaError_enum_CUDA_SUCCESS;
    use cutile::cuda_async::device_context::with_default_device_policy;
    use cutile::cuda_async::device_future::DeviceFuture;
    use cutile::{prelude::*, tensor::KernelOutputStored};

    /// Lease partition used when retaining the output allocation for native memset.
    const CLEAR_LEASE_TILE_ELEMENTS: usize = 256;

    pub struct Clear<'a>(pub &'a mut Tensor<f32>);

    #[expect(
        unsafe_code,
        reason = "The live tensor bounds the memset extent; its write lease is retained by the capture context and replay submission"
    )]
    impl DeviceOp for Clear<'_> {
        type Output = ();
        unsafe fn execute(self, context: &ExecutionContext) -> Result<(), DeviceError> {
            let pointer = self.0.device_pointer().cu_deviceptr();
            let bytes = self.0.num_bytes();
            KernelOutputStored::retain(&self.0.partition([CLEAR_LEASE_TILE_ELEMENTS]), context)?;
            // SAFETY: the tensor owns this exact byte range on the captured stream's device.
            let status = unsafe {
                cuMemsetD8Async(pointer, 0, bytes, context.get_cuda_stream().cu_stream())
            };
            if status == cudaError_enum_CUDA_SUCCESS {
                Ok(())
            } else {
                Err(DeviceError::Launch(format!("state clear: {status:?}")))
            }
        }
    }
    impl GraphNode for Clear<'_> {}

    impl IntoFuture for Clear<'_> {
        type Output = Result<(), DeviceError>;
        type IntoFuture = DeviceFuture<(), Self>;
        fn into_future(self) -> Self::IntoFuture {
            match with_default_device_policy(|policy| policy.next_stream()) {
                Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
                Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
            }
        }
    }
}
