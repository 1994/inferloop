//! Safe CUDA memory and kernel launch boundary.
use crate::constants::MATVEC_TILE_COLUMNS;
use crate::constants::MATVEC_TILE_ROWS;
use crate::mlp::Matrix;
use crate::resident::buffers::{Operand, buffer};
use crate::strategy::LinearTiling;
use crate::tuning::memory::arena_budget;
use infer_core::{Error as InferError, ErrorCode, Result};
use infer_cuda_kernels::contract::target::Arch;
use infer_cuda_kernels::contract::{Buffer, Kernels, Width};
use infer_cuda_kernels::jit;
use infer_cuda_kernels::kernels;
use infer_cuda_kernels::runtime::DeviceCopy;
use infer_cuda_kernels::runtime::Stream;
use infer_cuda_kernels::runtime::{Allocation, DType, Device as Runtime, Element, Transfers};
use infer_cuda_kernels::runtime::{Attribute, driver_version};
use infer_cuda_kernels::tile::projection::matvec::{MatvecGeometry, dense_matvec};
use serde::Serialize;
use std::fmt::Display;
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
pub(crate) mod image_preprocess;
pub use image_preprocess::{ImageBatch, Processor as CudaImageProcessor};
mod readback;
pub(crate) mod step_clock;
mod upload;
pub(crate) mod vision_position;
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
    pub(crate) stream: Arc<Stream>,
    /// The contract's view of `stream`, made once so every clone shares its pending transfers.
    runtime: Runtime,
    target: Arch,
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
        jit::enable_cache()
    }
    /// # Errors
    /// Fails when the requested CUDA device or stream is unavailable.
    pub fn new(ordinal: usize) -> Result<Self> {
        let (runtime, target) = Runtime::open(ordinal)?;
        let stream = Arc::clone(runtime.stream_handle());
        Ok(Self {
            runtime,
            stream,
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
        self.runtime.name()
    }

    /// This device's stream as the kernel contract sees it, borrowed for the hot path.
    #[must_use]
    pub const fn tile(&self) -> &Runtime {
        &self.runtime
    }

    /// This device's stream as the kernel contract sees it.
    #[must_use]
    pub fn runtime(&self) -> Runtime {
        self.runtime.clone()
    }

    /// This device's kernels; the single place a kernel source is chosen.
    #[must_use]
    pub fn kernels(&self) -> Arc<Kernels> {
        Arc::new(kernels(self.ordinal))
    }

    #[must_use]
    pub const fn target(&self) -> Arch {
        self.target
    }

    /// # Errors
    /// Returns a shape or device error for an invalid matrix-vector product.
    pub fn matvec<T: DType>(&self, x: &Allocation<f32>, w: &Matrix<T>) -> Result<Allocation<f32>> {
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
        x: &Allocation<f32>,
        w: &Matrix<T>,
        tiling: LinearTiling,
    ) -> Result<Allocation<f32>> {
        let (rows, columns) = (w.rows(), w.columns());
        if rows == 0 || columns == 0 {
            return Err(InferError::invalid("empty CUDA matrix"));
        }
        if x.len() != columns {
            return Err(InferError::invalid("CUDA matrix-vector dimensions"));
        }
        dense_matvec(
            &self.runtime,
            x.buffer(),
            w.allocation().buffer(),
            MatvecGeometry {
                rows,
                columns,
                tile_rows: tiling.rows(),
                tile_columns: tiling.columns(),
            },
        )
    }

    /// Read all of `source` back through `readbacks`, after the work already on this stream.
    /// # Errors
    /// Returns a busy or poisoned owner, or a failed transfer.
    pub(crate) async fn read_allocation<T: DType + DeviceCopy>(
        &self,
        source: &Allocation<T>,
        readbacks: &mut Readbacks<T>,
    ) -> Result<Vec<T>> {
        readbacks.read_async(self, source.clone()).await
    }

    /// Copy the memory of `buffer` back once all queued work has finished.
    ///
    /// # Errors
    /// Returns a backend error when the device is unbound or the copy fails.
    pub(crate) fn read_buffer<T: DType + Element>(
        &self,
        buffer: Buffer<'_, T::Marker>,
    ) -> Result<Vec<T>> {
        self.reclaim_barrier()?;
        self.runtime.read_buffer(buffer)
    }
}

/// Copy every `(dst_offset, src_offset, elements)` run of `src` into `dst` within one transfer
/// batch, which retains both owners once and records one completion event.
/// # Errors
/// Rejects a run outside either buffer or a copy the driver does not enqueue.
pub(crate) fn copy_runs<M: Width>(
    transfers: &Transfers<'_>,
    dst: Buffer<'_, M>,
    src: Buffer<'_, M>,
    runs: impl IntoIterator<Item = (usize, usize, usize)>,
) -> Result<()> {
    let outside = || InferError::invalid("device copy range outside tensor");
    let range = |offset: usize, elements: usize| {
        offset
            .checked_add(elements)
            .map(|end| offset..end)
            .ok_or_else(outside)
    };
    for (dst_offset, src_offset, elements) in runs {
        transfers.copy(
            dst.slice(range(dst_offset, elements)?)
                .map_err(|_| outside())?,
            src.slice(range(src_offset, elements)?)
                .map_err(|_| outside())?,
        )?;
    }
    Ok(())
}

#[expect(
    unsafe_code,
    reason = "Audited driver queries and stream drain with an owned device and a bound context"
)]
impl CudaDevice {
    pub(crate) fn attribute(&self, attribute: Attribute) -> Result<u32> {
        self.runtime.attribute(attribute)
    }

    /// CUDA driver API version, used to invalidate measured kernel decisions.
    /// # Errors
    /// Returns a driver error when the installed driver cannot report its version.
    pub(crate) fn driver_version() -> Result<u32> {
        driver_version()
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
        Ok(DeviceProfile {
            name: self.name()?,
            architecture: Arch::from_sm(
                self.attribute(Attribute::ComputeMajor)?,
                self.attribute(Attribute::ComputeMinor)?,
            )
            .name(),
            multiprocessors: self.attribute(Attribute::Multiprocessors)?,
            total_memory_bytes: self.memory_info()?.1,
            memory_clock_khz: self.attribute(Attribute::MemoryClockKhz)?,
            memory_bus_width_bits: self.attribute(Attribute::MemoryBusWidthBits)?,
            l2_cache_bytes: self.attribute(Attribute::L2CacheBytes)?,
        })
    }

    /// Copy `elements` values from `src_offset` of `src` to `dst_offset` of `dst` on this
    /// device's stream, with no host round trip; prefix caching snapshots KV state this way.
    /// Owners retain both operands through the next stream fence.
    /// # Errors
    /// Rejects a range outside either operand or a copy the driver does not enqueue.
    pub fn copy_d2d_at<
        M: Width,
        D: Operand<Marker = M> + ?Sized,
        S: Operand<Marker = M> + ?Sized,
    >(
        &self,
        dst: &D,
        src: &S,
        dst_offset: usize,
        src_offset: usize,
        elements: usize,
    ) -> Result<()> {
        self.copy_d2d_runs(dst, src, [(dst_offset, src_offset, elements)])
    }

    /// Copy every `(dst_offset, src_offset, elements)` run of `src` into `dst` as one batch on
    /// this device's stream: owners are retained once for the batch, so a block-by-block KV
    /// copy costs one completion event rather than one per block.
    /// # Errors
    /// Rejects a run outside either operand or a copy the driver does not enqueue.
    pub fn copy_d2d_runs<
        M: Width,
        D: Operand<Marker = M> + ?Sized,
        S: Operand<Marker = M> + ?Sized,
    >(
        &self,
        dst: &D,
        src: &S,
        runs: impl IntoIterator<Item = (usize, usize, usize)>,
    ) -> Result<()> {
        let (dst, src) = (
            buffer(dst).map_err(device_error)?,
            buffer(src).map_err(device_error)?,
        );
        self.runtime
            .enqueue(|transfers| copy_runs(transfers, dst, src, runs))
    }

    /// Enqueue `work` on this device's stream; see [`Runtime::enqueue`].
    /// # Errors
    /// Returns what `work` returns, or a failure to track the batch.
    pub(crate) fn enqueue<R>(&self, work: impl FnOnce(&Transfers<'_>) -> Result<R>) -> Result<R> {
        self.runtime.enqueue(work)
    }

    /// # Errors
    /// Returns a driver error if available device memory cannot be queried.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.runtime.memory_info()
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
        self.runtime.synchronize()?;
        // Memory finished transfers retained returns to the pool before the next query.
        self.runtime.reap()
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
        unsafe { self.stream.synchronize() }.map_err(device_error)?;
        self.runtime.reap()
    }
}

#[cfg(test)]
#[path = "../tests/unit/device_tensors.rs"]
mod tensors;

#[cfg(test)]
#[path = "../tests/unit/device.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/device_transfers.rs"]
mod transfer_tests;
