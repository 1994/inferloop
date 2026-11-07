//! The host's runtime over the CUDA driver alone, under the names the cuTile build gives it.
//!
//! An AOT build has no cuTile and no CUDA toolkit, so this runtime talks to `libcuda` through
//! the driver bindings. Device memory, copies, graph capture and replay are real driver work.
//! Operations the cuTile build completes asynchronously complete here before they return:
//! correct, but the host waits on the stream where the cuTile build would not.
#![expect(
    unsafe_code,
    reason = "Typed views of driver memory whose element types promise every bit pattern is valid"
)]
use super::driver;
use crate::contract::buffer::{Bf16, Buffer, Fp4Pair, Fp8, Owner, Width};
use crate::contract::recording::Recording;
use crate::contract::target::Arch;
use infer_core::{Error, Result};
use std::any::Any;
use std::ffi::c_void;
use std::future::{Future, ready};
use std::marker::PhantomData;
use std::mem::{size_of, size_of_val};
use std::ops::Range;
use std::result::Result as StdResult;
use std::slice;
use std::sync::{Arc, Weak};

mod events;
mod handles;
mod launch;
mod operation;
mod readback;
mod staging;
mod timing;

pub use super::driver::Function;
pub use events::{record_event, timing_event};
pub use half::{bf16, f16};
pub use handles::{CudaEvent, Event, Module, PinnedHostBuffer};
pub use launch::{Argument, KernelLaunch};
pub use operation::{Operation, Pending};
pub use readback::{Copies, Flight, Pinned, Readback};
pub use staging::Staging;
pub use timing::{Samples, Timing};

/// An element type device memory holds.
///
/// # Safety
/// Every bit pattern of `size_of::<Self>()` bytes must be a valid value, with no padding, so device
/// bytes can be read as `Self` and `Self` written as bytes.
pub unsafe trait DType: Copy + Default + Send + Sync + 'static {}

/// A type the driver may copy between host and device memory as plain bytes.
pub trait DeviceCopy: DType {}

/// One FP8 E4M3 code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
#[expect(
    non_camel_case_types,
    reason = "Named as the cuTile build names the element type"
)]
pub struct f8e4m3fn(pub u8);

impl f8e4m3fn {
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }
}

/// Two FP4 E2M1 codes packed into one byte, the low nibble first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
#[expect(
    non_camel_case_types,
    reason = "Named as the cuTile build names the element type"
)]
pub struct f4e2m1fnx2(u8);

impl f4e2m1fnx2 {
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }
}

macro_rules! plain {
    ($($t:ty),*) => {$(
        // SAFETY: a primitive or transparent wrapper of one, with no padding and no invalid bits.
        unsafe impl DType for $t {}
        impl DeviceCopy for $t {}
    )*};
}
plain!(
    f32, i32, u32, u8, i8, u16, i16, u64, i64, bf16, f16, f8e4m3fn, f4e2m1fnx2
);

/// The contract's marker for an element type.
pub trait Element {
    type Marker: Width;
}
impl Element for f32 {
    type Marker = Self;
}
impl Element for i32 {
    type Marker = Self;
}
impl Element for u32 {
    type Marker = Self;
}
impl Element for bf16 {
    type Marker = Bf16;
}
impl Element for f4e2m1fnx2 {
    type Marker = Fp4Pair;
}
impl Element for f8e4m3fn {
    type Marker = Fp8;
}

/// Device memory the host owns; clones share it.
pub struct Allocation<T: DType> {
    memory: Arc<driver::Buffer>,
    /// Shares `memory`, typed as the contract's owner, so buffers can name it.
    owner: Owner,
    device: usize,
    /// First element, counted in `T`, from the start of `memory`.
    offset: usize,
    len: usize,
    element: PhantomData<T>,
}

impl<T: DType> Clone for Allocation<T> {
    fn clone(&self) -> Self {
        Self {
            memory: Arc::clone(&self.memory),
            owner: Arc::clone(&self.owner),
            device: self.device,
            offset: self.offset,
            len: self.len,
            element: PhantomData,
        }
    }
}

impl<T: DType> Allocation<T> {
    fn whole(memory: driver::Buffer, device: usize, len: usize) -> Self {
        let memory = Arc::new(memory);
        let owner: Owner = Arc::clone(&memory) as Arc<dyn Any + Send + Sync>;
        Self {
            memory,
            owner,
            device,
            offset: 0,
            len,
            element: PhantomData,
        }
    }

    fn address(&self) -> u64 {
        self.memory.pointer() + (self.offset * size_of::<T>()) as u64
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Ordinal of the device holding the memory.
    #[must_use]
    pub const fn device(&self) -> usize {
        self.device
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The allocation as a kernel operand, kept alive by any graph that records it.
    #[must_use]
    pub fn buffer(&self) -> Buffer<'_, T> {
        // SAFETY: the range lies inside memory this value keeps alive through `owner`, and a
        // retaining recording keeps the owner for its graph's lifetime.
        unsafe { Buffer::owned(self.address(), self.len, &self.owner) }
    }

    /// A handle on this memory that does not keep it alive.
    #[must_use]
    pub fn watch(&self) -> Watch {
        Watch(Arc::downgrade(&self.owner))
    }

    /// Identifies this memory while any share of it lives.
    #[must_use]
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.owner).cast::<()>() as usize
    }

    /// Elements `range` as an allocation of their own, sharing this memory.
    /// # Errors
    /// Rejects a range outside the allocation.
    pub fn slice(&self, range: Range<usize>) -> Result<Self> {
        let len = range
            .end
            .checked_sub(range.start)
            .filter(|_| range.end <= self.len)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "slice {range:?} of an allocation of {} elements",
                    self.len
                ))
            })?;
        Ok(Self {
            offset: self.offset + range.start,
            len,
            ..self.clone()
        })
    }

    /// The allocation as `count` equal, disjoint rows, each sharing this memory.
    /// # Errors
    /// Rejects zero rows and memory that does not divide into `count` rows.
    pub fn rows(&self, count: usize) -> Result<Vec<Self>> {
        if count == 0 || !self.len.is_multiple_of(count) {
            return Err(Error::invalid(format!(
                "{} elements do not divide into {count} rows",
                self.len
            )));
        }
        let row = self.len / count;
        (0..count)
            .map(|index| self.slice(index * row..(index + 1) * row))
            .collect()
    }

    /// All of this memory read as `U`s, sharing it.
    /// # Errors
    /// Rejects memory whose bytes do not divide into `U`s or start misaligned for `U`.
    pub fn reinterpret<U: DType>(&self) -> Result<Allocation<U>> {
        let bytes = self.len * size_of::<T>();
        let start = self.offset * size_of::<T>();
        if !bytes.is_multiple_of(size_of::<U>()) || !start.is_multiple_of(size_of::<U>()) {
            return Err(Error::invalid(format!(
                "{bytes} bytes at {start} are not whole elements of {} bytes",
                size_of::<U>()
            )));
        }
        Ok(Allocation {
            memory: Arc::clone(&self.memory),
            owner: Arc::clone(&self.owner),
            device: self.device,
            offset: start / size_of::<U>(),
            len: bytes / size_of::<U>(),
            element: PhantomData,
        })
    }
}

impl<T: DType + Element> Allocation<T> {
    /// The allocation as the contract types it.
    #[must_use]
    pub fn operand(&self) -> Buffer<'_, T::Marker> {
        // SAFETY: as for `buffer`; each `Element` marker names exactly its element's width.
        unsafe { Buffer::owned(self.address(), self.len, &self.owner) }
    }
}

impl Allocation<bf16> {
    /// The allocation as a BF16 kernel operand.
    #[must_use]
    pub fn bf16_buffer(&self) -> Buffer<'_, Bf16> {
        self.operand()
    }
}

/// A weak handle on an allocation's memory; see [`Allocation::watch`].
#[derive(Clone, Debug)]
pub struct Watch(Weak<dyn Any + Send + Sync>);

impl Watch {
    /// Whether the watched memory is still held by anyone.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.0.strong_count() > 0
    }

    /// Whether `memory` is the allocation this watches, and it is still alive.
    #[must_use]
    pub fn watches<T: DType>(&self, memory: &Allocation<T>) -> bool {
        self.0
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, &memory.owner))
    }
}

/// One stream on one device. Clones share the stream.
#[derive(Clone)]
pub struct Device {
    driver: Arc<driver::Device>,
    ordinal: usize,
}

impl Device {
    /// Open device `ordinal` with a new stream, and the architecture it reports.
    /// # Errors
    /// Fails when the driver, the device or its stream is unavailable.
    pub fn open(ordinal: usize) -> Result<(Self, Arch)> {
        let driver = driver::Device::open(ordinal)?;
        let arch = driver.arch()?;
        Ok((
            Self {
                driver: Arc::new(driver),
                ordinal,
            },
            arch,
        ))
    }

    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    fn allocate<T: DType>(&self, len: usize) -> Result<Allocation<T>> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("allocation size overflow"))?;
        // A zero-length allocation still names an address, as the cuTile build's does.
        let memory = self.driver.alloc(bytes.max(1))?;
        Ok(Allocation::whole(memory, self.ordinal, len))
    }

    /// `len` zeroed elements, complete when this returns.
    /// # Errors
    /// Reports a failed allocation.
    pub fn zeros<T: DType>(&self, len: usize) -> Result<Allocation<T>> {
        let allocation = self.allocate::<T>(len)?;
        // SAFETY: the range is exactly the allocation just made.
        unsafe {
            self.driver
                .zero(allocation.address(), len * size_of::<T>())?;
        }
        self.driver.synchronize()?;
        Ok(allocation)
    }

    /// Zero `allocation` on the stream, without waiting.
    fn zero<T: DType>(&self, allocation: &Allocation<T>) -> Result<()> {
        // SAFETY: the range is exactly the allocation's, which it keeps alive.
        unsafe {
            self.driver
                .zero(allocation.address(), allocation.len() * size_of::<T>())
        }
    }

    /// Copy `values` into `dst` after the work already on the stream, and wait.
    fn write<T: DType, M: Width>(&self, dst: Buffer<'_, M>, values: &[T]) -> Result<()> {
        let bytes = size_of_val(values);
        if dst.len().checked_mul(M::BYTES) != Some(bytes) {
            return Err(Error::invalid("host write extent"));
        }
        // SAFETY: `DType` promises plain bytes, so the values read as their bytes.
        let source = unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), bytes) };
        // SAFETY: the buffer's contract guarantees `bytes` live bytes at its address, as checked.
        unsafe { self.driver.write(dst.address(), source) }
    }

    /// A staging ring of `depth` slots of `elements` values each, for [`Transfers::upload`].
    /// # Errors
    /// Rejects an empty ring.
    pub fn staging<T: DeviceCopy>(&self, elements: usize, depth: usize) -> Result<Staging<T>> {
        Staging::new(elements, depth)
    }

    /// `len` zeroed elements; the future resolves once the device has written them.
    /// # Errors
    /// Reports a failed allocation.
    pub fn zeroed<T: DType>(&self, len: usize) -> impl Future<Output = Result<Allocation<T>>> {
        ready(self.zeros(len))
    }

    /// A copy of `values`, complete when this returns.
    /// # Errors
    /// Reports a failed allocation or transfer.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "The cuTile build's signature, which keeps the values"
    )]
    pub fn upload<T: DType>(&self, values: Vec<T>) -> Result<Allocation<T>> {
        let allocation = self.allocate::<T>(values.len())?;
        // SAFETY: `DType` promises plain bytes, so the values read as their bytes.
        let bytes = unsafe {
            slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * size_of::<T>())
        };
        // SAFETY: the destination is the allocation just made, exactly `bytes.len()` long.
        unsafe { self.driver.write(allocation.address(), bytes) }?;
        Ok(allocation)
    }

    /// The contents of `allocation`, after the work already on this stream.
    /// # Errors
    /// Reports a failed transfer.
    pub fn read<T: DType>(&self, allocation: &Allocation<T>) -> Result<Vec<T>> {
        let mut values = vec![T::default(); allocation.len()];
        // SAFETY: `DType` promises any bytes are valid values, so the vector reads as bytes.
        let bytes = unsafe {
            slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<u8>(),
                values.len() * size_of::<T>(),
            )
        };
        // SAFETY: the allocation keeps its range alive for the call.
        unsafe { self.driver.read_into(allocation.address(), bytes) }?;
        Ok(values)
    }

    /// The `buffer.len()` elements at `buffer`, copied back synchronously.
    /// # Errors
    /// Reports a failed transfer.
    pub fn read_buffer<T: DType + Element>(&self, buffer: Buffer<'_, T::Marker>) -> Result<Vec<T>> {
        let mut values = vec![T::default(); buffer.len()];
        // SAFETY: as in `read`.
        let bytes = unsafe {
            slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<u8>(),
                values.len() * size_of::<T>(),
            )
        };
        // SAFETY: the buffer's contract guarantees `len` live elements at its address.
        unsafe { self.driver.read_into(buffer.address(), bytes) }?;
        Ok(values)
    }

    /// Record kernels into a graph that keeps what they address alive.
    /// # Errors
    /// Returns what `record` returns, or a capture failure.
    pub fn capture(&self, record: impl FnOnce(&Recording<'_>) -> Result<()>) -> Result<Graph> {
        Ok(Graph(Arc::new(self.driver.capture(record)?)))
    }

    /// Run a launch on this stream; it completes when the device has.
    /// # Errors
    /// Returns device failures.
    pub fn run(&self, launch: Launch) -> impl Future<Output = Result<()>> {
        ready(self.sync(launch))
    }

    /// Run a launch and wait until the device has finished it.
    /// # Errors
    /// Returns device failures.
    pub fn sync(&self, launch: Launch) -> Result<()> {
        for graph in launch.0 {
            graph.launch()?;
        }
        self.driver.synchronize()
    }

    /// Enqueue a replay of `graph` without waiting; dropping the graph waits for its stream.
    /// # Errors
    /// Rejects a graph captured on another stream, or returns the driver's refusal.
    pub fn submit(&self, graph: &Graph) -> Result<()> {
        if !graph.runs_on(self) {
            return Err(Error::invalid("graph replay on another stream"));
        }
        graph.0.launch()
    }

    /// Wait for everything queued on this stream.
    /// # Errors
    /// Returns device failures.
    pub fn fence(&self) -> impl Future<Output = Result<()>> {
        ready(self.driver.synchronize())
    }

    /// Wait until this device's stream has finished everything queued on it.
    /// # Errors
    /// Returns device failures.
    pub fn drain_stream(&self) -> Result<()> {
        self.driver.synchronize()
    }

    /// Wait until the device has finished its work; this runtime uses one stream per device.
    /// # Errors
    /// Returns device failures.
    pub fn synchronize(&self) -> Result<()> {
        self.driver.synchronize()
    }

    /// Enqueue `work`'s copies on this stream; they complete before this returns, so whatever
    /// they read or write may be released afterwards.
    /// # Errors
    /// Returns what `work` returns, or a device failure.
    pub fn enqueue<R>(&self, work: impl FnOnce(&Transfers<'_>) -> Result<R>) -> Result<R> {
        let result = work(&Transfers { device: self });
        self.driver.synchronize()?;
        result
    }

    /// Release what finished transfers retained; transfers here complete before they return.
    /// # Errors
    /// Never fails; the signature matches the cuTile build's.
    pub const fn reap(&self) -> Result<()> {
        Ok(())
    }

    /// `attribute` of this device.
    /// # Errors
    /// Returns the driver's failure, or a negative value.
    pub fn attribute(&self, attribute: Attribute) -> Result<u32> {
        self.driver.attribute(attribute.driver())
    }

    /// Free and total device memory, in bytes.
    /// # Errors
    /// Returns the driver's failure.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.driver.memory_info()
    }

    /// Memory an allocation pool holds but no allocation uses: none, since this runtime allocates
    /// from the driver directly.
    /// # Errors
    /// Never fails; the signature matches the cuTile build's.
    pub const fn pool_reclaimable_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    /// The device's marketing name.
    /// # Errors
    /// Returns the driver's failure.
    pub fn name(&self) -> Result<String> {
        self.driver.name()
    }

    /// Make this device's context current on the calling thread.
    /// # Errors
    /// Returns the driver's failure.
    pub fn bind_to_thread(&self) -> Result<()> {
        self.driver.bind()
    }

    /// Whether `other` enqueues on the same stream.
    #[must_use]
    pub fn shares_stream(&self, other: &Self) -> bool {
        self.driver.shares_session(&other.driver)
    }

    /// The stream's identity, which a kernel launch is pinned to.
    #[must_use]
    pub fn stream_id(&self) -> usize {
        self.driver.stream() as usize
    }

    /// The stream handle a vendor library enqueues on.
    #[must_use]
    pub fn raw_stream(&self) -> *mut c_void {
        self.driver.stream()
    }

    /// The context that owns this device's stream, as an opaque identity.
    /// # Errors
    /// Never fails; the signature matches the cuTile build's.
    pub fn context_identity(&self) -> Result<usize> {
        Ok(self.driver.context())
    }

    /// Accept an operand address for a vendor library. This runtime allocates every buffer with
    /// `cuMemAlloc` in this device's context, so any address it hands out belongs here.
    /// # Errors
    /// Never fails; the signature matches the cuTile build's.
    pub const fn check_operand(&self, _address: u64) -> StdResult<(), String> {
        Ok(())
    }
}

/// The installed driver's CUDA API version.
/// # Errors
/// Fails when the driver is missing or cannot report it.
pub fn driver_version() -> Result<u32> {
    driver::driver_version()
}

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
const COMPUTE_MAJOR: i32 = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
const COMPUTE_MINOR: i32 = 76;
/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
const MULTIPROCESSORS: i32 = 16;
/// `CU_DEVICE_ATTRIBUTE_WARP_SIZE`.
const WARP_SIZE: i32 = 10;
/// `CU_DEVICE_ATTRIBUTE_CLOCK_RATE`.
const CLOCK_RATE: i32 = 13;
/// `CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`.
const MEMORY_CLOCK_RATE: i32 = 36;
/// `CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`.
const BUS_WIDTH: i32 = 37;
/// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`.
const L2_CACHE: i32 = 38;

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
    /// The driver's `CUdevice_attribute` value, from `cuda.h`.
    const fn driver(self) -> i32 {
        match self {
            Self::ComputeMajor => COMPUTE_MAJOR,
            Self::ComputeMinor => COMPUTE_MINOR,
            Self::Multiprocessors => MULTIPROCESSORS,
            Self::WarpSize => WARP_SIZE,
            Self::ClockKhz => CLOCK_RATE,
            Self::MemoryClockKhz => MEMORY_CLOCK_RATE,
            Self::MemoryBusWidthBits => BUS_WIDTH,
            Self::L2CacheBytes => L2_CACHE,
        }
    }
}

/// A captured graph, replayable any number of times; dropping it waits for its stream.
pub struct Graph(Arc<driver::Graph>);

impl Graph {
    /// Whether this graph replays on `device`'s stream.
    #[must_use]
    pub fn runs_on(&self, device: &Device) -> bool {
        self.0.runs_on(&device.driver)
    }

    /// Wait until this graph's stream has finished everything queued so far.
    /// # Errors
    /// Returns device failures.
    pub fn drain(&self) -> Result<()> {
        self.0.synchronize()
    }

    /// One replay of this graph, for [`Device::run`] or [`Device::sync`].
    #[must_use]
    pub fn launch(&self) -> Launch {
        Launch(vec![Arc::clone(&self.0)])
    }
}

/// Replays to run in order.
pub struct Launch(Vec<Arc<driver::Graph>>);

impl Launch {
    /// Run `next` after this launch, in one submission.
    #[must_use]
    pub fn followed_by(mut self, next: impl FnOnce() -> Self) -> Self {
        self.0.extend(next().0);
        self
    }
}

/// Copies enqueued together on one stream; see [`Device::enqueue`].
pub struct Transfers<'a> {
    device: &'a Device,
}

impl Transfers<'_> {
    /// Copy `src` into `dst` after the work already on the stream. Copying a buffer onto itself
    /// does nothing.
    /// # Errors
    /// Rejects buffers of different lengths or partly overlapping memory, and a copy the driver
    /// does not enqueue.
    pub fn copy<T: Width>(&self, dst: Buffer<'_, T>, src: Buffer<'_, T>) -> Result<()> {
        if dst.len() != src.len() {
            return Err(Error::invalid("device copy lengths differ"));
        }
        let bytes = src
            .len()
            .checked_mul(T::BYTES)
            .ok_or_else(|| Error::invalid("device copy size overflow"))?;
        if dst.address() == src.address() || bytes == 0 {
            return Ok(());
        }
        if overlaps((dst.address(), bytes), (src.address(), bytes)) {
            return Err(Error::invalid("device copy ranges overlap"));
        }
        // SAFETY: both buffers' contracts guarantee `bytes` live bytes, disjoint as checked above.
        unsafe { self.device.driver.copy(dst.address(), src.address(), bytes) }
    }

    /// Copy the first `rows` positions of every head; the capacities may differ.
    /// # Errors
    /// Rejects layouts with different heads or widths, `rows` beyond either capacity, a buffer
    /// partly overlapping the other, and a copy the driver does not enqueue.
    pub fn copy_heads<T: Width>(
        &self,
        dst: HeadMajor<'_, T>,
        src: HeadMajor<'_, T>,
        rows: usize,
    ) -> Result<()> {
        if dst.heads != src.heads
            || dst.dim != src.dim
            || rows > dst.capacity
            || rows > src.capacity
        {
            return Err(Error::invalid("head prefix copy layout or range"));
        }
        let run = rows * src.dim;
        for head in 0..src.heads {
            self.copy(
                dst.buffer
                    .slice(head * dst.capacity * dst.dim..head * dst.capacity * dst.dim + run)?,
                src.buffer
                    .slice(head * src.capacity * src.dim..head * src.capacity * src.dim + run)?,
            )?;
        }
        Ok(())
    }
}

/// Extents of a head-major layout: heads, positions, width.
const HEAD_MAJOR_RANK: usize = 3;

/// A head-major `[heads, capacity, dim]` buffer, as a KV cache lays out its positions.
#[derive(Debug)]
pub struct HeadMajor<'a, T> {
    buffer: Buffer<'a, T>,
    heads: usize,
    capacity: usize,
    dim: usize,
}

impl<T> Clone for HeadMajor<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for HeadMajor<'_, T> {}

impl<'a, T> HeadMajor<'a, T> {
    /// # Errors
    /// Rejects a buffer that is not exactly `heads * capacity * dim` elements.
    pub fn new(
        buffer: Buffer<'a, T>,
        [heads, capacity, dim]: [usize; HEAD_MAJOR_RANK],
    ) -> Result<Self> {
        let len = heads
            .checked_mul(capacity)
            .and_then(|positions| positions.checked_mul(dim));
        if len != Some(buffer.len()) {
            return Err(Error::invalid("head-major buffer extent"));
        }
        Ok(Self {
            buffer,
            heads,
            capacity,
            dim,
        })
    }
}

/// Whether two `(address, bytes)` ranges share a byte.
fn overlaps((left, left_bytes): (u64, usize), (right, right_bytes): (u64, usize)) -> bool {
    let end = |address: u64, bytes: usize| {
        u64::try_from(bytes)
            .ok()
            .and_then(|bytes| address.checked_add(bytes))
            .unwrap_or(u64::MAX)
    };
    left < end(right, right_bytes) && right < end(left, left_bytes)
}

#[cfg(test)]
#[path = "../../tests/unit/aot_runtime.rs"]
mod tests;
