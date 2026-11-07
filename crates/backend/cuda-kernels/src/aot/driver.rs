//! Minimal CUDA driver bindings and the objects a kernel provider needs.
//!
//! This module is the `aot` source's device boundary. It deliberately depends on nothing from
//! the CUDA toolkit: the library is opened with `libloading` and every entry point is declared
//! here, so device allocation, transfers, module loading and launches do not need headers.
#![expect(
    unsafe_code,
    reason = "Every driver entry point is a raw FFI call whose validity is the caller's contract"
)]

use crate::contract::buffer::Owner;
use crate::contract::recording::Recording;
use crate::contract::target::Arch;
use infer_core::{Error, ErrorCode, Result};
use libloading::Library;
use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;
use std::mem::size_of_val;
use std::ptr;
#[cfg(not(all(feature = "tile", target_os = "linux")))]
use std::result::Result as StdResult;
use std::slice;
use std::sync::Arc;
#[cfg(not(all(feature = "tile", target_os = "linux")))]
use std::sync::OnceLock;

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
const ATTRIBUTE_SM_MAJOR: i32 = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
const ATTRIBUTE_SM_MINOR: i32 = 76;
/// Bytes a device name query may write, NUL included.
const NAME_BYTES: usize = 256;
/// `CU_STREAM_CAPTURE_MODE_GLOBAL`.
const CAPTURE_MODE_GLOBAL: u32 = 0;
/// `CU_EVENT_DEFAULT`: an event that records timing.
const EVENT_DEFAULT: u32 = 0;
/// `CU_EVENT_RECORD_DEFAULT`.
const RECORD_DEFAULT: u32 = 0;
/// `CU_EVENT_RECORD_EXTERNAL`: a record inside a capture stays observable from the host.
const RECORD_EXTERNAL: u32 = 1;

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuDeviceptr = u64;
type CuStream = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;
type CuEvent = *mut c_void;

type Init = unsafe extern "C" fn(u32) -> CuResult;
type DeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type PrimaryCtxRetain = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type PrimaryCtxRelease = unsafe extern "C" fn(CuDevice) -> CuResult;
type CtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type MemAlloc = unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult;
type MemFree = unsafe extern "C" fn(CuDeviceptr) -> CuResult;
type MemcpyHtoD = unsafe extern "C" fn(CuDeviceptr, *const c_void, usize, CuStream) -> CuResult;
type MemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize, CuStream) -> CuResult;
type StreamCreate = unsafe extern "C" fn(*mut CuStream, u32) -> CuResult;
type StreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;
type StreamSync = unsafe extern "C" fn(CuStream) -> CuResult;
type ModuleLoad = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type ModuleGetFunction = unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type ModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type Launch = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type DeviceAttribute = unsafe extern "C" fn(*mut i32, i32, CuDevice) -> CuResult;
type BeginCapture = unsafe extern "C" fn(CuStream, u32) -> CuResult;
type EndCapture = unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult;
type GraphInstantiate = unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u32) -> CuResult;
type GraphLaunch = unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult;
type GraphExecDestroy = unsafe extern "C" fn(CuGraphExec) -> CuResult;
type GraphDestroy = unsafe extern "C" fn(CuGraph) -> CuResult;
type GetErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;
type MemsetD8 = unsafe extern "C" fn(CuDeviceptr, u8, usize, CuStream) -> CuResult;
type MemcpyDtoD = unsafe extern "C" fn(CuDeviceptr, CuDeviceptr, usize, CuStream) -> CuResult;
type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;
type DeviceGetName = unsafe extern "C" fn(*mut c_char, i32, CuDevice) -> CuResult;
type DriverGetVersion = unsafe extern "C" fn(*mut i32) -> CuResult;
type EventCreate = unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult;
type EventDestroy = unsafe extern "C" fn(CuEvent) -> CuResult;
type EventRecord = unsafe extern "C" fn(CuEvent, CuStream, u32) -> CuResult;
type EventElapsed = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
type MemAllocHost = unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult;
type MemFreeHost = unsafe extern "C" fn(*mut c_void) -> CuResult;

struct Driver {
    _library: Library,
    init: Init,
    device_get: DeviceGet,
    primary_ctx_retain: PrimaryCtxRetain,
    primary_ctx_release: PrimaryCtxRelease,
    ctx_set_current: CtxSetCurrent,
    mem_alloc: MemAlloc,
    mem_free: MemFree,
    memcpy_htod: MemcpyHtoD,
    memcpy_dtoh: MemcpyDtoH,
    stream_create: StreamCreate,
    stream_destroy: StreamDestroy,
    stream_sync: StreamSync,
    module_load: ModuleLoad,
    module_get_function: ModuleGetFunction,
    module_unload: ModuleUnload,
    launch: Launch,
    device_attribute: DeviceAttribute,
    begin_capture: BeginCapture,
    end_capture: EndCapture,
    graph_instantiate: GraphInstantiate,
    graph_launch: GraphLaunch,
    graph_exec_destroy: GraphExecDestroy,
    graph_destroy: GraphDestroy,
    get_error_name: GetErrorName,
    memset_d8: MemsetD8,
    memcpy_dtod: MemcpyDtoD,
    mem_get_info: MemGetInfo,
    device_get_name: DeviceGetName,
    version: DriverGetVersion,
    event_create: EventCreate,
    event_destroy: EventDestroy,
    event_record: EventRecord,
    event_elapsed: EventElapsed,
    mem_alloc_host: MemAllocHost,
    mem_free_host: MemFreeHost,
}

impl Driver {
    fn load() -> Result<Self> {
        // SAFETY: `libcuda.so.1` is the driver's stable soname; every symbol below is looked up
        // once and kept alive by `_library` for the lifetime of the driver object.
        let library = unsafe { Library::new("libcuda.so.1") }
            .map_err(|error| backend(format!("CUDA driver unavailable: {error}")))?;
        Ok(Self {
            init: symbol(&library, b"cuInit\0")?,
            device_get: symbol(&library, b"cuDeviceGet\0")?,
            primary_ctx_retain: symbol(&library, b"cuDevicePrimaryCtxRetain\0")?,
            primary_ctx_release: symbol(&library, b"cuDevicePrimaryCtxRelease_v2\0")?,
            ctx_set_current: symbol(&library, b"cuCtxSetCurrent\0")?,
            mem_alloc: symbol(&library, b"cuMemAlloc_v2\0")?,
            mem_free: symbol(&library, b"cuMemFree_v2\0")?,
            memcpy_htod: symbol(&library, b"cuMemcpyHtoDAsync_v2\0")?,
            memcpy_dtoh: symbol(&library, b"cuMemcpyDtoHAsync_v2\0")?,
            stream_create: symbol(&library, b"cuStreamCreate\0")?,
            stream_destroy: symbol(&library, b"cuStreamDestroy_v2\0")?,
            stream_sync: symbol(&library, b"cuStreamSynchronize\0")?,
            module_load: symbol(&library, b"cuModuleLoadData\0")?,
            module_get_function: symbol(&library, b"cuModuleGetFunction\0")?,
            module_unload: symbol(&library, b"cuModuleUnload\0")?,
            launch: symbol(&library, b"cuLaunchKernel\0")?,
            device_attribute: symbol(&library, b"cuDeviceGetAttribute\0")?,
            begin_capture: symbol(&library, b"cuStreamBeginCapture_v2\0")?,
            end_capture: symbol(&library, b"cuStreamEndCapture\0")?,
            graph_instantiate: symbol(&library, b"cuGraphInstantiateWithFlags\0")?,
            graph_launch: symbol(&library, b"cuGraphLaunch\0")?,
            graph_exec_destroy: symbol(&library, b"cuGraphExecDestroy\0")?,
            graph_destroy: symbol(&library, b"cuGraphDestroy\0")?,
            get_error_name: symbol(&library, b"cuGetErrorName\0")?,
            memset_d8: symbol(&library, b"cuMemsetD8Async\0")?,
            memcpy_dtod: symbol(&library, b"cuMemcpyDtoDAsync_v2\0")?,
            mem_get_info: symbol(&library, b"cuMemGetInfo_v2\0")?,
            device_get_name: symbol(&library, b"cuDeviceGetName\0")?,
            version: symbol(&library, b"cuDriverGetVersion\0")?,
            event_create: symbol(&library, b"cuEventCreate\0")?,
            event_destroy: symbol(&library, b"cuEventDestroy_v2\0")?,
            event_record: symbol(&library, b"cuEventRecordWithFlags\0")?,
            event_elapsed: symbol(&library, b"cuEventElapsedTime\0")?,
            mem_alloc_host: symbol(&library, b"cuMemAllocHost_v2\0")?,
            mem_free_host: symbol(&library, b"cuMemFreeHost\0")?,
            _library: library,
        })
    }

    fn check(&self, code: CuResult) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut name = ptr::null();
        // SAFETY: `get_error_name` writes a static string pointer for a non-success code.
        let status = unsafe { (self.get_error_name)(code, &raw mut name) };
        let message = if status == 0 && !name.is_null() {
            // SAFETY: the driver returned a NUL-terminated static string.
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("driver code {code}")
        };
        Err(backend(format!("CUDA driver error: {message}")))
    }
}

/// Primary context and stream shared by everything created from one [`Device`].
///
/// Every object that talks to the driver holds this session, so a buffer or module can never
/// outlive the context it belongs to. Each operation binds the context on the calling thread.
struct Session {
    driver: Driver,
    handle: CuDevice,
    context: CuContext,
    stream: CuStream,
}

impl Session {
    fn bind(&self) -> Result<()> {
        // SAFETY: the retained context stays live for the lifetime of this session.
        let status = unsafe { (self.driver.ctx_set_current)(self.context) };
        self.driver.check(status)
    }

    fn synchronize(&self) -> Result<()> {
        self.bind()?;
        // SAFETY: the stream belongs to this session and its context is current.
        let status = unsafe { (self.driver.stream_sync)(self.stream) };
        self.driver.check(status)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Close paths cannot report through `Drop`; `Device::synchronize` surfaces stream errors.
        if !self.stream.is_null() {
            let _ = self.synchronize();
            // SAFETY: the stream was created by this session and is destroyed once.
            let _ = unsafe { (self.driver.stream_destroy)(self.stream) };
        }
        // SAFETY: the primary context was retained by this session and is released once.
        let _ = unsafe { (self.driver.primary_ctx_release)(self.handle) };
    }
}

/// One opened device with its primary context and stream.
pub struct Device {
    session: Arc<Session>,
    ordinal: usize,
}

impl Device {
    /// Open `ordinal`.
    /// # Errors
    /// Fails when the driver is missing, the device cannot be opened, or the stream cannot be created.
    pub fn open(ordinal: usize) -> Result<Self> {
        let driver = Driver::load()?;
        let ordinal_i32 =
            i32::try_from(ordinal).map_err(|_| backend("device ordinal exceeds i32".to_owned()))?;
        // SAFETY: the signature matches the driver ABI declared in this module.
        let status = unsafe { (driver.init)(0) };
        driver.check(status)?;
        let mut handle: CuDevice = 0;
        // SAFETY: `handle` is writable and the driver writes one ordinal-selected device.
        let status = unsafe { (driver.device_get)(&raw mut handle, ordinal_i32) };
        driver.check(status)?;
        let mut context = ptr::null_mut();
        // SAFETY: `context` is writable; ownership moves into `Session` on success.
        let status = unsafe { (driver.primary_ctx_retain)(&raw mut context, handle) };
        driver.check(status)?;
        // The session owns the context from here, so a later failure releases it through `Drop`.
        let mut session = Session {
            driver,
            handle,
            context,
            stream: ptr::null_mut(),
        };
        session.bind()?;
        let mut stream = ptr::null_mut();
        // SAFETY: `stream` is writable and the session's context is current.
        let status = unsafe { (session.driver.stream_create)(&raw mut stream, 0) };
        session.driver.check(status)?;
        session.stream = stream;
        Ok(Self {
            session: Arc::new(session),
            ordinal,
        })
    }

    /// Ordinal this device was opened for.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Compute capability reported by the driver.
    /// # Errors
    /// Fails when the driver cannot answer or reports a negative capability.
    pub fn arch(&self) -> Result<Arch> {
        let major = self.attribute(ATTRIBUTE_SM_MAJOR)?;
        let minor = self.attribute(ATTRIBUTE_SM_MINOR)?;
        Ok(Arch::from_sm(major, minor))
    }

    pub(crate) fn attribute(&self, attribute: i32) -> Result<u32> {
        let session = &self.session;
        session.bind()?;
        let mut value = 0;
        // SAFETY: `value` is writable and the handle belongs to this session.
        let status =
            unsafe { (session.driver.device_attribute)(&raw mut value, attribute, session.handle) };
        session.driver.check(status)?;
        u32::try_from(value).map_err(|_| backend("negative device attribute".to_owned()))
    }

    /// Capture everything `record` launches into a graph; nothing runs until [`Graph::launch`].
    /// # Errors
    /// Fails when `record` fails, or the driver cannot capture or instantiate the graph. Capture
    /// mode always ends, so the stream stays usable after an error.
    pub fn capture(&self, record: impl FnOnce(&Recording<'_>) -> Result<()>) -> Result<Graph> {
        let session = &self.session;
        session.bind()?;
        // SAFETY: the stream belongs to this session and its context is current.
        let status = unsafe { (session.driver.begin_capture)(session.stream, CAPTURE_MODE_GLOBAL) };
        session.driver.check(status)?;
        let owners = RefCell::new(Vec::new());
        let recorded = record(&Recording::Stream {
            handle: session.stream,
            owners: Some(&owners),
            stream: PhantomData,
        });
        let mut graph = ptr::null_mut();
        // SAFETY: the stream entered capture mode above; `graph` is writable.
        let status = unsafe { (session.driver.end_capture)(session.stream, &raw mut graph) };
        // A failed capture may still return a graph; `Graph` owns and destroys it either way.
        let mut owned = Graph {
            session: Arc::clone(session),
            captured: graph,
            exec: ptr::null_mut(),
            _owners: owners.into_inner(),
        };
        recorded?;
        session.driver.check(status)?;
        // SAFETY: `graph` is the captured graph and `exec` is writable.
        let status =
            unsafe { (session.driver.graph_instantiate)(&raw mut owned.exec, owned.captured, 0) };
        session.driver.check(status)?;
        Ok(owned)
    }

    /// Allocate `bytes` of device memory.
    /// # Errors
    /// Fails when the driver rejects the allocation.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        self.session.bind()?;
        let mut pointer: CuDeviceptr = 0;
        // SAFETY: `pointer` is writable and only read after a successful call.
        let status = unsafe { (self.session.driver.mem_alloc)(&raw mut pointer, bytes) };
        self.session.driver.check(status)?;
        Ok(Buffer {
            session: Arc::clone(&self.session),
            pointer,
            bytes,
        })
    }

    /// Copy host bytes into `destination` and wait for the copy, so `source` can be released.
    /// # Errors
    /// Fails when the source is larger than the destination or the copy fails.
    pub fn upload(&self, destination: &Buffer, source: &[u8]) -> Result<()> {
        if source.len() > destination.bytes {
            return Err(Error::invalid("upload exceeds the destination buffer"));
        }
        self.session.bind()?;
        // SAFETY: the destination holds at least `source.len()` bytes and `source` stays borrowed
        // until the stream synchronizes below.
        let status = unsafe {
            (self.session.driver.memcpy_htod)(
                destination.pointer,
                source.as_ptr().cast(),
                source.len(),
                self.session.stream,
            )
        };
        self.session.driver.check(status)?;
        self.session.synchronize()
    }

    /// Copy a typed `f32` slice into `destination` and wait for the copy.
    /// # Errors
    /// Fails when the source is larger than the destination or the copy fails.
    pub fn upload_f32(&self, destination: &Buffer, source: &[f32]) -> Result<()> {
        // SAFETY: f32 has no padding and the source slice is readable for its full length.
        let bytes =
            unsafe { slice::from_raw_parts(source.as_ptr().cast::<u8>(), size_of_val(source)) };
        self.upload(destination, bytes)
    }

    /// Copy device bytes into a typed `f32` slice and wait for the copy.
    /// # Errors
    /// Fails when the destination is smaller than the source or the copy fails.
    pub fn download_f32(&self, source: &Buffer, destination: &mut [f32]) -> Result<()> {
        // SAFETY: f32 has no padding, every bit pattern is valid and the host slice is writable
        // for its full length.
        let bytes = unsafe {
            slice::from_raw_parts_mut(
                destination.as_mut_ptr().cast::<u8>(),
                size_of_val(destination),
            )
        };
        self.download(source, bytes)
    }

    /// Copy device bytes into `destination` and wait for the copy, ordered after earlier launches.
    /// # Errors
    /// Fails when the destination is smaller than the source or the copy fails.
    pub fn download(&self, source: &Buffer, destination: &mut [u8]) -> Result<()> {
        if destination.len() < source.bytes {
            return Err(Error::invalid("download exceeds the destination slice"));
        }
        self.session.bind()?;
        // SAFETY: the source holds `source.bytes` bytes and `destination` stays mutably borrowed
        // until the stream synchronizes below.
        let status = unsafe {
            (self.session.driver.memcpy_dtoh)(
                destination.as_mut_ptr().cast(),
                source.pointer,
                source.bytes,
                self.session.stream,
            )
        };
        self.session.driver.check(status)?;
        self.session.synchronize()
    }

    /// Zero `bytes` bytes at `address`, ordered on the stream; the caller waits when it must.
    /// # Safety
    /// `address` must start `bytes` writable bytes of this context's device memory.
    pub(crate) unsafe fn zero(&self, address: u64, bytes: usize) -> Result<()> {
        self.session.bind()?;
        // SAFETY: the caller guarantees the range; the stream belongs to this session.
        let status =
            unsafe { (self.session.driver.memset_d8)(address, 0, bytes, self.session.stream) };
        self.session.driver.check(status)
    }

    /// Copy `bytes` bytes device to device, ordered on the stream.
    /// # Safety
    /// Both ranges must be live device memory of this context that do not overlap.
    pub(crate) unsafe fn copy(&self, destination: u64, source: u64, bytes: usize) -> Result<()> {
        self.session.bind()?;
        // SAFETY: the caller guarantees both ranges; the stream belongs to this session.
        let status = unsafe {
            (self.session.driver.memcpy_dtod)(destination, source, bytes, self.session.stream)
        };
        self.session.driver.check(status)
    }

    /// Copy `source` into device memory at `address` and wait, so `source` can be released.
    /// # Safety
    /// `address` must start `source.len()` writable bytes of this context's device memory.
    pub(crate) unsafe fn write(&self, address: u64, source: &[u8]) -> Result<()> {
        self.session.bind()?;
        // SAFETY: the caller guarantees the destination; `source` stays borrowed until the stream
        // synchronizes below.
        let status = unsafe {
            (self.session.driver.memcpy_htod)(
                address,
                source.as_ptr().cast(),
                source.len(),
                self.session.stream,
            )
        };
        self.session.driver.check(status)?;
        self.session.synchronize()
    }

    /// Copy device memory at `address` into `destination` and wait, after earlier work.
    /// # Safety
    /// `address` must start `destination.len()` readable bytes of this context's device memory.
    pub(crate) unsafe fn read_into(&self, address: u64, destination: &mut [u8]) -> Result<()> {
        self.session.bind()?;
        // SAFETY: the caller guarantees the source; `destination` stays borrowed until the stream
        // synchronizes below.
        let status = unsafe {
            (self.session.driver.memcpy_dtoh)(
                destination.as_mut_ptr().cast(),
                address,
                destination.len(),
                self.session.stream,
            )
        };
        self.session.driver.check(status)?;
        self.session.synchronize()
    }

    /// Copy `bytes` host bytes at `source` to device memory at `address`, ordered on the stream
    /// without waiting.
    /// # Safety
    /// `address` must start `bytes` writable bytes of this context's device memory, and `source`
    /// `bytes` readable host bytes that stay unchanged and allocated until the stream has passed
    /// the copy.
    pub(crate) unsafe fn enqueue_write(
        &self,
        address: u64,
        source: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        self.session.bind()?;
        // SAFETY: the caller guarantees both ranges until the stream passes the copy.
        let status = unsafe {
            (self.session.driver.memcpy_htod)(address, source, bytes, self.session.stream)
        };
        self.session.driver.check(status)
    }

    /// Copy `bytes` device bytes at `address` to host memory at `destination`, ordered on the
    /// stream without waiting.
    /// # Safety
    /// `address` must start `bytes` readable bytes of this context's device memory, and
    /// `destination` `bytes` writable host bytes that nothing else touches until the stream has
    /// passed the copy.
    pub(crate) unsafe fn enqueue_read(
        &self,
        destination: *mut c_void,
        address: u64,
        bytes: usize,
    ) -> Result<()> {
        self.session.bind()?;
        // SAFETY: the caller guarantees both ranges until the stream passes the copy.
        let status = unsafe {
            (self.session.driver.memcpy_dtoh)(destination, address, bytes, self.session.stream)
        };
        self.session.driver.check(status)
    }

    /// A timing event in this device's context.
    /// # Errors
    /// Fails when the driver cannot create it.
    pub fn event(&self) -> Result<Event> {
        self.session.bind()?;
        let mut event = ptr::null_mut();
        // SAFETY: `event` is writable and the context is current; ownership moves into `Event`.
        let status = unsafe { (self.session.driver.event_create)(&raw mut event, EVENT_DEFAULT) };
        self.session.driver.check(status)?;
        Ok(Event {
            session: Arc::clone(&self.session),
            event,
        })
    }

    /// `bytes` zeroed bytes of page-locked host memory in this device's context.
    /// # Errors
    /// Fails when the driver cannot allocate them.
    pub fn pinned(&self, bytes: usize) -> Result<Pinned> {
        self.session.bind()?;
        let mut pointer = ptr::null_mut();
        // A zero-length request still names an address, so the slice views stay well defined.
        let bytes = bytes.max(1);
        // SAFETY: `pointer` is writable and the context is current; ownership moves into `Pinned`.
        let status = unsafe { (self.session.driver.mem_alloc_host)(&raw mut pointer, bytes) };
        self.session.driver.check(status)?;
        // SAFETY: the driver just allocated `bytes` writable host bytes at `pointer`.
        unsafe { ptr::write_bytes(pointer.cast::<u8>(), 0, bytes) };
        Ok(Pinned {
            session: Arc::clone(&self.session),
            pointer,
            bytes,
        })
    }

    /// Free and total device memory, in bytes.
    /// # Errors
    /// Fails when the driver cannot report them.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.session.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both outputs are live and the context is current.
        let status = unsafe { (self.session.driver.mem_get_info)(&raw mut free, &raw mut total) };
        self.session.driver.check(status)?;
        Ok((free as u64, total as u64))
    }

    /// The device's marketing name.
    /// # Errors
    /// Fails when the driver cannot report it.
    pub fn name(&self) -> Result<String> {
        let mut name = [0 as c_char; NAME_BYTES];
        let length = i32::try_from(name.len()).map_err(|_| backend("name buffer".to_owned()))?;
        // SAFETY: `name` holds `length` writable bytes and the handle belongs to this session.
        let status = unsafe {
            (self.session.driver.device_get_name)(name.as_mut_ptr(), length, self.session.handle)
        };
        self.session.driver.check(status)?;
        // SAFETY: the driver writes a NUL-terminated string into the buffer.
        Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// The installed driver's CUDA API version.
    /// # Errors
    /// Fails when the driver cannot report it.
    pub fn driver_version(&self) -> Result<u32> {
        let mut version = 0;
        // SAFETY: `version` is a live output; the query needs no context.
        let status = unsafe { (self.session.driver.version)(&raw mut version) };
        self.session.driver.check(status)?;
        u32::try_from(version).map_err(|_| backend("negative driver version".to_owned()))
    }

    /// Make this device's context current on the calling thread.
    /// # Errors
    /// Fails when the driver cannot bind it.
    pub fn bind(&self) -> Result<()> {
        self.session.bind()
    }

    /// The stream this device enqueues on.
    #[must_use]
    pub fn stream(&self) -> *mut c_void {
        self.session.stream
    }

    /// The context this device's stream belongs to, as an opaque identity.
    #[must_use]
    pub fn context(&self) -> usize {
        self.session.context as usize
    }

    /// Whether `other` enqueues on the same stream.
    #[must_use]
    pub fn shares_session(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.session, &other.session)
    }

    /// Wait for every operation enqueued on this device's stream.
    /// # Errors
    /// Fails when the driver reports a stream error.
    pub fn synchronize(&self) -> Result<()> {
        self.session.synchronize()
    }

    /// Load a cubin image.
    /// # Errors
    /// Fails when the image is not a valid module for this device.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        self.session.bind()?;
        let mut module = ptr::null_mut();
        // SAFETY: `image` is a complete cubin for the duration of the call; the driver copies it.
        let status =
            unsafe { (self.session.driver.module_load)(&raw mut module, image.as_ptr().cast()) };
        self.session.driver.check(status)?;
        Ok(Module {
            inner: Arc::new(LoadedModule {
                session: Arc::clone(&self.session),
                module,
            }),
        })
    }
}

/// Captured kernels that can be replayed any number of times.
pub struct Graph {
    session: Arc<Session>,
    captured: CuGraph,
    exec: CuGraphExec,
    /// What the recorded launches address; kept until the graph is destroyed.
    _owners: Vec<Owner>,
}

impl Graph {
    /// Wait until this graph's stream has finished everything queued so far.
    /// # Errors
    /// Fails when the driver reports a stream error.
    pub fn synchronize(&self) -> Result<()> {
        self.session.synchronize()
    }

    /// Whether this graph replays on `device`'s stream.
    #[must_use]
    pub fn runs_on(&self, device: &Device) -> bool {
        Arc::ptr_eq(&self.session, &device.session)
    }

    /// Enqueue one replay; [`Device::synchronize`] waits for it.
    /// # Errors
    /// Fails when the driver rejects the launch.
    pub fn launch(&self) -> Result<()> {
        self.session.bind()?;
        // SAFETY: the graph was instantiated by this session, and the addresses it recorded were
        // promised valid for every replay by the `Projection` caller.
        let status = unsafe { (self.session.driver.graph_launch)(self.exec, self.session.stream) };
        self.session.driver.check(status)
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.session.synchronize();
        if !self.exec.is_null() {
            // SAFETY: the exec was instantiated by this object and is destroyed once.
            let _ = unsafe { (self.session.driver.graph_exec_destroy)(self.exec) };
        }
        if !self.captured.is_null() {
            // SAFETY: the graph was captured by this object and is destroyed once.
            let _ = unsafe { (self.session.driver.graph_destroy)(self.captured) };
        }
    }
}

/// Owned device allocation; dropping it waits for queued work before freeing.
pub struct Buffer {
    session: Arc<Session>,
    pointer: CuDeviceptr,
    bytes: usize,
}

impl Buffer {
    /// Device address of the allocation.
    #[must_use]
    pub const fn pointer(&self) -> u64 {
        self.pointer
    }

    /// Length in bytes.
    #[must_use]
    pub const fn len_bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // A queued launch may still read this allocation, so drain before freeing.
        let _ = self.session.synchronize();
        // SAFETY: the allocation was created by this session and is freed once.
        let _ = unsafe { (self.session.driver.mem_free)(self.pointer) };
    }
}

struct LoadedModule {
    session: Arc<Session>,
    module: CuModule,
}

impl Drop for LoadedModule {
    fn drop(&mut self) {
        let _ = self.session.synchronize();
        // SAFETY: the module was loaded by this session and is unloaded once.
        let _ = unsafe { (self.session.driver.module_unload)(self.module) };
    }
}

/// Loaded cubin module.
pub struct Module {
    inner: Arc<LoadedModule>,
}

impl Module {
    /// Resolve one kernel by name; the function keeps the module loaded.
    /// # Errors
    /// Fails when the module has no function with that name.
    pub fn function(&self, name: &CStr) -> Result<Function> {
        let session = &self.inner.session;
        session.bind()?;
        let mut function = ptr::null_mut();
        // SAFETY: the module is loaded and `name` is NUL-terminated for the call.
        let status = unsafe {
            (session.driver.module_get_function)(
                &raw mut function,
                self.inner.module,
                name.as_ptr(),
            )
        };
        session.driver.check(status)?;
        Ok(Function {
            module: Arc::clone(&self.inner),
            function,
        })
    }
}

/// Resolved kernel ready to launch.
pub struct Function {
    module: Arc<LoadedModule>,
    function: CuFunction,
}

impl Function {
    /// Launch on `recording`: captured when the stream is capturing, enqueued otherwise.
    ///
    /// # Safety
    /// `parameters` must match the kernel signature and each entry must point to a live argument
    /// for this call. Every device address they carry must stay valid, with no conflicting access,
    /// until each resulting launch or replay has finished.
    /// # Errors
    /// Fails when the recording is not a driver stream or the driver rejects the launch.
    pub unsafe fn launch(
        &self,
        recording: &Recording<'_>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        let Some(stream) = recording.stream() else {
            return Err(Error::unsupported("AOT launches record on a driver stream"));
        };
        // SAFETY: the caller upholds the parameter contract and the recording borrows a live
        // stream of this context.
        unsafe { self.launch_on(stream, grid, block, parameters) }
    }

    /// Launch on `stream`, captured when it is capturing.
    ///
    /// # Safety
    /// As for [`Self::launch`], and `stream` must be a live stream of this function's context.
    /// # Errors
    /// Fails when the driver rejects the launch.
    pub unsafe fn launch_on(
        &self,
        stream: *mut c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        let session = &self.module.session;
        session.bind()?;
        // SAFETY: the caller upholds the parameter contract and names a live stream of this
        // context; `self.module` keeps the function loaded.
        let status = unsafe {
            (session.driver.launch)(
                self.function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                stream,
                parameters.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        session.driver.check(status)
    }
}

/// A driver event for device timing; dropping it destroys it.
pub struct Event {
    session: Arc<Session>,
    event: CuEvent,
}

impl Event {
    /// Record this event at the current end of `device`'s stream.
    /// # Errors
    /// Rejects a device of another context, or returns the driver's refusal.
    pub fn record(&self, device: &Device) -> Result<()> {
        if self.session.context != device.session.context {
            return Err(Error::invalid("event recorded on another context"));
        }
        // SAFETY: the stream belongs to `device`, whose context this event was created in.
        unsafe { self.record_on(device.session.stream, false) }
    }

    /// Record this event on `stream`, external when a host must observe a record inside a
    /// capture.
    /// # Safety
    /// `stream` must be a live stream of this event's context.
    pub(crate) unsafe fn record_on(&self, stream: *mut c_void, external: bool) -> Result<()> {
        self.session.bind()?;
        let flags = if external {
            RECORD_EXTERNAL
        } else {
            RECORD_DEFAULT
        };
        // SAFETY: the event is live and the caller guarantees the stream.
        let status = unsafe { (self.session.driver.event_record)(self.event, stream, flags) };
        self.session.driver.check(status)
    }

    /// Milliseconds of device time from this event to `end`, both recorded and complete.
    /// # Errors
    /// Returns the driver's failure, for example while either event is pending.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32> {
        self.session.bind()?;
        let mut ms = 0f32;
        // SAFETY: both events are live and `ms` is writable.
        let status =
            unsafe { (self.session.driver.event_elapsed)(&raw mut ms, self.event, end.event) };
        self.session.driver.check(status)?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let _ = self.session.bind();
        // SAFETY: the event was created by this object and is destroyed once; the driver defers
        // the release past any record still pending.
        let _ = unsafe { (self.session.driver.event_destroy)(self.event) };
    }
}

/// Page-locked host memory; dropping it waits for queued work before freeing.
pub struct Pinned {
    session: Arc<Session>,
    pointer: *mut c_void,
    bytes: usize,
}

impl Pinned {
    /// Host address of the memory.
    #[must_use]
    pub const fn pointer(&self) -> *mut c_void {
        self.pointer
    }

    /// Length in bytes, at least one.
    #[must_use]
    pub const fn len_bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // A queued copy may still use this memory, so drain before freeing.
        let _ = self.session.synchronize();
        // SAFETY: the memory was allocated by this object and is freed once.
        let _ = unsafe { (self.session.driver.mem_free_host)(self.pointer) };
    }
}

// SAFETY: driver entry points are thread-safe once a context is current, and every operation binds
// the session's context on the calling thread first. The handles are never dereferenced by Rust.
unsafe impl Send for Session {}
// SAFETY: as for `Send`; the session has no interior state outside the driver.
unsafe impl Sync for Session {}
// SAFETY: the module handle is immutable after load and only passed back to the driver.
unsafe impl Send for LoadedModule {}
// SAFETY: as for `Send`.
unsafe impl Sync for LoadedModule {}
// SAFETY: the graph handles are immutable after capture and only passed back to the driver.
unsafe impl Send for Graph {}
// SAFETY: as for `Send`.
unsafe impl Sync for Graph {}
// SAFETY: the function handle is immutable after resolution and only passed back to the driver.
unsafe impl Send for Function {}
// SAFETY: as for `Send`.
unsafe impl Sync for Function {}
// SAFETY: the event handle is only passed back to the driver, which synchronizes its records.
unsafe impl Send for Event {}
// SAFETY: as for `Send`.
unsafe impl Sync for Event {}
// SAFETY: the memory is owned by this object; Rust reaches it only through `PinnedHostBuffer`'s
// borrows, which follow the aliasing rules.
unsafe impl Send for Pinned {}
// SAFETY: as for `Send`; shared access hands out only shared views.
unsafe impl Sync for Pinned {}

/// The driver every foreign call shares: a recording hands such a call only a stream, and the
/// entry points do not depend on which session loaded them.
#[cfg(not(all(feature = "tile", target_os = "linux")))]
static SHARED: OnceLock<StdResult<Driver, String>> = OnceLock::new();

#[cfg(not(all(feature = "tile", target_os = "linux")))]
fn shared() -> Result<&'static Driver> {
    SHARED
        .get_or_init(|| Driver::load().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|message| backend(message.clone()))
}

/// Copy `bytes` bytes device to device on `stream`, captured when it is capturing.
/// # Safety
/// `stream` must be a live stream whose context is current, and both ranges live device memory of
/// that context that do not overlap, until the copy or every replay of its graph has run.
#[cfg(not(all(feature = "tile", target_os = "linux")))]
pub unsafe fn copy_on(
    stream: *mut c_void,
    destination: u64,
    source: u64,
    bytes: usize,
) -> Result<()> {
    let driver = shared()?;
    // SAFETY: the caller guarantees the stream and both ranges.
    let status = unsafe { (driver.memcpy_dtod)(destination, source, bytes, stream) };
    driver.check(status)
}

/// The installed driver's CUDA API version, without opening a device.
/// # Errors
/// Fails when the driver is missing or cannot report it.
pub fn driver_version() -> Result<u32> {
    let driver = Driver::load()?;
    let mut version = 0;
    // SAFETY: `version` is a live output; the query needs no context.
    let status = unsafe { (driver.version)(&raw mut version) };
    driver.check(status)?;
    u32::try_from(version).map_err(|_| backend("negative driver version".to_owned()))
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    // SAFETY: the caller names a driver entry point with the ABI signature of `T`; the library
    // outlives every copied function pointer.
    unsafe { library.get::<T>(name).map(|symbol| *symbol) }
        .map_err(|error| backend(format!("CUDA driver symbol unavailable: {error}")))
}

fn backend(message: String) -> Error {
    Error::new(ErrorCode::Backend, message)
}
