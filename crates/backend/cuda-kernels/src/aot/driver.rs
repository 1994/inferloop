//! Minimal CUDA driver bindings and the objects a kernel provider needs.
//!
//! This module is the `aot` profile's device boundary. It deliberately depends on nothing from
//! the CUDA toolkit: the library is opened with `libloading` and every entry point is declared
//! here. The AOT artifacts and their manifests arrive in a later slice; this layer only proves
//! that device allocation, transfers, module loading and launches do not need headers.
#![expect(
    unsafe_code,
    reason = "Every driver entry point is a raw FFI call whose validity is the caller's contract"
)]

use infer_core::{Error, ErrorCode, Result};
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::Arc;

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuDeviceptr = u64;
type CuStream = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

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
type GetErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

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
    get_error_name: GetErrorName,
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
            get_error_name: symbol(&library, b"cuGetErrorName\0")?,
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

/// Opens one device and owns its primary context and stream.
pub struct Device {
    driver: Arc<Driver>,
    cu_handle: CuDevice,
    context: CuContext,
    stream: CuStream,
    ordinal: usize,
}

impl Device {
    /// Open `ordinal` and bind its primary context on the calling thread.
    /// # Errors
    /// Fails when the driver is missing, the device cannot be opened, or the stream cannot be created.
    pub fn open(ordinal: usize) -> Result<Self> {
        let driver = Arc::new(Driver::load()?);
        let ordinal_i32 =
            i32::try_from(ordinal).map_err(|_| backend("device ordinal exceeds i32".to_owned()))?;
        // SAFETY: each call uses the signature declared in this module and its result is checked
        // before the handle is used; the library outlives every pointer.
        let status = unsafe { (driver.init)(0) };
        driver.check(status)?;
        let mut device: CuDevice = 0;
        // SAFETY: `device` is writable and the driver writes one ordinal-selected device.
        let status = unsafe { (driver.device_get)(&raw mut device, ordinal_i32) };
        driver.check(status)?;
        let mut context = ptr::null_mut();
        // SAFETY: `context` is writable and retained for the lifetime of `Self`.
        let status = unsafe { (driver.primary_ctx_retain)(&raw mut context, device) };
        driver.check(status)?;
        // SAFETY: the retained context is live and becomes current on this thread.
        let status = unsafe { (driver.ctx_set_current)(context) };
        driver.check(status)?;
        let mut stream = ptr::null_mut();
        // SAFETY: `stream` is writable and destroyed by `Drop`.
        let status = unsafe { (driver.stream_create)(&raw mut stream, 0) };
        driver.check(status)?;
        Ok(Self {
            driver,
            cu_handle: device,
            context,
            stream,
            ordinal,
        })
    }

    /// Ordinal this device was opened for.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Context handle owned by this device.
    #[must_use]
    pub const fn context(&self) -> *mut c_void {
        self.context
    }

    /// Allocate `bytes` of device memory.
    /// # Errors
    /// Fails when the driver rejects the allocation.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        let mut pointer: CuDeviceptr = 0;
        // SAFETY: the pointer is only read after a successful driver call.
        unsafe {
            self.driver
                .check((self.driver.mem_alloc)(&raw mut pointer, bytes))?;
        }
        Ok(Buffer {
            driver: Arc::clone(&self.driver),
            pointer,
            bytes,
        })
    }

    /// Copy host bytes into `destination` on the device stream.
    /// # Errors
    /// Fails when the source is larger than the destination or the copy fails.
    pub fn upload(&self, destination: &Buffer, source: &[u8]) -> Result<()> {
        if source.len() > destination.bytes {
            return Err(Error::invalid("upload exceeds the destination buffer"));
        }
        // SAFETY: the destination is a live allocation of at least `source.len()` bytes and the
        // stream belongs to this device.
        unsafe {
            self.driver.check((self.driver.memcpy_htod)(
                destination.pointer,
                source.as_ptr().cast(),
                source.len(),
                self.stream,
            ))
        }
    }

    /// Copy device bytes into `destination`.
    /// # Errors
    /// Fails when the destination is smaller than the source or the copy fails.
    pub fn download(&self, source: &Buffer, destination: &mut [u8]) -> Result<()> {
        if destination.len() < source.bytes {
            return Err(Error::invalid("download exceeds the destination slice"));
        }
        // SAFETY: the source is a live allocation of `source.bytes` bytes and the host slice is
        // writable for the same length.
        unsafe {
            self.driver.check((self.driver.memcpy_dtoh)(
                destination.as_mut_ptr().cast(),
                source.pointer,
                source.bytes,
                self.stream,
            ))
        }
    }

    /// Wait for every operation enqueued on this device's stream.
    /// # Errors
    /// Fails when the driver reports a stream error.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: the stream belongs to this device and stays alive through the call.
        unsafe { self.driver.check((self.driver.stream_sync)(self.stream)) }
    }

    /// Load a cubin image.
    /// # Errors
    /// Fails when the image is not a valid module for this device.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        let mut module = ptr::null_mut();
        // SAFETY: `image` is a complete cubin for the duration of the call; the driver copies it.
        unsafe {
            self.driver.check((self.driver.module_load)(
                &raw mut module,
                image.as_ptr().cast(),
            ))?;
        }
        Ok(Module {
            driver: Arc::clone(&self.driver),
            module,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: the stream was created by this object and is destroyed once.
        unsafe {
            let _ = (self.driver.stream_destroy)(self.stream);
        }
        // SAFETY: the primary context was retained by this object and is released once.
        unsafe {
            let _ = (self.driver.primary_ctx_release)(self.cu_handle);
        }
    }
}

/// Owned device allocation.
pub struct Buffer {
    driver: Arc<Driver>,
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
        // SAFETY: the allocation was created by this driver and is freed once.
        unsafe {
            let _ = (self.driver.mem_free)(self.pointer);
        }
    }
}

/// Loaded cubin module.
pub struct Module {
    driver: Arc<Driver>,
    module: CuModule,
}

impl Module {
    /// Resolve one kernel by name.
    /// # Errors
    /// Fails when the module has no function with that name.
    pub fn function(&self, name: &[u8]) -> Result<Function> {
        let mut symbol = ptr::null_mut();
        let mut terminated = name.to_vec();
        terminated.push(0);
        // SAFETY: the module is loaded and the name is NUL-terminated for the call.
        unsafe {
            self.driver.check((self.driver.module_get_function)(
                &raw mut symbol,
                self.module,
                terminated.as_ptr().cast(),
            ))?;
        }
        Ok(Function {
            driver: Arc::clone(&self.driver),
            function: symbol,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: the module was loaded by this object and is unloaded once.
        unsafe {
            let _ = (self.driver.module_unload)(self.module);
        }
    }
}

/// Resolved kernel ready to launch.
pub struct Function {
    driver: Arc<Driver>,
    function: CuFunction,
}

impl Function {
    /// Launch on the device stream.
    /// # Errors
    /// Fails when the driver rejects the launch.
    pub fn launch(
        &self,
        device: &Device,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        // SAFETY: the caller supplies parameter pointers matching the cubin signature and a
        // stream belonging to the device; the driver validates the launch configuration.
        unsafe {
            self.driver.check((self.driver.launch)(
                self.function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                device.stream,
                parameters.as_mut_ptr(),
                ptr::null_mut(),
            ))
        }
    }
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
