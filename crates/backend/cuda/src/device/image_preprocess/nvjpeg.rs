//! Optional nvJPEG hardware decoder. All capability detection and FFI stay in this module.
use crate::device::device_error;
use cuda_core::CudaContext;
use infer_core::{Error, ErrorCode, Result};
use infer_cuda_kernels::contract::target::Arch;
use infer_ir::{EncodedImageInfo, ImageEncoding};
use infer_spi::DeviceHostExecutor;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::{NonNull, from_mut, null, null_mut};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, ThreadId};

// Values and signatures are from nvjpeg.h; only the hardware RGB interleaved path is used.
const SUCCESS: i32 = 0;
const BAD_JPEG: i32 = 3;
const JPEG_NOT_SUPPORTED: i32 = 4;
const ARCH_MISMATCH: i32 = 7;
const IMPLEMENTATION_NOT_SUPPORTED: i32 = 9;
const INCOMPLETE_BITSTREAM: i32 = 10;
const HARDWARE: i32 = 3;
const RGB_INTERLEAVED: i32 = 5;
const UPSAMPLE_INTERPOLATION: u32 = 1 << 5;
const COMPONENTS: usize = 4;
const RGB_CHANNELS: usize = 3;
/// nvJPEG reports major, minor and patch components.
const VERSION_COMPONENTS: usize = 3;
const JPEG_MAGIC: &[u8] = &[0xff, 0xd8];
type Handle = *mut c_void;
type Create = unsafe extern "C" fn(i32, Handle, Handle, u32, *mut Handle) -> i32;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type CreateState = unsafe extern "C" fn(Handle, *mut Handle) -> i32;
type DecoderInfo = unsafe extern "C" fn(Handle, *mut u32, *mut u32) -> i32;
type Initialize = unsafe extern "C" fn(Handle, Handle, i32, i32, i32) -> i32;
type ParseHeader = unsafe extern "C" fn(Handle, *const u8, usize, Handle) -> i32;
type Supported = unsafe extern "C" fn(Handle, Handle, *mut i32) -> i32;
type Dimensions = unsafe extern "C" fn(Handle, *mut u32, *mut u32) -> i32;
type Property = unsafe extern "C" fn(i32, *mut i32) -> i32;
type Decode =
    unsafe extern "C" fn(Handle, Handle, *const *const u8, *const usize, *mut Image, Handle) -> i32;

#[repr(C)]
struct Image {
    channel: [*mut u8; COMPONENTS],
    pitch: [usize; COMPONENTS],
}

struct Api {
    _library: libloading::Library,
    create: Create,
    destroy: Destroy,
    state_create: CreateState,
    state_destroy: Destroy,
    stream_create: CreateState,
    stream_destroy: Destroy,
    hardware_info: DecoderInfo,
    initialize: Initialize,
    parse_header: ParseHeader,
    supported: Supported,
    dimensions: Dimensions,
    property: Property,
    decode: Decode,
}

#[expect(
    unsafe_code,
    reason = "The library names are toolkit sonames and the loaded library outlives its symbols"
)]
fn open_library() -> Option<libloading::Library> {
    for name in [
        "libnvjpeg.so.13",
        "libnvjpeg.so.12",
        "libnvjpeg.so.11",
        "libnvjpeg.so",
    ] {
        // SAFETY: these are standard nvJPEG sonames; Api keeps the library alive for all calls.
        if let Ok(library) = unsafe { libloading::Library::new(name) } {
            return Some(library);
        }
    }
    None
}

#[expect(
    unsafe_code,
    reason = "The loader resolves only documented nvJPEG functions with their declared C signatures"
)]
fn symbol<T: Copy>(library: &libloading::Library, name: &[u8]) -> Option<T> {
    // SAFETY: each caller supplies the exact function signature from nvjpeg.h.
    unsafe { library.get::<T>(name).ok().map(|value| *value) }
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| {
        let library = open_library()?;
        Some(Api {
            create: symbol(&library, b"nvjpegCreateEx\0")?,
            destroy: symbol(&library, b"nvjpegDestroy\0")?,
            state_create: symbol(&library, b"nvjpegJpegStateCreate\0")?,
            state_destroy: symbol(&library, b"nvjpegJpegStateDestroy\0")?,
            stream_create: symbol(&library, b"nvjpegJpegStreamCreate\0")?,
            stream_destroy: symbol(&library, b"nvjpegJpegStreamDestroy\0")?,
            hardware_info: symbol(&library, b"nvjpegGetHardwareDecoderInfo\0")?,
            initialize: symbol(&library, b"nvjpegDecodeBatchedInitialize\0")?,
            parse_header: symbol(&library, b"nvjpegJpegStreamParseHeader\0")?,
            supported: symbol(&library, b"nvjpegDecodeBatchedSupported\0")?,
            dimensions: symbol(&library, b"nvjpegJpegStreamGetFrameDimensions\0")?,
            property: symbol(&library, b"nvjpegGetProperty\0")?,
            decode: symbol(&library, b"nvjpegDecodeBatched\0")?,
            _library: library,
        })
    })
    .as_ref()
}

fn check(status: i32, operation: &str) -> Result<()> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::Backend,
            format!("nvJPEG {operation} failed: status {status}"),
        ))
    }
}

struct DecodeArgs {
    bytes: Option<Arc<Vec<u8>>>,
    pointers: [*const u8; 1],
    lengths: [usize; 1],
    outputs: [Image; 1],
}

#[derive(Clone, Copy)]
enum SessionKind {
    Probe,
    Device,
}

struct Session {
    api: &'static Api,
    context: Arc<CudaContext>,
    handle: Option<NonNull<c_void>>,
    state: Option<NonNull<c_void>>,
    jpeg: Option<NonNull<c_void>>,
    args: Option<Box<DecodeArgs>>,
    owner: ThreadId,
    healthy: bool,
}

#[expect(
    unsafe_code,
    clippy::non_send_fields_in_send_ty,
    reason = "Exclusive decoder leases transfer to the completion owner and retain device buffers through the fence"
)]
// SAFETY: a session is exclusively leased and can execute only on the thread that created it.
// The pool may move ownership to another thread for release, but checks the creator before use;
// every call and destruction binds the retained primary context. Pending bytes and descriptors
// stay in the session until its submitting stream has fenced.
unsafe impl Send for Session {}

impl Session {
    #[expect(
        unsafe_code,
        reason = "The nvJPEG handles are created and validated within their retained primary CUDA context"
    )]
    fn create(
        context: Arc<CudaContext>,
        api: &'static Api,
        kind: SessionKind,
    ) -> Result<Option<Self>> {
        context.bind_to_thread().map_err(device_error)?;
        let mut session = Self {
            api,
            context,
            handle: None,
            state: None,
            jpeg: None,
            args: None,
            owner: thread::current().id(),
            healthy: true,
        };
        let mut raw = null_mut();
        // SAFETY: allocator pointers are null, output points to a valid handle slot, and the
        // current context is the retained device's primary context.
        let result = unsafe {
            (api.create)(
                HARDWARE,
                null_mut(),
                null_mut(),
                UPSAMPLE_INTERPOLATION,
                &raw mut raw,
            )
        };
        if result == ARCH_MISMATCH || result == IMPLEMENTATION_NOT_SUPPORTED {
            return Ok(None);
        }
        check(result, "create")?;
        session.handle = NonNull::new(raw);
        let handle = session
            .handle
            .ok_or_else(|| Error::invariant("nvJPEG created a null handle"))?
            .as_ptr();
        let (mut engines, mut cores) = (0, 0);
        // SAFETY: the handle is live and the writable output slots are valid.
        let result = unsafe { (api.hardware_info)(handle, &raw mut engines, &raw mut cores) };
        if result == ARCH_MISMATCH || result == IMPLEMENTATION_NOT_SUPPORTED {
            return Ok(None);
        }
        check(result, "hardware query")?;
        if engines == 0 || cores == 0 {
            return Ok(None);
        }
        if matches!(kind, SessionKind::Device) {
            session.initialize_state(handle)?;
        }
        raw = null_mut();
        // SAFETY: the handle is live and the output points to a valid stream slot.
        let result = unsafe { (api.stream_create)(handle, &raw mut raw) };
        check(result, "JPEG stream create")?;
        session.jpeg = NonNull::new(raw);
        if session.jpeg.is_none() {
            return Err(Error::invariant("nvJPEG created a null JPEG stream"));
        }
        Ok(Some(session))
    }

    #[expect(
        unsafe_code,
        reason = "The device decoder state belongs to the exclusively leased nvJPEG handle"
    )]
    fn initialize_state(&mut self, handle: Handle) -> Result<()> {
        let mut raw = null_mut();
        // SAFETY: the handle is live and the output points to a valid state slot.
        let result = unsafe { (self.api.state_create)(handle, &raw mut raw) };
        check(result, "state create")?;
        self.state = NonNull::new(raw);
        let state = self
            .state
            .ok_or_else(|| Error::invariant("nvJPEG created a null state"))?
            .as_ptr();
        // SAFETY: both handles belong to this context; the fixed batch has one RGBI destination.
        let result = unsafe { (self.api.initialize)(handle, state, 1, 1, RGB_INTERLEAVED) };
        check(result, "initialize")
    }

    fn handles(&self) -> Result<(Handle, Handle)> {
        if self.owner != thread::current().id() {
            return Err(Error::invariant(
                "nvJPEG state used from a different thread",
            ));
        }
        let handle = self
            .handle
            .ok_or_else(|| Error::invariant("nvJPEG handle missing"))?;
        let jpeg = self
            .jpeg
            .ok_or_else(|| Error::invariant("nvJPEG stream missing"))?;
        Ok((handle.as_ptr(), jpeg.as_ptr()))
    }

    #[expect(
        unsafe_code,
        reason = "JPEG header inspection uses checked host slices and a session exclusively bound to the device"
    )]
    fn inspect(&mut self, bytes: &[u8]) -> Result<Option<[usize; 2]>> {
        if !bytes.starts_with(JPEG_MAGIC) {
            return Ok(None);
        }
        self.context.bind_to_thread().map_err(device_error)?;
        let (handle, jpeg) = self.handles()?;
        // SAFETY: bytes is readable for its exact length and both handles are live and exclusive.
        let result = unsafe { (self.api.parse_header)(handle, bytes.as_ptr(), bytes.len(), jpeg) };
        if [BAD_JPEG, JPEG_NOT_SUPPORTED, INCOMPLETE_BITSTREAM].contains(&result) {
            return Ok(None);
        }
        check(result, "parse header")?;
        let mut unsupported = 1;
        // SAFETY: the JPEG stream was parsed successfully; the output slot is valid.
        let result = unsafe { (self.api.supported)(handle, jpeg, &raw mut unsupported) };
        check(result, "support query")?;
        if unsupported != 0 {
            return Ok(None);
        }
        let (mut width, mut height) = (0, 0);
        // SAFETY: the parsed stream and both writable output slots are valid.
        let result = unsafe { (self.api.dimensions)(jpeg, &raw mut width, &raw mut height) };
        check(result, "dimensions")?;
        let height = usize::try_from(height).map_err(|_| Error::invalid("JPEG height overflow"))?;
        let width = usize::try_from(width).map_err(|_| Error::invalid("JPEG width overflow"))?;
        Ok(Some([height, width]))
    }

    #[expect(
        unsafe_code,
        reason = "The nvJPEG launch uses checked output geometry and retains all inputs through the CUDA fence"
    )]
    fn decode(
        &mut self,
        bytes: Arc<Vec<u8>>,
        output: u64,
        width: usize,
        stream: Handle,
    ) -> Result<()> {
        self.context.bind_to_thread().map_err(device_error)?;
        let (handle, _) = self.handles()?;
        let state = self
            .state
            .ok_or_else(|| Error::invariant("nvJPEG state missing"))?
            .as_ptr();
        let address =
            usize::try_from(output).map_err(|_| Error::invalid("JPEG output address overflow"))?;
        let pitch = width
            .checked_mul(RGB_CHANNELS)
            .ok_or_else(|| Error::invalid("JPEG pitch overflow"))?;
        let pointer = bytes.as_ptr();
        let length = bytes.len();
        let args = self.args.get_or_insert_with(|| {
            Box::new(DecodeArgs {
                bytes: None,
                pointers: [null()],
                lengths: [0],
                outputs: [Image {
                    channel: [null_mut(); COMPONENTS],
                    pitch: [0; COMPONENTS],
                }],
            })
        });
        args.bytes = Some(bytes);
        args.pointers[0] = pointer;
        args.lengths[0] = length;
        args.outputs[0].channel[0] = address as *mut u8;
        args.outputs[0].pitch[0] = pitch;
        // SAFETY: the header was rechecked against the output allocation, descriptors and byte
        // owner remain in the retained session until the supplied CUDA stream has completed.
        let result = unsafe {
            (self.api.decode)(
                handle,
                state,
                args.pointers.as_ptr(),
                args.lengths.as_ptr(),
                args.outputs.as_mut_ptr(),
                stream,
            )
        };
        if result != SUCCESS {
            self.healthy = false;
        }
        check(result, "decode")
    }
}

#[expect(
    unsafe_code,
    reason = "nvJPEG handles must be released in reverse order on their retained primary device context"
)]
impl Drop for Session {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind_to_thread() {
            tracing::error!(%error, "nvJPEG context unavailable while releasing handles");
            return;
        }
        for (name, handle, destroy) in [
            ("JPEG stream", self.jpeg.take(), self.api.stream_destroy),
            ("state", self.state.take(), self.api.state_destroy),
            ("handle", self.handle.take(), self.api.destroy),
        ] {
            if let Some(handle) = handle {
                // SAFETY: the session exclusively owns each live handle, uses the bound context,
                // and can only be returned to its pool after the device submission fences.
                let result = unsafe { destroy(handle.as_ptr()) };
                if result != SUCCESS {
                    tracing::error!(name, result, "nvJPEG handle release failed");
                }
            }
        }
    }
}

struct Pool {
    context: Arc<CudaContext>,
    api: &'static Api,
    kind: SessionKind,
    available: Mutex<HashMap<ThreadId, Vec<Session>>>,
}

pub(super) struct Decoder {
    probes: Arc<Pool>,
    devices: Arc<Pool>,
    identity: String,
}

/// Keeps CPU-only deployments independent of nvJPEG initialization.
pub(super) struct LazyDecoder {
    context: Arc<CudaContext>,
    target: Arch,
    decoder: OnceLock<Result<Option<Arc<Decoder>>>>,
    executor: OnceLock<Arc<dyn DeviceHostExecutor>>,
}

impl LazyDecoder {
    pub const fn new(context: Arc<CudaContext>, target: Arch) -> Self {
        Self {
            context,
            target,
            decoder: OnceLock::new(),
            executor: OnceLock::new(),
        }
    }

    pub fn get(&self) -> Result<Option<Arc<Decoder>>> {
        self.decoder
            .get_or_init(|| Decoder::load(self.target, Arc::clone(&self.context)))
            .clone()
    }

    pub fn inspect(&self, bytes: &[u8]) -> Result<Option<EncodedImageInfo>> {
        if !bytes.starts_with(JPEG_MAGIC) {
            return Ok(None);
        }
        self.get()?
            .map_or(Ok(None), |decoder| decoder.inspect(bytes))
    }

    pub fn set_host_executor(&self, executor: Arc<dyn DeviceHostExecutor>) -> Result<()> {
        self.executor
            .set(executor)
            .map_err(|_| Error::invariant("nvJPEG host executor is already installed"))
    }

    pub fn host_executor(&self) -> Result<Arc<dyn DeviceHostExecutor>> {
        self.executor
            .get()
            .cloned()
            .ok_or_else(|| Error::unsupported("nvJPEG requires a bounded host executor"))
    }
}

impl Decoder {
    #[expect(
        unsafe_code,
        reason = "Version properties are read into valid host integers from a loaded nvJPEG library"
    )]
    fn load(target: Arch, context: Arc<CudaContext>) -> Result<Option<Arc<Self>>> {
        let Some(api) = api() else { return Ok(None) };
        let Some(session) = Session::create(Arc::clone(&context), api, SessionKind::Probe)? else {
            return Ok(None);
        };
        let mut version = [0; VERSION_COMPONENTS];
        for (property, value) in version.iter_mut().enumerate() {
            let property =
                i32::try_from(property).map_err(|_| Error::invariant("nvJPEG version index"))?;
            // SAFETY: the property is a libraryPropertyType discriminant and value is writable.
            let result = unsafe { (api.property)(property, from_mut(value)) };
            check(result, "version")?;
        }
        let identity = format!(
            "nvjpeg-hardware-{}.{}.{}-interpolated-{:?}",
            version[0], version[1], version[2], target
        );
        Ok(Some(Arc::new(Self {
            probes: Arc::new(Pool {
                context: Arc::clone(&context),
                api,
                kind: SessionKind::Probe,
                available: Mutex::new(HashMap::from([(session.owner, vec![session])])),
            }),
            devices: Arc::new(Pool {
                context,
                api,
                kind: SessionKind::Device,
                available: Mutex::new(HashMap::new()),
            }),
            identity,
        })))
    }

    fn acquire(pool: &Arc<Pool>) -> Result<Arc<Lease>> {
        let session = {
            let mut available = pool
                .available
                .lock()
                .map_err(|_| Error::invariant("nvJPEG pool poisoned"))?;
            available
                .get_mut(&thread::current().id())
                .and_then(Vec::pop)
        };
        let session = match session {
            Some(session) => session,
            None => Session::create(Arc::clone(&pool.context), pool.api, pool.kind)?
                .ok_or_else(|| Error::unsupported("nvJPEG hardware decoder became unavailable"))?,
        };
        Ok(Arc::new(Lease {
            session: Mutex::new(Some(session)),
            pool: Arc::clone(pool),
        }))
    }

    pub fn inspect(&self, bytes: &[u8]) -> Result<Option<EncodedImageInfo>> {
        if !bytes.starts_with(JPEG_MAGIC) {
            return Ok(None);
        }
        let Some(source) = Self::acquire(&self.probes)?.inspect(bytes)? else {
            return Ok(None);
        };
        EncodedImageInfo::new(source, ImageEncoding::Jpeg, self.identity.clone()).map(Some)
    }

    pub fn prepare(&self, bytes: &[u8], expected: &EncodedImageInfo) -> Result<Arc<Lease>> {
        if expected.encoding() != ImageEncoding::Jpeg
            || expected.decoder_identity() != self.identity
        {
            return Err(Error::invalid("compressed image decoder identity mismatch"));
        }
        let lease = Self::acquire(&self.devices)?;
        if lease.inspect(bytes)? != Some(expected.source()) {
            return Err(Error::invalid(
                "compressed JPEG header or geometry mismatch",
            ));
        }
        Ok(lease)
    }
}

pub(super) struct Lease {
    session: Mutex<Option<Session>>,
    pool: Arc<Pool>,
}

impl Lease {
    fn inspect(&self, bytes: &[u8]) -> Result<Option<[usize; 2]>> {
        self.session
            .lock()
            .map_err(|_| Error::invariant("nvJPEG session poisoned"))?
            .as_mut()
            .ok_or_else(|| Error::invariant("nvJPEG session missing"))?
            .inspect(bytes)
    }

    pub fn decode(
        &self,
        bytes: Arc<Vec<u8>>,
        output: u64,
        width: usize,
        stream: Handle,
    ) -> Result<()> {
        self.session
            .lock()
            .map_err(|_| Error::invariant("nvJPEG session poisoned"))?
            .as_mut()
            .ok_or_else(|| Error::invariant("nvJPEG session missing"))?
            .decode(bytes, output, width, stream)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let Ok(slot) = self.session.get_mut() else {
            tracing::error!("nvJPEG session poisoned during release");
            return;
        };
        if let Some(mut session) = slot.take() {
            // The lease reaches this point only after the submission fence, so the input and
            // descriptor storage no longer need to stay in an otherwise idle pooled session.
            if let Some(args) = &mut session.args {
                args.bytes = None;
                args.pointers[0] = null();
                args.outputs[0].channel[0] = null_mut();
            }
            if session.healthy {
                if let Ok(mut pool) = self.pool.available.lock() {
                    pool.entry(session.owner).or_default().push(session);
                } else {
                    tracing::error!("nvJPEG pool poisoned during release");
                }
            }
        }
    }
}
