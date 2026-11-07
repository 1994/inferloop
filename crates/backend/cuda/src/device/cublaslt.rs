//! Minimal cuBLAS binding for the projection GEMMs.
//!
//! The library is opened on first use rather than linked, so a host without the toolkit still
//! builds and every other backend path is unaffected. Launches go through `cublasGemmEx` with the
//! handle in device pointer mode, which is what lets `alpha` and `beta` live in device memory: a
//! host scalar cannot be captured into a CUDA graph, and a captured step is how this engine runs.
//!
//! Layouts follow the row-major to column-major mapping a projection needs:
//! `out[m, n] = activations[m, k] * weights[n, k]^T`, so `A` is the activation with `OP_T`
//! (`lda = k`), `B` is the weight with `OP_N` (`ldb = k`), and `C` is the output with `ldc = m`.
use crate::device::CudaDevice;
use crate::device::device_error;
use crate::mlp::Matrix;
use crate::resident::tensor_buffer;
use crate::tuning::prefill::delegate_dense;
use infer_core::Result;
use infer_core::{Error, ErrorCode};
use infer_cuda_kernels::contract::{
    Bf16, Buffer, Foreign, Kernels, NARROW_GRANULE, Owner, Recording,
};
use infer_cuda_kernels::runtime::Allocation;
use infer_cuda_kernels::runtime::bf16;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Display;
use std::ptr::null_mut;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, OnceLock};

/// The driver's opaque stream handle, as the library ABI takes it.
type CUstream = *mut c_void;

type Handle = *mut c_void;
type CudaStream = CUstream;

const OP_N: i32 = 0;
const OP_T: i32 = 1;
const COMPUTE_32F: i32 = 68;
const DATATYPE_BF16: i32 = 14;
const DATATYPE_F32: i32 = 0;
const POINTER_MODE_DEVICE: i32 = 1;
const GEMM_DEFAULT: i32 = -1;
/// Extents of one GEMM, `[m, n, k]`, and its three device operands.
const GEMM_OPERANDS: usize = 3;

type CreateHandle = unsafe extern "C" fn(*mut Handle) -> i32;
type DestroyHandle = unsafe extern "C" fn(Handle) -> i32;
type SetPointerMode = unsafe extern "C" fn(Handle, i32) -> i32;
type SetStream = unsafe extern "C" fn(Handle, CudaStream) -> i32;
type GemmEx = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    i32,
    *const c_void,
    i32,
    i32,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
) -> i32;

struct Api {
    _library: libloading::Library,
    create: CreateHandle,
    destroy: DestroyHandle,
    set_pointer_mode: SetPointerMode,
    set_stream: SetStream,
    gemm: GemmEx,
}

/// Open the toolkit library, if this host has one.
#[expect(
    unsafe_code,
    reason = "Audited loader: opening the toolkit library the build already targets by its soname"
)]
fn open_library() -> Option<libloading::Library> {
    for name in ["libcublas.so.13", "libcublas.so"] {
        // SAFETY: dlopen of a toolkit soname; the library stays alive in `Api` for the process.
        if let Ok(library) = unsafe { libloading::Library::new(name) } {
            return Some(library);
        }
    }
    None
}

/// Resolve one symbol with its own type.
///
/// `Library::get::<T>` returns the symbol's address only when `T` is the function's own type:
/// asking for a `*const c_void` would read the first bytes of the function body instead.
#[expect(
    unsafe_code,
    reason = "Audited loader: resolving one documented cuBLAS entry point from a library that outlives every call"
)]
fn resolve<T: Copy>(library: &libloading::Library, name: &[u8]) -> Option<T> {
    // SAFETY: the caller passes the exact signature transcribed from cublas_api.h.
    unsafe { library.get::<T>(name).ok().map(|symbol| *symbol) }
}

/// Load every entry point, returning `None` if the toolkit is absent or older than expected.
fn load() -> Option<Api> {
    let library = open_library()?;
    macro_rules! take {
        ($name:literal, $ty:ty) => {
            resolve::<$ty>(&library, $name)?
        };
    }
    Some(Api {
        create: take!(b"cublasCreate_v2", CreateHandle),
        destroy: take!(b"cublasDestroy_v2", DestroyHandle),
        set_pointer_mode: take!(b"cublasSetPointerMode_v2", SetPointerMode),
        set_stream: take!(b"cublasSetStream_v2", SetStream),
        gemm: take!(b"cublasGemmEx", GemmEx),
        _library: library,
    })
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(load).as_ref()
}

/// Whether cuBLAS can be used at all on this host.
pub fn available() -> bool {
    api().is_some()
}

/// One cuBLAS handle with the `1.0`/`0.0` device scalars every launch needs.
pub struct Context {
    ordinal: usize,
    stream: CudaStream,
    handle: Handle,
    alpha: Allocation<f32>,
    beta: Allocation<f32>,
}

#[expect(
    unsafe_code,
    reason = "a cuBLAS handle is device-scoped, not thread-scoped, and is only used through a launched GEMM"
)]
// SAFETY: a cuBLAS handle is not thread-affine; every use is a GEMM on the owning device's stream
// and the two scalars are plain device allocations owned by this value.
unsafe impl Send for Context {}
#[expect(
    unsafe_code,
    reason = "a cuBLAS handle is device-scoped, not thread-scoped, and is only used through a launched GEMM"
)]
// SAFETY: nothing in the context mutates through a shared reference; the handle is only passed to
// cuBLAS, which serializes its own internal state.
unsafe impl Sync for Context {}

/// Borrow the context for this device and stream, creating it on first use.
///
/// A cuBLAS handle belongs to the CUDA device it was created on and launches on the stream it was
/// bound to, so one handle cannot serve a second device instance. The cache is keyed by both, and
/// every launch validates that the stream it is handed is the one the handle was built for.
pub fn context_for(device: &CudaDevice) -> Option<Arc<Context>> {
    type Cache = Mutex<HashMap<(usize, usize), Arc<Context>>>;
    static CONTEXTS: OnceLock<Cache> = OnceLock::new();
    let key = (device.ordinal(), device.tile().stream_id());
    let cache = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let existing = cache.lock().ok()?.get(&key).cloned();
    if let Some(existing) = existing {
        return Some(existing);
    }
    let built = Arc::new(build(device)?);
    cache.lock().ok()?.insert(key, Arc::clone(&built));
    Some(built)
}

/// Destroy a handle, reporting nothing: every caller is already on an error path.
#[expect(
    unsafe_code,
    reason = "Audited teardown: the handle was created by this module and is destroyed exactly once"
)]
fn destroy(api: &Api, handle: Handle) {
    // SAFETY: the handle is owned by the context this module builds.
    unsafe {
        (api.destroy)(handle);
    }
}

#[expect(
    unsafe_code,
    reason = "Audited handle creation: one cuBLAS handle per process, packed with the device scalars every launch reads"
)]
fn build(device: &CudaDevice) -> Option<Context> {
    let api = api()?;
    // cuBLAS creates its handle on the device current for this thread, so bind before creating it.
    device.tile().bind_to_thread().ok()?;
    let mut handle: Handle = null_mut();
    // SAFETY: the handle is created here and owned by the returned context.
    let created = unsafe { (api.create)(&raw mut handle) };
    if created != 0 {
        return None;
    }
    // SAFETY: the handle is live; device pointer mode is what makes the scalars below legal.
    let mode = unsafe { (api.set_pointer_mode)(handle, POINTER_MODE_DEVICE) };
    if mode != 0 {
        destroy(api, handle);
        return None;
    }
    // SAFETY: the handle is live and the stream belongs to this device.
    let bound = unsafe { (api.set_stream)(handle, device.tile().raw_stream()) };
    if bound != 0 {
        destroy(api, handle);
        return None;
    }
    let runtime = device.runtime();
    let scalar = |value: f32| runtime.upload(vec![value]).ok();
    let (Some(alpha), Some(beta)) = (scalar(1.0), scalar(0.0)) else {
        destroy(api, handle);
        return None;
    };
    Some(Context {
        ordinal: device.ordinal(),
        stream: device.tile().raw_stream(),
        handle,
        alpha,
        beta,
    })
}

#[expect(
    unsafe_code,
    reason = "Audited launch: the operands are live device tensors whose extents the caller checked against each other"
)]
unsafe fn launch(context: &Context, product: &Product) -> StdResult<(), i32> {
    let Some(api) = api() else {
        return Err(-1);
    };
    let Product {
        m,
        n,
        k,
        pointers: [activations, weights, out],
    } = *product;
    let (activations, weights) = (activations as *const c_void, weights as *const c_void);
    let out = out as *mut c_void;
    // SAFETY: `alpha` and `beta` are device scalars owned by the context, the operands are the
    // caller's live buffers, and the handle's stream belongs to the same device.
    unsafe {
        let alpha = context.alpha.buffer().address() as *const c_void;
        let beta = context.beta.buffer().address() as *const c_void;
        // `out` is row-major `[m, n]`, which is column-major `[n, m]` with `ld = n`, so cuBLAS
        // computes the transposed product: `op(A) = weights`, `op(B) = activations`.
        let status = (api.gemm)(
            context.handle,
            OP_T,
            OP_N,
            n,
            m,
            k,
            alpha,
            weights,
            DATATYPE_BF16,
            k,
            activations,
            DATATYPE_BF16,
            k,
            beta,
            out,
            DATATYPE_F32,
            n,
            COMPUTE_32F,
            GEMM_DEFAULT,
        );
        if status == 0 { Ok(()) } else { Err(status) }
    }
}

/// `out[m, n] = activations[m, k] * weights[n, k]^T`, with extents checked against the buffers.
///
/// The extents are explicit rather than read from ranks so a flat activation buffer works: the
/// cast that produces it writes flat tiles.
#[derive(Clone, Copy)]
struct Product {
    m: i32,
    n: i32,
    k: i32,
    pointers: [u64; GEMM_OPERANDS],
}

impl Product {
    fn new(
        [m, n, k]: [usize; GEMM_OPERANDS],
        activations: Buffer<'_, Bf16>,
        weights: Buffer<'_, Bf16>,
        out: Buffer<'_, f32>,
    ) -> Result<Self> {
        let extent =
            |value: usize| i32::try_from(value).map_err(|_| Error::invalid("cuBLAS extent"));
        let elements = |rows: usize, columns: usize| rows.checked_mul(columns);
        if Some(activations.len()) != elements(m, k)
            || Some(weights.len()) != elements(n, k)
            || Some(out.len()) != elements(m, n)
        {
            return Err(Error::invalid("cuBLAS operand shape"));
        }
        Ok(Self {
            m: extent(m)?,
            n: extent(n)?,
            k: extent(k)?,
            pointers: [activations.address(), weights.address(), out.address()],
        })
    }
}

/// Run `out[m, n] = activations[m, k] * weights[n, k]^T` in bf16 with f32 accumulation, with
/// `extents` `[m, n, k]`.
///
/// # Errors
/// Rejects operands whose extents disagree, or reports the driver's status.
#[expect(
    unsafe_code,
    reason = "Audited launch: the three buffers outlive the call and their extents are checked against each other"
)]
pub fn gemm_bf16(
    device: &CudaDevice,
    extents: [usize; GEMM_OPERANDS],
    activations: Buffer<'_, Bf16>,
    weights: Buffer<'_, Bf16>,
    out: Buffer<'_, f32>,
) -> Result<()> {
    let context = context_for(device)
        .ok_or_else(|| Error::invalid("cuBLAS handle unavailable for this device"))?;
    let product = Product::new(extents, activations, weights, out)?;
    // SAFETY: the three buffers are live for the call and their extents were just checked.
    let status = unsafe { launch(&context, &product) };
    status.map_err(|code| Error::new(ErrorCode::Backend, format!("cublasGemmEx status {code}")))
}

/// Vendor GEMM support for one program: the handle and the BF16 activation scratches.
///
/// The scratch is keyed by element count and allocated the first time a shape needs it, which
/// happens while a graph is being built and never while one is being captured.
pub struct Support {
    device: CudaDevice,
    context: Option<Arc<Context>>,
    activations: HashMap<usize, Allocation<bf16>>,
}

impl Support {
    /// Large BF16 projections use the tuning policy and the available vendor library.
    pub fn new(device: &CudaDevice) -> Self {
        Self::new_with(device, true)
    }

    /// Build the support with delegation decided by the caller, for benchmarks and tests that
    /// must not depend on the environment.
    pub fn new_with(device: &CudaDevice, enabled: bool) -> Self {
        Self {
            device: device.clone(),
            context: enabled.then(|| context_for(device)).flatten(),
            activations: HashMap::new(),
        }
    }

    /// Whether this program delegates at all.
    pub const fn is_enabled(&self) -> bool {
        self.context.is_some()
    }

    /// The context this support launches through, which a recorded op also needs. `None` when
    /// delegation is off, so callers check [`Self::is_enabled`] or handle the absence.
    #[must_use]
    pub const fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }

    /// Run one GEMM per shape on the device stream before any graph captures it.
    ///
    /// cuBLAS chooses workspace and algorithm the first time it sees a configuration, and that
    /// host-side work is rejected while a stream is capturing, so every shape a graph will record
    /// has to be exercised here first.
    #[expect(
        unsafe_code,
        reason = "Audited warmup launch: both operands are allocated here with the extents the launch is told about"
    )]
    pub fn warm(&mut self, m: usize, weight: &Matrix<bf16>, n: usize, k: usize) -> Result<()> {
        let Some(context) = self.context.clone() else {
            return Ok(());
        };
        if !delegate_dense(m, n, k) {
            return Ok(());
        }
        let scratch = self.scratch(m.saturating_mul(k))?;
        let out = self
            .device
            .runtime()
            .zeros::<f32>(m.saturating_mul(n))
            .map_err(internal)?;
        let Ok(product) = Product::new(
            [m, n, k],
            scratch.operand(),
            tensor_buffer(weight)?,
            out.operand(),
        ) else {
            return Ok(());
        };
        // SAFETY: the operands are live allocations whose extents were checked, and the stream
        // belongs to this device.
        let status = unsafe { launch(&context, &product) };
        status.map_err(|code| device_error(format!("cublasGemmEx warmup status {code}")))?;
        self.device
            .reclaim_barrier()
            .map_err(|error| device_error(error.to_string()))?;
        Ok(())
    }

    /// Record `out[m, n] = input[m, k] * weight[n, k]^T` through cuBLAS when the shape is one the
    /// measurement favours, returning whether it did.
    ///
    /// # Errors
    /// Propagates launch failures; a shape that is not worth delegating is not an error.
    pub fn record_dense(
        &mut self,
        (kernels, scope): (&Kernels, &Recording<'_>),
        input: Buffer<'_, f32>,
        weight: &Matrix<bf16>,
        out: Buffer<'_, f32>,
        (m, n, k): (usize, usize, usize),
    ) -> Result<bool> {
        let Some(context) = self.context.clone() else {
            return Ok(false);
        };
        let activation_elements = m.saturating_mul(k);
        if !delegate_dense(m, n, k)
            || !activation_elements.is_multiple_of(NARROW_GRANULE)
            || input.len() != activation_elements
        {
            return Ok(false);
        }
        let scratch = self.scratch(activation_elements)?;
        let recording = scope;
        kernels
            .projection
            .narrow_bf16(recording, scratch.operand(), input)
            .map_err(|error| device_error(error.to_string()))?;
        let weight = tensor_buffer(weight)?;
        let op =
            GemmBf16::new(context, [m, n, k], scratch.operand(), weight, out).map_err(internal)?;
        // The cuBLAS node is consumed during capture, so the graph holds what it reads instead.
        let keep = [Some(scratch.operand()), Some(weight)]
            .into_iter()
            .flatten()
            .filter_map(Buffer::owner)
            .chain(out.owner())
            .cloned()
            .collect::<Vec<Owner>>();
        recording
            .foreign(op, keep)
            .map_err(|error| device_error(error.to_string()))?;
        Ok(true)
    }

    /// The BF16 activation scratch of `elements`, allocated the first time a shape needs it.
    fn scratch(&mut self, elements: usize) -> Result<Allocation<bf16>> {
        if let Some(scratch) = self.activations.get(&elements) {
            return Ok(scratch.clone());
        }
        let scratch = self
            .device
            .runtime()
            .zeros::<bf16>(elements)
            .map_err(internal)?;
        self.activations.insert(elements, scratch.clone());
        Ok(scratch)
    }
}

fn internal(error: impl Display) -> Error {
    device_error(error.to_string())
}

/// A bf16 projection that can be recorded into a captured graph.
///
/// The launch takes `alpha` and `beta` from device memory: anything read from the host while the
/// stream captures would be frozen into the graph.
pub struct GemmBf16 {
    context: Arc<Context>,
    product: Product,
}

impl GemmBf16 {
    /// Describe `out[m, n] = activations[m, k] * weights[n, k]^T` for `extents` `[m, n, k]`.
    ///
    /// # Errors
    /// Rejects operands whose element counts disagree with the extents.
    pub fn new(
        context: Arc<Context>,
        extents: [usize; GEMM_OPERANDS],
        activations: Buffer<'_, Bf16>,
        weights: Buffer<'_, Bf16>,
        out: Buffer<'_, f32>,
    ) -> Result<Self> {
        Ok(Self {
            context,
            product: Product::new(extents, activations, weights, out)?,
        })
    }
}

#[expect(
    unsafe_code,
    reason = "Audited in-graph launch: the recording keeps the operands alive and the stream is the one the handle was bound to"
)]
impl Foreign for GemmBf16 {
    fn enqueue(self, stream: *mut c_void) -> Result<()> {
        // A handle belongs to the device it was created on and launches on the stream it was bound
        // to. Recording it into a graph on another stream would place the GEMM outside that graph's
        // dependencies, so refuse rather than launch into the wrong stream.
        if stream != self.context.stream.cast() {
            return Err(Error::invalid(format!(
                "cuBLAS projection for device {} was built for a different stream than this context",
                self.context.ordinal
            )));
        }
        // SAFETY: the operand extents were checked in `new`, the recording keeps them alive, and
        // the stream matches the one the handle was created for.
        let status = unsafe { launch(&self.context, &self.product) };
        status.map_err(|code| Error::new(ErrorCode::Backend, format!("cublasGemmEx status {code}")))
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/device_cublaslt.rs"]
mod tests;
