//! FP8 channel projections through cuBLASLt with the existing quantization.
//! Scalar matrix scales stay one; the exact per-token/per-channel factors are applied afterwards.
//! Descriptors and zero-workspace algorithms are warmed before graph capture. Native kernels
//! remain the fallback for an absent library, unsupported shape or failed exact tuning gate.
use super::CudaDevice;
use super::library_names::CUBLAS_LT;
use cuda_core::f8e4m3fn;
use cuda_core::sys::CUcontext;
use cuda_core::sys::CUdeviceptr;
use cuda_core::sys::CUmemoryPool;
use cuda_core::sys::CUmemorytype;
use cuda_core::sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
use cuda_core::sys::CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_CONTEXT;
use cuda_core::sys::CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
use cuda_core::sys::CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_POINTER;
use cuda_core::sys::CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
use cuda_core::sys::CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE;
use cuda_core::sys::CUstream;
use cuda_core::sys::cuCtxGetCurrent;
use cuda_core::sys::cuPointerGetAttribute;
use cuda_core::sys::cuStreamGetCtx;
use cuda_core::sys::cudaError_enum_CUDA_SUCCESS;
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::{
    device_future::DeviceFuture,
    device_operation::{DeviceOp, ExecutionContext, GraphNode},
    error::DeviceError,
};
use cutile::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use std::ptr::null_mut;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    sync::{Arc, Mutex, OnceLock},
};

type Handle = *mut c_void;
type Stream = CUstream;
const SUCCESS: i32 = 0;
const OP_N: i32 = 0;
const OP_T: i32 = 1;
const COMPUTE_F32: i32 = 68;
const TYPE_F32: i32 = 0;
const TYPE_E4M3: i32 = 28;
const TRANSA: i32 = 3;
const TRANSB: i32 = 4;
const FAST_ACCUM: i32 = 25;
const A_SCALE_MODE: i32 = 31;
const B_SCALE_MODE: i32 = 32;
const SCALAR_F32: i32 = 0;
const ALGO_ID: i32 = 0;
const ALGO_TILE: i32 = 1;
const ALGO_SPLIT_K: i32 = 2;
const ALGO_REDUCTION: i32 = 3;
const ALGO_SWIZZLE: i32 = 4;
const ALGO_CUSTOM: i32 = 5;
const ALGO_STAGES: i32 = 6;
const ALGO_INNER: i32 = 7;
const ALGO_CLUSTER: i32 = 8;
const MAX_WORKSPACE: i32 = 1;
// cublasLtMatmulPreferenceAttributes_t: uint32 minimum A/B/C/D pointer alignments.
const MIN_ALIGNMENTS: [i32; 4] = [5, 6, 7, 8];
const ALIGNMENT: u32 = 256;
const HEURISTIC_COUNT: usize = 8;
const ALGO_WORDS: usize = 8;
const HEURISTIC_RESERVED: usize = 4;
const MATRIX_LAYOUTS: usize = 4;
const GEMM_RANK: usize = 3;
const SCALE_TILE: [usize; 2] = [1, 256];

/// Transcribed from CUDA 13.4 cublasLt.h: `cublasLtMatmulAlgo_t` has eight uint64 words.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Algo {
    data: [u64; ALGO_WORDS],
}
/// Transcribed from `cublasLtMatmulHeuristicResult_t`, including the four reserved ints.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Heuristic {
    algo: Algo,
    workspace: usize,
    state: i32,
    waves: f32,
    reserved: [i32; HEURISTIC_RESERVED],
}
const _: [(); 64] = [(); size_of::<Algo>()];
const _: [(); 96] = [(); size_of::<Heuristic>()];

type Create = unsafe extern "C" fn(*mut Handle) -> i32;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type DescCreate = unsafe extern "C" fn(*mut Handle, i32, i32) -> i32;
type SetAttribute = unsafe extern "C" fn(Handle, i32, *const c_void, usize) -> i32;
type LayoutCreate = unsafe extern "C" fn(*mut Handle, i32, u64, u64, i64) -> i32;
type Query = unsafe extern "C" fn(
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    i32,
    *mut Heuristic,
    *mut i32,
) -> i32;
type Matmul = unsafe extern "C" fn(
    Handle,
    Handle,
    *const c_void,
    *const c_void,
    Handle,
    *const c_void,
    Handle,
    *const c_void,
    *const c_void,
    Handle,
    *mut c_void,
    Handle,
    *const Algo,
    *mut c_void,
    usize,
    Stream,
) -> i32;

// Official cublasLt.h signatures. Opaque Algo bytes remain process-local only.
type Version = unsafe extern "C" fn() -> usize;
type AlgoGet = unsafe extern "C" fn(*const Algo, i32, *mut c_void, usize, *mut usize) -> i32;
type AlgoSet = unsafe extern "C" fn(*mut Algo, i32, *const c_void, usize) -> i32;
type AlgoInit = unsafe extern "C" fn(Handle, i32, i32, i32, i32, i32, i32, i32, *mut Algo) -> i32;
type AlgoCheck = unsafe extern "C" fn(
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    *const Algo,
    *mut Heuristic,
) -> i32;
/// SDK algorithm attributes use i32 for ID/split-K, u32 for tile/reduction/swizzle/
/// custom/stages and u16 for inner/cluster shape. Never serialize the opaque Algo ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub id: i32,
    pub tile: u32,
    pub split_k: i32,
    pub reduction: u32,
    pub swizzle: u32,
    pub custom: u32,
    pub stages: u32,
    pub inner: u16,
    pub cluster: u16,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LibraryIdentity {
    pub lt_version: usize,
    pub cudart_version: usize,
}
struct Api {
    _library: libloading::Library,
    create: Create,
    destroy: Destroy,
    desc_create: DescCreate,
    desc_destroy: Destroy,
    desc_set: SetAttribute,
    layout_create: LayoutCreate,
    layout_destroy: Destroy,
    pref_create: Create,
    pref_destroy: Destroy,
    pref_set: SetAttribute,
    query: Query,
    matmul: Matmul,
    version: Version,
    cudart_version: Version,
    algo_get: AlgoGet,
    algo_set: AlgoSet,
    algo_init: AlgoInit,
    algo_check: AlgoCheck,
}
#[expect(
    unsafe_code,
    reason = "Audited loader resolves documented CUDA 13 Lt signatures and keeps the library alive"
)]
fn load() -> Option<Api> {
    let library = CUBLAS_LT
        .into_iter()
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        .find_map(|name| unsafe { libloading::Library::new(name).ok() })?;
    macro_rules! take {
        ($name:literal, $ty:ty) => {
            // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
            unsafe { *library.get::<$ty>($name).ok()? }
        };
    }
    Some(Api {
        create: take!(b"cublasLtCreate", Create),
        destroy: take!(b"cublasLtDestroy", Destroy),
        desc_create: take!(b"cublasLtMatmulDescCreate", DescCreate),
        desc_destroy: take!(b"cublasLtMatmulDescDestroy", Destroy),
        desc_set: take!(b"cublasLtMatmulDescSetAttribute", SetAttribute),
        layout_create: take!(b"cublasLtMatrixLayoutCreate", LayoutCreate),
        layout_destroy: take!(b"cublasLtMatrixLayoutDestroy", Destroy),
        pref_create: take!(b"cublasLtMatmulPreferenceCreate", Create),
        pref_destroy: take!(b"cublasLtMatmulPreferenceDestroy", Destroy),
        pref_set: take!(b"cublasLtMatmulPreferenceSetAttribute", SetAttribute),
        query: take!(b"cublasLtMatmulAlgoGetHeuristic", Query),
        matmul: take!(b"cublasLtMatmul", Matmul),
        version: take!(b"cublasLtGetVersion", Version),
        cudart_version: take!(b"cublasLtGetCudartVersion", Version),
        algo_get: take!(b"cublasLtMatmulAlgoConfigGetAttribute", AlgoGet),
        algo_set: take!(b"cublasLtMatmulAlgoConfigSetAttribute", AlgoSet),
        algo_init: take!(b"cublasLtMatmulAlgoInit", AlgoInit),
        algo_check: take!(b"cublasLtMatmulAlgoCheck", AlgoCheck),
        _library: library,
    })
}
fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(load).as_ref()
}
fn error(message: impl Into<String>) -> DeviceError {
    DeviceError::Internal(message.into())
}
fn status(code: i32, operation: &str) -> Result<(), DeviceError> {
    if code == SUCCESS {
        Ok(())
    } else {
        Err(error(format!("cuBLASLt {operation} status {code}")))
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
}
impl Shape {
    fn from(
        input: &Tensor<f8e4m3fn>,
        weight: &Tensor<f8e4m3fn>,
        out: &Tensor<f32>,
    ) -> Result<Self, DeviceError> {
        let [m, k] = input.shape() else {
            return Err(error("vendor FP8 input rank"));
        };
        let [n, wk] = weight.shape() else {
            return Err(error("vendor FP8 weight rank"));
        };
        let [om, on] = out.shape() else {
            return Err(error("vendor FP8 output rank"));
        };
        if m != om || n != on || k != wk {
            return Err(error("vendor FP8 operand extents"));
        }
        let shape = Self {
            m: usize::try_from(*m).map_err(|e| error(e.to_string()))?,
            n: usize::try_from(*n).map_err(|e| error(e.to_string()))?,
            k: usize::try_from(*k).map_err(|e| error(e.to_string()))?,
        };
        if shape.m == 0 || shape.n == 0 || shape.k == 0 {
            return Err(error("vendor FP8 zero extent"));
        }
        Ok(shape)
    }
}
/// Lt layouts below represent contiguous row-major matrices only. Pin a single alignment
/// class in the preference and guard every pointer at record time, so a shape warmed on a
/// 256-byte allocation cannot be reused accidentally on a less-aligned arena row alias.
fn compatible(input: &Tensor<f8e4m3fn>, weight: &Tensor<f8e4m3fn>, out: &Tensor<f32>) -> bool {
    input.is_contiguous()
        && weight.is_contiguous()
        && out.is_contiguous()
        && [
            input.device_pointer().cu_deviceptr(),
            weight.device_pointer().cu_deviceptr(),
            out.device_pointer().cu_deviceptr(),
        ]
        .into_iter()
        .all(|pointer| pointer.is_multiple_of(u64::from(ALIGNMENT)))
}
struct Context {
    api: &'static Api,
    device: CudaDevice,
    handle: Handle,
    ordinal: usize,
    stream: Stream,
    primary: usize,
    calls: Mutex<()>,
    #[cfg(test)]
    heuristic_queries: AtomicUsize,
}
#[expect(
    unsafe_code,
    reason = "An Lt handle and its CUDA stream are device-scoped; calls are mutex serialized and stream validated"
)]
// SAFETY: The device is lifetime-owned; handle calls are serialized, and every launch binds and checks its context/stream.
unsafe impl Send for Context {}
#[expect(
    unsafe_code,
    reason = "Immutable device context metadata; only Lt entry points touch the handle under the calls mutex"
)]
// SAFETY: The device is lifetime-owned; handle calls are serialized, and every launch binds and checks its context/stream.
unsafe impl Sync for Context {}
impl Drop for Context {
    #[expect(
        unsafe_code,
        reason = "Audited teardown destroys the uniquely owned Lt handle after captured program use ends"
    )]
    fn drop(&mut self) {
        let _ = self.device.stream.device().bind_to_thread();
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        unsafe {
            (self.api.destroy)(self.handle);
        }
    }
}
#[expect(
    unsafe_code,
    reason = "Audited creation binds the owning device and queries the existing stream context without launching kernels"
)]
fn context(device: &CudaDevice) -> Option<Arc<Context>> {
    if !device.target().supports_compute(infer_ir::DType::Fp8E4M3) {
        return None;
    }
    let api = api()?;
    device.stream.device().bind_to_thread().ok()?;
    let stream = device.stream.cu_stream();
    let mut primary = null_mut();
    // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
    if unsafe { cuStreamGetCtx(stream, &raw mut primary) } != cudaError_enum_CUDA_SUCCESS {
        return None;
    }
    let mut handle = null_mut();
    // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
    if unsafe { (api.create)(&raw mut handle) } != SUCCESS {
        return None;
    }
    Some(Arc::new(Context {
        api,
        device: device.clone(),
        handle,
        ordinal: device.ordinal(),
        stream,
        primary: primary as usize,
        calls: Mutex::new(()),
        #[cfg(test)]
        heuristic_queries: AtomicUsize::new(0),
    }))
}
#[expect(
    unsafe_code,
    reason = "Audited metadata query rejects operands allocated in a foreign CUDA context before the Lt launch"
)]
fn same_context(context: &Context, pointer: CUdeviceptr) -> Result<(), DeviceError> {
    let mut owner: CUcontext = null_mut();
    // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
    let code = unsafe {
        cuPointerGetAttribute(
            (&raw mut owner).cast::<c_void>(),
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_CONTEXT,
            pointer,
        )
    };
    if code == cudaError_enum_CUDA_SUCCESS
        && (owner as usize == context.primary
            || (owner.is_null() && same_device_pool(context, pointer)))
    {
        return Ok(());
    }
    Err(error(format!(
        "vendor FP8 operand is outside device {} context: context_status={code}, owner={owner:p}, expected=0x{:x}, {}",
        context.ordinal,
        context.primary,
        pointer_diagnostics(context, pointer)
    )))
}
/// CUDA Programming Guide 4.3.5.4 specifies that stream-ordered pool allocations have a
/// successful CONTEXT query with a null owner. Admit that documented allocation kind only
/// after proving a device allocation on this ordinal, visible through the current stream's
/// owning context, and associated with a real pool. A null owner alone never grants access.
/// <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html#pointer-attributes>
#[expect(
    unsafe_code,
    reason = "Audited CUDA header attribute-specific types prove same-device pool allocation accessibility before Lt use"
)]
fn same_device_pool(context: &Context, pointer: CUdeviceptr) -> bool {
    let mut current: CUcontext = null_mut();
    // SAFETY: Each output is a live stack slot with its cuda.h attribute-specific type.
    if unsafe { cuCtxGetCurrent(&raw mut current) } != cudaError_enum_CUDA_SUCCESS
        || current as usize != context.primary
    {
        return false;
    }
    let mut ordinal = -1_i32;
    let mut memory_type: CUmemorytype = 0;
    let mut accessible: CUdeviceptr = 0;
    let mut pool: CUmemoryPool = null_mut();
    for (attribute, destination) in [
        (
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            (&raw mut ordinal).cast::<c_void>(),
        ),
        (
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            (&raw mut memory_type).cast::<c_void>(),
        ),
        (
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
            (&raw mut accessible).cast::<c_void>(),
        ),
        (
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
            (&raw mut pool).cast::<c_void>(),
        ),
    ] {
        // SAFETY: Destinations match the exact signed-int, enum, CUdeviceptr and pool ABI.
        if unsafe { cuPointerGetAttribute(destination, attribute, pointer) }
            != cudaError_enum_CUDA_SUCCESS
        {
            return false;
        }
    }
    usize::try_from(ordinal).is_ok_and(|ordinal| ordinal == context.ordinal)
        && memory_type == CUmemorytype_enum_CU_MEMORYTYPE_DEVICE
        && accessible == pointer
        && !pool.is_null()
}
/// Failure-only metadata: do not infer allocation ownership from a null context.
#[expect(
    unsafe_code,
    reason = "Audited failure diagnostics use CUDA header attribute-specific output types; these queries neither allocate nor launch GPU work"
)]
fn pointer_diagnostics(context: &Context, pointer: CUdeviceptr) -> String {
    let mut current: CUcontext = null_mut();
    // SAFETY: current points to one live CUcontext-sized stack slot.
    let current_status = unsafe { cuCtxGetCurrent(&raw mut current) };
    let mut ordinal = -1_i32;
    // SAFETY: DEVICE_ORDINAL writes one signed int as specified by cuda.h.
    let ordinal_status = unsafe {
        cuPointerGetAttribute(
            (&raw mut ordinal).cast::<c_void>(),
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            pointer,
        )
    };
    let mut memory_type: CUmemorytype = 0;
    // SAFETY: MEMORY_TYPE writes one CUmemorytype/unsigned-int enum as specified by cuda.h.
    let memory_status = unsafe {
        cuPointerGetAttribute(
            (&raw mut memory_type).cast::<c_void>(),
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            pointer,
        )
    };
    let mut accessible: CUdeviceptr = 0;
    // SAFETY: DEVICE_POINTER writes one CUdeviceptr, valid for the current CUDA context.
    let accessible_status = unsafe {
        cuPointerGetAttribute(
            (&raw mut accessible).cast::<c_void>(),
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
            pointer,
        )
    };
    let mut pool: CUmemoryPool = null_mut();
    // SAFETY: MEMPOOL_HANDLE writes one CUmemoryPool opaque pointer as specified by cuda.h.
    let pool_status = unsafe {
        cuPointerGetAttribute(
            (&raw mut pool).cast::<c_void>(),
            CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
            pointer,
        )
    };
    format!(
        "pointer=0x{pointer:x}, stream={:p}, device_context={:p}, current_status={current_status}, current={current:p}, ordinal_status={ordinal_status}, ordinal={ordinal}, memory_status={memory_status}, memory_type={memory_type}, accessible_status={accessible_status}, accessible=0x{accessible:x}, pool_status={pool_status}, pool={pool:p}",
        context.stream,
        context.device.stream.device().cu_ctx()
    )
}
struct Descriptors {
    api: &'static Api,
    desc: Handle,
    layouts: [Handle; MATRIX_LAYOUTS],
}
#[expect(
    unsafe_code,
    reason = "Descriptors are immutable after warmup; their only use is a serialized launch on the owning context"
)]
// SAFETY: Descriptors are immutable after construction; Arc owners keep them live while serialized handle calls borrow them.
unsafe impl Send for Descriptors {}
#[expect(
    unsafe_code,
    reason = "Descriptors and opaque algorithm bytes stay immutable throughout all captured replays"
)]
// SAFETY: Descriptors are immutable after construction; Arc owners keep them live while serialized handle calls borrow them.
unsafe impl Sync for Descriptors {}
impl Drop for Descriptors {
    #[expect(
        unsafe_code,
        reason = "Audited teardown destroys each created descriptor exactly once"
    )]
    fn drop(&mut self) {
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        for layout in self.layouts {
            if !layout.is_null() {
                // SAFETY: Each nonnull layout was created by this API and is destroyed exactly once.
                unsafe {
                    (self.api.layout_destroy)(layout);
                }
            }
        }
        if !self.desc.is_null() {
            // SAFETY: The descriptor was created by this API and no graph retains this final owner.
            unsafe {
                (self.api.desc_destroy)(self.desc);
            }
        }
    }
}
/// Diagnostic/tuning graph owners: captures are destroyed before this owner field.
pub struct CaptureOwner {
    _context: Arc<Context>,
    _prepared: Arc<Prepared>,
}
struct Prepared {
    descriptors: Arc<Descriptors>,
    algo: Algo,
}
impl Deref for Prepared {
    type Target = Descriptors;
    fn deref(&self) -> &Descriptors {
        &self.descriptors
    }
}
#[expect(
    unsafe_code,
    reason = "SDK typed algorithm getters copy scalar attributes into ABI-matching live stack slots"
)]
fn algorithm_config(api: &Api, algo: &Algo) -> Result<AlgorithmConfig, DeviceError> {
    fn get<T: Default>(api: &Api, algo: &Algo, attribute: i32) -> Result<T, DeviceError> {
        let mut value = T::default();
        let mut written = 0;
        // SAFETY: Attribute callers provide the exact cublasLt.h scalar type and length.
        status(
            // SAFETY: This single SDK call uses live lifetime-owned handles and pointers with the declared CUDA header ABI.
            unsafe {
                (api.algo_get)(
                    algo,
                    attribute,
                    (&raw mut value).cast(),
                    size_of::<T>(),
                    &raw mut written,
                )
            },
            "algorithm get",
        )?;
        if written != size_of::<T>() {
            return Err(error("Lt algorithm attribute size"));
        }
        Ok(value)
    }
    Ok(AlgorithmConfig {
        id: get(api, algo, ALGO_ID)?,
        tile: get(api, algo, ALGO_TILE)?,
        split_k: get(api, algo, ALGO_SPLIT_K)?,
        reduction: get(api, algo, ALGO_REDUCTION)?,
        swizzle: get(api, algo, ALGO_SWIZZLE)?,
        custom: get(api, algo, ALGO_CUSTOM)?,
        stages: get(api, algo, ALGO_STAGES)?,
        inner: get(api, algo, ALGO_INNER)?,
        cluster: get(api, algo, ALGO_CLUSTER)?,
    })
}
#[expect(
    unsafe_code,
    reason = "SDK typed algorithm init/set reconstructs a config only for the current device and library"
)]
fn restore_algorithm(context: &Context, config: AlgorithmConfig) -> Result<Algo, DeviceError> {
    fn set<T: Copy>(
        api: &Api,
        algo: &mut Algo,
        attribute: i32,
        value: T,
    ) -> Result<(), DeviceError> {
        // SAFETY: Values use the SDK scalar type and are copied synchronously.
        status(
            // SAFETY: This single SDK call uses live lifetime-owned handles and pointers with the declared CUDA header ABI.
            unsafe { (api.algo_set)(algo, attribute, (&raw const value).cast(), size_of::<T>()) },
            "algorithm set",
        )
    }
    let api = context.api;
    let mut algo = Algo::default();
    // SAFETY: Descriptor compute, scale and all matrix types match the initialized FP8 layouts.
    status(
        // SAFETY: This single SDK call uses live lifetime-owned handles and pointers with the declared CUDA header ABI.
        unsafe {
            (api.algo_init)(
                context.handle,
                COMPUTE_F32,
                TYPE_F32,
                TYPE_E4M3,
                TYPE_E4M3,
                TYPE_F32,
                TYPE_F32,
                config.id,
                &raw mut algo,
            )
        },
        "algorithm init",
    )?;
    set(api, &mut algo, ALGO_TILE, config.tile)?;
    set(api, &mut algo, ALGO_SPLIT_K, config.split_k)?;
    set(api, &mut algo, ALGO_REDUCTION, config.reduction)?;
    set(api, &mut algo, ALGO_SWIZZLE, config.swizzle)?;
    set(api, &mut algo, ALGO_CUSTOM, config.custom)?;
    set(api, &mut algo, ALGO_STAGES, config.stages)?;
    set(api, &mut algo, ALGO_INNER, config.inner)?;
    set(api, &mut algo, ALGO_CLUSTER, config.cluster)?;
    Ok(algo)
}
#[expect(
    unsafe_code,
    reason = "Audited scalar attribute values match cublasLt.h and are copied synchronously by the API"
)]
fn set<T: Copy>(api: &Api, desc: Handle, attr: i32, value: T) -> Result<(), DeviceError> {
    status(
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        unsafe { (api.desc_set)(desc, attr, (&raw const value).cast(), size_of::<T>()) },
        "attribute",
    )
}
#[expect(
    unsafe_code,
    reason = "Audited raw launch uses shape-validated live tensors and a zero-workspace warmed algorithm"
)]
fn launch(
    context: &Context,
    prepared: &Prepared,
    input: &Tensor<f8e4m3fn>,
    weight: &Tensor<f8e4m3fn>,
    output: &Tensor<f32>,
) -> Result<(), DeviceError> {
    let _guard = context
        .calls
        .lock()
        .map_err(|_| error("vendor FP8 handle mutex"))?;
    // HOST scalar mode copies these constants into the captured launch parameters. The
    // standalone 32-case pilot verified this mode in CUDA graphs; no host pointer is retained.
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    status(
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        unsafe {
            (context.api.matmul)(
                context.handle,
                prepared.desc,
                (&raw const alpha).cast(),
                weight.device_pointer().cu_deviceptr() as *const c_void,
                prepared.layouts[0],
                input.device_pointer().cu_deviceptr() as *const c_void,
                prepared.layouts[1],
                (&raw const beta).cast(),
                output.device_pointer().cu_deviceptr() as *const c_void,
                prepared.layouts[2],
                output.device_pointer().cu_deviceptr() as *mut c_void,
                prepared.layouts[3],
                &raw const prepared.algo,
                null_mut(),
                0,
                context.stream,
            )
        },
        "matmul",
    )
}

#[expect(
    unsafe_code,
    reason = "Audited SDK descriptors match the documented FP8 T/N layouts and exact F32 accumulation contract"
)]
fn make_descriptors(context: &Context, shape: Shape) -> Result<Arc<Descriptors>, DeviceError> {
    let api = context.api;
    let mut result = Descriptors {
        api,
        desc: null_mut(),
        layouts: [null_mut(); MATRIX_LAYOUTS],
    };
    status(
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        unsafe { (api.desc_create)(&raw mut result.desc, COMPUTE_F32, TYPE_F32) },
        "desc create",
    )?;
    set(api, result.desc, TRANSA, OP_T)?;
    set(api, result.desc, TRANSB, OP_N)?;
    set(api, result.desc, FAST_ACCUM, 0_i8)?;
    set(api, result.desc, A_SCALE_MODE, SCALAR_F32)?;
    set(api, result.desc, B_SCALE_MODE, SCALAR_F32)?;
    for (index, (dtype, rows, columns, stride)) in [
        (TYPE_E4M3, shape.k, shape.n, shape.k),
        (TYPE_E4M3, shape.k, shape.m, shape.k),
        (TYPE_F32, shape.n, shape.m, shape.n),
        (TYPE_F32, shape.n, shape.m, shape.n),
    ]
    .into_iter()
    .enumerate()
    {
        status(
            // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
            unsafe {
                (api.layout_create)(
                    &raw mut result.layouts[index],
                    dtype,
                    u64::try_from(rows).map_err(|e| error(e.to_string()))?,
                    u64::try_from(columns).map_err(|e| error(e.to_string()))?,
                    i64::try_from(stride).map_err(|e| error(e.to_string()))?,
                )
            },
            "layout create",
        )?;
    }
    Ok(Arc::new(result))
}
#[expect(
    unsafe_code,
    reason = "Audited SDK preference and heuristic query use typed alignment and workspace attributes"
)]
fn heuristic_candidates(
    context: &Context,
    descriptors: &Descriptors,
) -> Result<Vec<Heuristic>, DeviceError> {
    let api = context.api;
    let mut preference = null_mut();
    status(
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        unsafe { (api.pref_create)(&raw mut preference) },
        "preference create",
    )?;
    let zero_workspace = 0_u64;
    // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
    let mut pref_status = unsafe {
        (api.pref_set)(
            preference,
            MAX_WORKSPACE,
            (&raw const zero_workspace).cast(),
            size_of::<u64>(),
        )
    };
    let minimum_alignment = ALIGNMENT;
    for attr in MIN_ALIGNMENTS {
        if pref_status != SUCCESS {
            break;
        }
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        pref_status = unsafe {
            (api.pref_set)(
                preference,
                attr,
                (&raw const minimum_alignment).cast(),
                size_of::<u32>(),
            )
        };
    }
    let mut candidates = [Heuristic::default(); HEURISTIC_COUNT];
    let mut count = 0_i32;
    let requested = i32::try_from(HEURISTIC_COUNT).map_err(|e| error(e.to_string()))?;
    let query = if pref_status == SUCCESS {
        // Count actual SDK query attempts on this specific handle, including SDK rejections.
        #[cfg(test)]
        context.heuristic_queries.fetch_add(1, Ordering::Relaxed);
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        unsafe {
            (api.query)(
                context.handle,
                descriptors.desc,
                descriptors.layouts[0],
                descriptors.layouts[1],
                descriptors.layouts[2],
                descriptors.layouts[3],
                preference,
                requested,
                candidates.as_mut_ptr(),
                &raw mut count,
            )
        }
    } else {
        pref_status
    };
    // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
    unsafe {
        (api.pref_destroy)(preference);
    }
    if query == SUCCESS {
        Ok(candidates
            .into_iter()
            .take(usize::try_from(count).unwrap_or_default())
            .collect())
    } else {
        Ok(Vec::new())
    }
}

/// Descriptor owners must be retained beside all graphs that captured this support.
pub struct Support {
    device: CudaDevice,
    context: Option<Arc<Context>>,
    prepared: BTreeMap<Shape, Arc<Prepared>>,
    algorithms: BTreeMap<(Shape, AlgorithmConfig), Arc<Prepared>>,
    available: BTreeMap<Shape, Vec<AlgorithmConfig>>,
    channel_scales: BTreeMap<CUdeviceptr, Arc<Tensor<f32>>>,
    rejected: BTreeSet<Shape>,
    /// The Lt node is consumed during capture, so allocations are anchored here instead.
    weights: BTreeMap<CUdeviceptr, Arc<Tensor<f8e4m3fn>>>,
}
impl Support {
    pub(crate) fn new_with(device: &CudaDevice, enabled: bool) -> Self {
        Self {
            device: device.clone(),
            context: enabled.then(|| context(device)).flatten(),
            prepared: BTreeMap::new(),
            algorithms: BTreeMap::new(),
            available: BTreeMap::new(),
            channel_scales: BTreeMap::new(),
            rejected: BTreeSet::new(),
            weights: BTreeMap::new(),
        }
    }
    /// Keep every projection allocation alive independently of its shared shape algorithm.
    /// Cloning an Arc holds the original GPU bytes; it does not allocate or copy device data.
    pub(crate) fn retain_projection(&mut self, weight: &Arc<Tensor<f8e4m3fn>>) {
        if self.context.is_some() {
            self.weights
                .entry(weight.device_pointer().cu_deviceptr())
                .or_insert_with(|| Arc::clone(weight));
        }
    }

    /// Test-only evidence that typed restoration never consults a heuristic shortlist.
    #[cfg(test)]
    pub(crate) fn heuristic_query_count(&self) -> usize {
        self.context.as_ref().map_or(0, |context| {
            context.heuristic_queries.load(Ordering::Relaxed)
        })
    }

    pub(crate) const fn is_enabled(&self) -> bool {
        self.context.is_some()
    }

    /// Select and execute a zero-workspace algorithm before any Scope starts capture.
    /// Metadata is cached by full M/N/K, independently of checkpoint weight identity.
    pub(crate) fn prepare(
        &mut self,
        input: &Tensor<f8e4m3fn>,
        weight: &Arc<Tensor<f8e4m3fn>>,
        output: &Tensor<f32>,
    ) -> Result<bool, DeviceError> {
        let configs = self.prepare_candidates(input, weight, output)?;
        let Some(config) = configs.first() else {
            return Ok(false);
        };
        let shape = Shape::from(input, weight, output)?;
        if let Some(prepared) = self.algorithms.get(&(shape, *config)).cloned() {
            self.prepared.insert(shape, prepared);
            return Ok(true);
        }
        Ok(false)
    }
    pub(crate) fn retain_channels(
        &mut self,
        codes: &Arc<Tensor<f8e4m3fn>>,
        scales: &Arc<Tensor<f32>>,
    ) {
        self.retain_projection(codes);
        if self.context.is_some() {
            self.channel_scales
                .entry(scales.device_pointer().cu_deviceptr())
                .or_insert_with(|| Arc::clone(scales));
        }
    }
    #[expect(
        unsafe_code,
        reason = "SDK version functions are pure scalar queries on the lifetime-owned library"
    )]
    pub(crate) fn library_identity(&self) -> Option<LibraryIdentity> {
        if !self.is_enabled() {
            return None;
        }
        let api = self.context.as_ref()?.api;
        let identity = LibraryIdentity {
            // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
            lt_version: unsafe { (api.version)() },
            // SAFETY: This pure SDK version query borrows the lifetime-owned loaded library.
            cudart_version: unsafe { (api.cudart_version)() },
        };
        (identity.lt_version > 0 && identity.cudart_version > 0).then_some(identity)
    }
    pub(crate) fn capture_owner(
        &self,
        shape: [usize; GEMM_RANK],
        config: AlgorithmConfig,
    ) -> Option<CaptureOwner> {
        let [m, n, k] = shape;
        Some(CaptureOwner {
            _context: Arc::clone(self.context.as_ref()?),
            _prepared: Arc::clone(self.algorithms.get(&(Shape { m, n, k }, config))?),
        })
    }
    /// Recreate a persisted typed algorithm and perform real zero-workspace warmup before capture.
    #[expect(
        unsafe_code,
        reason = "SDK AlgoCheck validates current descriptors; live guarded buffers validate the actual launch"
    )]
    pub(crate) fn prepare_config(
        &mut self,
        input: &Tensor<f8e4m3fn>,
        weight: &Arc<Tensor<f8e4m3fn>>,
        output: &Tensor<f32>,
        config: AlgorithmConfig,
    ) -> Result<bool, DeviceError> {
        if !compatible(input, weight, output) || self.context.is_none() {
            return Ok(false);
        }
        let shape = Shape::from(input, weight, output)?;
        let Some(context) = self.context.clone() else {
            return Ok(false);
        };
        self.device
            .reclaim_barrier()
            .map_err(|e| error(e.to_string()))?;
        for pointer in [
            input.device_pointer().cu_deviceptr(),
            weight.device_pointer().cu_deviceptr(),
            output.device_pointer().cu_deviceptr(),
        ] {
            same_context(&context, pointer)?;
        }
        self.retain_projection(weight);
        if self.algorithms.contains_key(&(shape, config)) {
            return Ok(true);
        }
        // An SDK-valid cached configuration need not appear in the current heuristic shortlist.
        let descriptors = make_descriptors(&context, shape)?;
        let algo = restore_algorithm(&context, config)?;
        let mut checked = Heuristic::default();
        // SAFETY: The SDK call uses live lifetime-owned handles and pointers with the declared header ABI.
        let code = unsafe {
            (context.api.algo_check)(
                context.handle,
                descriptors.desc,
                descriptors.layouts[0],
                descriptors.layouts[1],
                descriptors.layouts[2],
                descriptors.layouts[3],
                &raw const algo,
                &raw mut checked,
            )
        };
        if code != SUCCESS || checked.state != SUCCESS || checked.workspace != 0 {
            return Ok(false);
        }
        let prepared = Prepared { descriptors, algo };
        launch(&context, &prepared, input, weight, output)?;
        self.device
            .reclaim_barrier()
            .map_err(|e| error(e.to_string()))?;
        self.algorithms.insert((shape, config), Arc::new(prepared));
        Ok(true)
    }

    pub(crate) fn prepare_candidates(
        &mut self,
        input: &Tensor<f8e4m3fn>,
        weight: &Arc<Tensor<f8e4m3fn>>,
        output: &Tensor<f32>,
    ) -> Result<Vec<AlgorithmConfig>, DeviceError> {
        let Some(context) = self.context.clone() else {
            return Ok(Vec::new());
        };
        if !compatible(input, weight, output) {
            return Ok(Vec::new());
        }
        let shape = Shape::from(input, weight, output)?;
        // cuda.h cuMemAllocAsync: the pointer is returned immediately, but allocation must
        // complete before host metadata queries. This occurs only in pre-capture preparation.
        self.device
            .reclaim_barrier()
            .map_err(|e| error(e.to_string()))?;
        for pointer in [
            input.device_pointer().cu_deviceptr(),
            weight.device_pointer().cu_deviceptr(),
            output.device_pointer().cu_deviceptr(),
        ] {
            same_context(&context, pointer)?;
        }
        // Retain even when a previous projection has already warmed this M/N/K.
        self.retain_projection(weight);
        if let Some(configs) = self.available.get(&shape) {
            return Ok(configs.clone());
        }
        if self.rejected.contains(&shape) {
            return Ok(Vec::new());
        }
        let descriptors = make_descriptors(&context, shape)?;
        let candidates = heuristic_candidates(&context, &descriptors)?;
        let api = context.api;
        let mut accepted = Vec::new();
        for candidate in &candidates {
            if candidate.state != SUCCESS || candidate.workspace != 0 {
                continue;
            }
            let Ok(config) = algorithm_config(api, &candidate.algo) else {
                continue;
            };
            if accepted.contains(&config) {
                continue;
            }
            let prepared = Prepared {
                descriptors: Arc::clone(&descriptors),
                algo: candidate.algo,
            };
            if launch(&context, &prepared, input, weight, output).is_ok() {
                self.device
                    .reclaim_barrier()
                    .map_err(|e| error(e.to_string()))?;
                self.algorithms.insert((shape, config), Arc::new(prepared));
                accepted.push(config);
            }
        }
        if accepted.is_empty() {
            self.rejected.insert(shape);
        }
        self.available.insert(shape, accepted.clone());
        Ok(accepted)
    }

    /// Record only a previously warmed shape. No allocation or heuristic occurs in capture.
    /// False is returned before recording any node. An error aborts the entire capture; callers
    /// must not append native kernels after an error from a partially recorded vendor path.
    pub(crate) fn record(
        &self,
        scope: &Scope,
        input: &Tensor<f8e4m3fn>,
        weight: &Tensor<f8e4m3fn>,
        output: &mut Tensor<f32>,
        xs: &Tensor<f32>,
        ws: &Tensor<f32>,
    ) -> Result<bool, DeviceError> {
        let shape = Shape::from(input, weight, output)?;
        let Some(prepared) = self.prepared.get(&shape) else {
            return Ok(false);
        };
        self.record_prepared(scope, prepared, input, weight, output, xs, ws)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "The audited graph boundary receives all GEMM operands and scales with the selected algorithm owner"
    )]
    pub(crate) fn record_config(
        &self,
        scope: &Scope,
        config: AlgorithmConfig,
        input: &Tensor<f8e4m3fn>,
        weight: &Tensor<f8e4m3fn>,
        output: &mut Tensor<f32>,
        xs: &Tensor<f32>,
        ws: &Tensor<f32>,
    ) -> Result<bool, DeviceError> {
        let shape = Shape::from(input, weight, output)?;
        let Some(prepared) = self.algorithms.get(&(shape, config)) else {
            return Ok(false);
        };
        self.record_prepared(scope, prepared, input, weight, output, xs, ws)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "The audited graph boundary receives all GEMM operands and scales with the selected algorithm owner"
    )]
    fn record_prepared(
        &self,
        scope: &Scope,
        prepared: &Arc<Prepared>,
        input: &Tensor<f8e4m3fn>,
        weight: &Tensor<f8e4m3fn>,
        output: &mut Tensor<f32>,
        xs: &Tensor<f32>,
        ws: &Tensor<f32>,
    ) -> Result<bool, DeviceError> {
        let Some(context) = self.context.as_ref() else {
            return Ok(false);
        };
        if !compatible(input, weight, output) || !xs.is_contiguous() || !ws.is_contiguous() {
            return Ok(false);
        }
        let shape = Shape::from(input, weight, output)?;
        if xs.size() != shape.m || ws.size() != shape.n {
            return Err(error("vendor FP8 channel scale extents"));
        }
        for pointer in [
            input.device_pointer().cu_deviceptr(),
            weight.device_pointer().cu_deviceptr(),
            output.device_pointer().cu_deviceptr(),
            xs.device_pointer().cu_deviceptr(),
            ws.device_pointer().cu_deviceptr(),
        ] {
            same_context(context, pointer)?;
        }
        scope.record(GemmFp8 {
            context: context.clone(),
            prepared: prepared.clone(),
            input,
            weight,
            output,
        })?;
        scope.record(kernels::scale(output.partition(SCALE_TILE), xs, ws))?;
        Ok(true)
    }
}
struct GemmFp8<'a> {
    context: Arc<Context>,
    prepared: Arc<Prepared>,
    input: &'a Tensor<f8e4m3fn>,
    weight: &'a Tensor<f8e4m3fn>,
    output: &'a Tensor<f32>,
}
impl GraphNode for GemmFp8<'_> {}
#[expect(
    unsafe_code,
    reason = "Audited graph boundary checks stream identity and delegates shape-validated tensors to the owned Lt context"
)]
impl DeviceOp for GemmFp8<'_> {
    type Output = ();
    unsafe fn execute(self, execution: &ExecutionContext) -> Result<(), DeviceError> {
        if execution.get_cuda_stream().cu_stream() != self.context.stream {
            return Err(error(format!(
                "vendor FP8 device {} stream mismatch",
                self.context.ordinal
            )));
        }
        launch(
            &self.context,
            &self.prepared,
            self.input,
            self.weight,
            self.output,
        )
    }
}
impl IntoFuture for GemmFp8<'_> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        LoadTileMutAtCurrentBlock, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tile_2, ftz, get_tile_block_id, mulf, rounding,
    };
    #[cutile::entry()]
    fn scale(
        output: &mut Tensor<f32, { [1, 256] }>,
        xs: &Tensor<f32, { [-1, 1] }>,
        ws: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let raw: Tile<f32, { [1, 256] }> = output.load();
        let row = xs
            .partition(shape![1, 1])
            .load([pid.0, 0i32])
            .broadcast(shape![1, 256]);
        let column = ws
            .partition(shape![256])
            .load([pid.1])
            .reshape(shape![1, 256]);
        let first = mulf(raw, row, rounding::NearestEven, ftz::Disabled);
        output.store(mulf(first, column, rounding::NearestEven, ftz::Disabled));
    }
}

pub mod nvfp4;
