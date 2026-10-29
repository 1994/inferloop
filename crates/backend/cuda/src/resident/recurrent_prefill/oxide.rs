//! Exact F32 recurrence from pinned Rust SIMT source, with graph operand leases.
use crate::device::CudaDevice;
use crate::device::architecture::is_sm120;
use crate::device::device_error;
use crate::tuning::recurrent::exact::select as exact_select;
use crate::tuning::resident_gemm::ResidentTuningSession;
use cuda_core::{Function, LaunchConfig};
use cutile::{
    cuda_async::{
        device_context::with_default_device_policy, device_future::DeviceFuture,
        launch::AsyncKernelLaunch,
    },
    prelude::*,
};
use infer_core::Result;
#[cfg(test)]
use std::cell::Cell;
use std::result::Result as StdResult;
const KH: usize = 16;
const VH: usize = 48;
const DIM: usize = 128;
const BLOCK: usize = 8;
const SHAPE_RANK: usize = 4;
const GEOMETRY_RANK: usize = 3;
const SUPPORTED_LANES: [usize; 5] = [64, 256, 512, 1024, 2048];
const STATE_SHAPE: [i32; 3] = [48, 128, 128];
const REQUIRED_METADATA_FIELDS: usize = 3;
const GRID_BLOCKS: u32 = 768;
const THREADS: u32 = 128;
const PTX: &str = include_str!("../../../kernels/oxide-gdn/gdn-sm120.ptx");
pub struct Support {
    function: Arc<Function>,
    stream: usize,
}
impl Support {
    pub fn load(device: &CudaDevice) -> Result<Self> {
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(PTX)
            .map_err(device_error)?;
        Ok(Self {
            function: Arc::new(module.load_function("gdn_f32_128").map_err(device_error)?),
            stream: device.stream.cu_stream() as usize,
        })
    }
    pub(super) fn select(
        device: &CudaDevice,
        shape: [usize; SHAPE_RANK],
        autotune: bool,
        session: &ResidentTuningSession,
    ) -> Result<Option<Self>> {
        if shape[..GEOMETRY_RANK] != [KH, VH, DIM]
            || !SUPPORTED_LANES.contains(&shape[3])
            || !is_sm120(&device.profile()?.architecture)
        {
            return Ok(None);
        }
        #[cfg(test)]
        if DISABLED.get() {
            return Ok(None);
        }
        if exact_select(device, shape, autotune, session) {
            Self::load(device).map(Some)
        } else {
            Ok(None)
        }
    }
    pub fn record(
        &self,
        scope: &Scope,
        bindings: Bindings<'_>,
        lanes: usize,
    ) -> StdResult<(), DeviceError> {
        record(scope, &self.function, bindings, self.stream, lanes)
    }
}
#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}
pub struct Bindings<'a> {
    pub state: &'a mut Tensor<f32>,
    pub output: &'a mut Tensor<f32>,
    pub qkv: &'a Tensor<f32>,
    pub normalized: &'a Tensor<f32>,
    pub beta: &'a Tensor<f32>,
    pub decay: &'a Tensor<f32>,
    pub metadata: &'a Tensor<i32>,
}
struct Recorded<'a> {
    launch: AsyncKernelLaunch,
    bindings: Bindings<'a>,
    stream: usize,
}
impl GraphNode for Recorded<'_> {}
#[expect(
    unsafe_code,
    reason = "launch uses graph-retained checked contiguous operands on the owning CUDA stream"
)]
impl DeviceOp for Recorded<'_> {
    type Output = ();
    unsafe fn execute(self, ctx: &ExecutionContext) -> StdResult<(), DeviceError> {
        if ctx.get_cuda_stream().cu_stream() as usize != self.stream {
            return Err(DeviceError::Internal("GDN stream mismatch".into()));
        }
        for input in [
            self.bindings.qkv,
            self.bindings.normalized,
            self.bindings.beta,
            self.bindings.decay,
        ] {
            KernelInputStored::retain(&input, ctx)?;
        }
        KernelInputStored::retain(&self.bindings.metadata, ctx)?;
        KernelOutputStored::retain(&(&mut *self.bindings.state).partition([1, DIM, BLOCK]), ctx)?;
        KernelOutputStored::retain(&(&mut *self.bindings.output).partition([1, 1, BLOCK]), ctx)?;
        ctx.retain(Arc::clone(&self.launch.func))?;
        // SAFETY: module and storage leases are retained, buffers are bound to the validated stream.
        unsafe { self.launch.execute(ctx) }
    }
}
impl IntoFuture for Recorded<'_> {
    type Output = StdResult<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(error)) | Err(error) => DeviceFuture::failed(error),
        }
    }
}
#[expect(
    unsafe_code,
    reason = "explicit PTX ABI for fixed contiguous F32 operands"
)]
pub(super) fn record(
    scope: &Scope,
    function: &Arc<Function>,
    bindings: Bindings<'_>,
    stream: usize,
    lanes: usize,
) -> StdResult<(), DeviceError> {
    if !SUPPORTED_LANES.contains(&lanes)
        || bindings.state.shape() != STATE_SHAPE
        || bindings.output.size() != VH * lanes * DIM
        || bindings.qkv.size() != lanes * (2 * KH + VH) * DIM
        || bindings.normalized.size() != lanes * 2 * KH * DIM
        || bindings.beta.size() != lanes * VH
        || bindings.decay.size() != lanes * VH
        || bindings.metadata.size() < REQUIRED_METADATA_FIELDS
        || ![
            &*bindings.state,
            &*bindings.output,
            bindings.qkv,
            bindings.normalized,
            bindings.beta,
            bindings.decay,
        ]
        .iter()
        .all(|t| t.is_contiguous())
        || !bindings.metadata.is_contiguous()
    {
        return Err(DeviceError::Internal(
            "Rust GDN contiguous operand contract".into(),
        ));
    }
    let mut launch = AsyncKernelLaunch::new(Arc::clone(function));
    for pointer in [
        bindings.state.device_pointer().cu_deviceptr(),
        bindings.output.device_pointer().cu_deviceptr(),
        bindings.qkv.device_pointer().cu_deviceptr(),
        bindings.normalized.device_pointer().cu_deviceptr(),
        bindings.beta.device_pointer().cu_deviceptr(),
        bindings.decay.device_pointer().cu_deviceptr(),
        bindings.metadata.device_pointer().cu_deviceptr(),
    ] {
        // SAFETY: recorded operation borrows and retains every fixed-shape fixture allocation.
        unsafe {
            launch.push_device_ptr(pointer);
        }
    }
    launch.push_arg(u32::try_from(lanes).map_err(|e| DeviceError::Internal(e.to_string()))?);
    launch.set_launch_config(LaunchConfig {
        grid_dim: (GRID_BLOCKS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    });
    scope.record(Recorded {
        launch,
        bindings,
        stream,
    })?;
    #[cfg(test)]
    CAPTURES.set(CAPTURES.get() + 1);
    Ok(())
}

#[cfg(test)]
pub fn with_cutile<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let guard = Reset(DISABLED.replace(true));
    let result = f();
    drop(guard);
    result
}
