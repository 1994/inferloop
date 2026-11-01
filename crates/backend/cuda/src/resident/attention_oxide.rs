//! Exact F32 prefill attention compiled from the pinned Rust SIMT source.
//! The generated PTX is embedded so host builds keep their stable toolchain.
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::{CudaDevice, device_error};
use cuda_core::{Function, LaunchConfig};
use cutile::{
    cuda_async::{
        device_context::with_default_device_policy, device_future::DeviceFuture,
        launch::AsyncKernelLaunch,
    },
    prelude::*,
};
use infer_core::Result;
use infer_ir::{DataflowGraph, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::result::Result as StdResult;
use std::sync::OnceLock;
mod shared;
mod vector;

const PTX: &str = include_str!("../../kernels/oxide-attention/attention-sm120.ptx");
const HEADS: usize = 24;
const KV_HEADS: usize = 4;
const DIM: usize = 256;
const THREADS: u32 = 256;
const SUPPORTED_LANES: [usize; 5] = [64, 256, 512, 1024, 2048];
const REQUIRED_METADATA_FIELDS: usize = 3;

pub struct Support {
    function: Arc<Function>,
    stream: usize,
    threads: u32,
    rows_per_cta: usize,
    lanes: &'static [usize],
}

pub struct Bindings<'a, E: DType> {
    pub output: &'a mut Tensor<f32>,
    pub query: &'a Tensor<f32>,
    pub keys: &'a Tensor<E>,
    pub values: &'a Tensor<E>,
    pub metadata: &'a Tensor<i32>,
    pub table: &'a Tensor<i32>,
}

impl Support {
    pub(super) fn eligible(device: &CudaDevice, graph: &DataflowGraph) -> bool {
        #[cfg(test)]
        if DISABLED.get() {
            return false;
        }
        device.target().has_simt_artifacts()
            && graph.nodes.iter().any(|node| {
                matches!(
                    node.op,
                    TensorOp::Attention {
                        query_heads: HEADS,
                        kv_heads: KV_HEADS,
                        head_dim: DIM,
                        ..
                    }
                )
            })
    }

    pub fn load(device: &CudaDevice) -> Result<Option<Self>> {
        if !device.target().has_simt_artifacts() {
            return Ok(None);
        }
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(PTX)
            .map_err(device_error)?;
        Ok(Some(Self {
            function: Arc::new(
                module
                    .load_function("attention_fp8_256")
                    .map_err(device_error)?,
            ),
            stream: device.stream.cu_stream() as usize,
            threads: THREADS,
            rows_per_cta: 1,
            lanes: &SUPPORTED_LANES,
        }))
    }

    pub fn load_vector(device: &CudaDevice) -> Result<Option<Self>> {
        vector::load(device)
    }

    pub fn load_shared(device: &CudaDevice) -> Result<Option<Self>> {
        shared::load(device)
    }

    #[cfg(test)]
    pub(crate) fn load_test_ptx(device: &CudaDevice, ptx: &str) -> Result<Self> {
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(ptx)
            .map_err(device_error)?;
        Ok(Self {
            function: Arc::new(
                module
                    .load_function("attention_fp8_256")
                    .map_err(device_error)?,
            ),
            stream: device.stream.cu_stream() as usize,
            threads: THREADS,
            rows_per_cta: 1,
            lanes: &SUPPORTED_LANES,
        })
    }
    /// Whether this kernel's fixed layout fits these operands. A pool whose shared pages outnumber
    /// a slot's capacity does not, so the caller takes its next path.
    pub fn accepts<E: DType>(
        &self,
        bindings: &Bindings<'_, E>,
        (heads, kv_heads, dim, lanes, capacity): (usize, usize, usize, usize, usize),
        scales: [f32; 2],
    ) -> StdResult<bool, DeviceError> {
        Ok(!((heads, kv_heads, dim) != (HEADS, KV_HEADS, DIM)
            || !self.lanes.contains(&lanes)
            || E::DTYPE != cuda_core::f8e4m3fn::DTYPE
            || size_of::<E>() != 1
            || capacity == 0
            || !capacity.is_multiple_of(KV_BLOCK_TOKENS)
            || !scales.iter().all(|x| x.is_finite() && *x > 0.0)
            || bindings.query.size() != lanes * HEADS * DIM
            || !bindings.query.is_contiguous()
            || bindings.keys.shape()
                != [
                    i32::try_from(KV_HEADS).map_err(super::capture::error)?,
                    i32::try_from(capacity).map_err(super::capture::error)?,
                    i32::try_from(DIM).map_err(super::capture::error)?,
                ]
            || bindings.values.shape() != bindings.keys.shape()
            || bindings.output.size() != lanes * HEADS * DIM
            || !bindings.output.is_contiguous()
            || !bindings.keys.is_contiguous()
            || !bindings.values.is_contiguous()
            || !bindings.metadata.is_contiguous()
            || bindings.metadata.size() < REQUIRED_METADATA_FIELDS
            || !bindings.table.is_contiguous()
            || bindings.table.size() == 0))
    }

    #[expect(
        unsafe_code,
        reason = "graph-owned tensors with checked contiguous extents are marshalled into the pinned Rust kernel"
    )]
    pub fn record<E: DType>(
        &self,
        scope: &Scope,
        bindings: Bindings<'_, E>,
        (heads, kv_heads, dim, lanes, capacity): (usize, usize, usize, usize, usize),
        window: i32,
        scales: [f32; 2],
    ) -> StdResult<(), DeviceError> {
        if !self.accepts(&bindings, (heads, kv_heads, dim, lanes, capacity), scales)? {
            return Err(super::capture::error(
                "selected SIMT attention cannot record these operands",
            ));
        }
        let mut launch = AsyncKernelLaunch::new(Arc::clone(&self.function));
        for pointer in [
            bindings.query.device_pointer().cu_deviceptr(),
            bindings.keys.device_pointer().cu_deviceptr(),
            bindings.values.device_pointer().cu_deviceptr(),
            bindings.metadata.device_pointer().cu_deviceptr(),
            bindings.table.device_pointer().cu_deviceptr(),
            bindings.output.device_pointer().cu_deviceptr(),
        ] {
            // SAFETY: the program owns all six buffers until its captured graphs are destroyed.
            // The resident allocator validates logical positions and every physical table entry.
            unsafe {
                launch.push_device_ptr(pointer);
            }
        }
        launch.push_arg(u32::try_from(capacity).map_err(super::capture::error)?);
        launch.push_arg(window);
        launch.push_arg(scales[0]);
        launch.push_arg(scales[1]);
        launch.set_launch_config(LaunchConfig {
            grid_dim: (
                u32::try_from((lanes * HEADS).div_ceil(self.rows_per_cta))
                    .map_err(super::capture::error)?,
                1,
                1,
            ),
            block_dim: (self.threads, 1, 1),
            shared_mem_bytes: 0,
        });
        scope.record(Recorded {
            launch,
            stream: self.stream,
            bindings,
        })?;
        #[cfg(test)]
        CAPTURES.set(CAPTURES.get() + 1);
        Ok(())
    }
}

struct Recorded<'a, E: DType> {
    launch: AsyncKernelLaunch,
    stream: usize,
    bindings: Bindings<'a, E>,
}
impl<E: DType> GraphNode for Recorded<'_, E> {}
#[expect(
    unsafe_code,
    reason = "audited graph launch on the prepared device stream with live borrowed operands"
)]
impl<E: DType> DeviceOp for Recorded<'_, E> {
    type Output = ();
    unsafe fn execute(self, ctx: &ExecutionContext) -> StdResult<(), DeviceError> {
        if ctx.get_cuda_stream().cu_stream() as usize != self.stream {
            return Err(DeviceError::Internal(
                "Rust attention stream mismatch".into(),
            ));
        }
        KernelInputStored::retain(&self.bindings.query, ctx)?;
        KernelInputStored::retain(&self.bindings.keys, ctx)?;
        KernelInputStored::retain(&self.bindings.values, ctx)?;
        KernelInputStored::retain(&self.bindings.metadata, ctx)?;
        KernelInputStored::retain(&self.bindings.table, ctx)?;
        KernelOutputStored::retain(&(&mut *self.bindings.output).partition([1, DIM]), ctx)?;
        ctx.retain(Arc::clone(&self.launch.func))?;
        // SAFETY: the stream was checked, all buffers remain owned by the program and the module is retained.
        unsafe { self.launch.execute(ctx) }
    }
}
impl<E: DType> IntoFuture for Recorded<'_, E> {
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

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub fn disabled() -> bool {
    DISABLED.get()
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
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

pub fn artifact_identity() -> &'static str {
    use sha2::{Digest, Sha256};
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let source = include_str!("../../kernels/oxide-attention/src/main.rs");
        let provenance = include_str!("../../kernels/oxide-attention/provenance.json");
        format!(
            "{:x}:{:x}:{:x}",
            Sha256::digest(PTX),
            Sha256::digest(source),
            Sha256::digest(provenance)
        )
    })
}

#[cfg(test)]
#[path = "../../tests/unit/resident_attention_oxide_resources.rs"]
mod resources_tests;

pub fn shared_identity() -> &'static str {
    shared::artifact_identity()
}

pub fn vector_identity() -> &'static str {
    vector::artifact_identity()
}
