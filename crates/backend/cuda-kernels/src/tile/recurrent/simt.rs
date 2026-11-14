//! The exact F32 prompt recurrence compiled from pinned Rust SIMT source, launched as PTX.
//!
//! It computes [`Recurrent::delta_prepared`] bit for bit on the one geometry it was built for.
//! The module is loaded once, outside capture, because loading is not a stream operation.
//!
//! [`Recurrent::delta_prepared`]: crate::contract::Recurrent::delta_prepared
#![expect(
    unsafe_code,
    reason = "Explicit PTX ABI over checked contiguous operands, on the stream of the module's device, with every lease retained"
)]
use crate::contract::buffer::Buffer;
use crate::contract::kernels::PreparedDeltaBuffers;
use crate::contract::plan::RecurrentShape;
use crate::contract::recording::Recording;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::{Device, Function, LaunchConfig};
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::launch::AsyncKernelLaunch;
use cutile::prelude::*;
use cutile::tensor::{KernelInputStored, KernelOutputStored};
use infer_core::{Error, Result};
use std::result::Result as StdResult;
use std::sync::{Arc, OnceLock};

/// The pinned build; its bytes are part of the exact recurrence tuning identity.
const PTX: &str = include_str!("../../../../cuda/kernels/oxide-gdn/gdn-sm120.ptx");
/// Entry point of the build.
const ENTRY: &str = "gdn_f32_128";
/// Key heads the build was compiled for.
const KEY_HEADS: usize = 16;
/// Value heads the build was compiled for.
const VALUE_HEADS: usize = 48;
/// Channels of one head.
const DIM: usize = 128;
/// Value columns one thread block owns.
const BLOCK: usize = 8;
/// Prompt widths the build was compiled for.
const LANES: [usize; 5] = [64, 256, 512, 1024, 2048];
/// Metadata fields the build reads: base, count and offset.
const METADATA_FIELDS: usize = 3;
/// Thread blocks of one launch.
const GRID_BLOCKS: u32 = 768;
/// Threads of one block.
const THREADS: u32 = 128;
/// Read-only F32 operands: qkv, normalized, betas and decays.
const INPUTS: usize = 4;

/// The SIMT build, loaded on first preparation.
pub(super) struct Simt {
    function: OnceLock<Arc<Function>>,
}

/// Whether the build covers `shape`.
fn covers(shape: RecurrentShape) -> bool {
    (shape.key_heads, shape.value_heads, shape.dim) == (KEY_HEADS, VALUE_HEADS, DIM)
        && LANES.contains(&shape.lanes)
}

impl Simt {
    pub(super) const fn new() -> Self {
        Self {
            function: OnceLock::new(),
        }
    }

    /// Load the build for device `ordinal`; `false` when it does not cover `shape`.
    pub(super) fn prepare(&self, ordinal: usize, shape: RecurrentShape) -> Result<bool> {
        if !covers(shape) {
            return Ok(false);
        }
        if self.function.get().is_none() {
            let function = Device::new(ordinal)
                .and_then(|device| device.load_module_from_ptx_src(PTX))
                .and_then(|module| module.load_function(ENTRY))
                .map_err(backend_error)?;
            self.function.get_or_init(|| Arc::new(function));
        }
        Ok(true)
    }

    /// Record the recurrence over prepared normalization and gates.
    pub(super) fn record(
        &self,
        recording: &Recording<'_>,
        ordinal: usize,
        shape: RecurrentShape,
        buffers: PreparedDeltaBuffers<'_>,
    ) -> Result<()> {
        let function = self
            .function
            .get()
            .ok_or_else(|| Error::invalid("SIMT recurrence recorded before it was prepared"))?;
        let lanes = shape.lanes;
        let PreparedDeltaBuffers {
            state,
            output,
            qkv,
            normalized,
            betas,
            decays,
            metadata,
        } = buffers;
        let state = state
            .as_f32()
            .ok_or_else(|| Error::unsupported("the SIMT recurrence keeps an F32 state"))?;
        if !covers(shape)
            || state.len() != VALUE_HEADS * DIM * DIM
            || output.len() != VALUE_HEADS * lanes * DIM
            || qkv.len() != lanes * (2 * KEY_HEADS + VALUE_HEADS) * DIM
            || normalized.len() != lanes * 2 * KEY_HEADS * DIM
            || betas.len() != lanes * VALUE_HEADS
            || decays.len() != lanes * VALUE_HEADS
            || metadata.len() < METADATA_FIELDS
        {
            return Err(Error::invalid("SIMT recurrence operands"));
        }
        let scope = capture_scope(recording)?;
        let mut launch = AsyncKernelLaunch::new(Arc::clone(function));
        for address in [
            state.address(),
            output.address(),
            qkv.address(),
            normalized.address(),
            betas.address(),
            decays.address(),
            metadata.address(),
        ] {
            // SAFETY: every buffer was checked above against the build's fixed extents, and its
            // owner keeps it alive for as long as the graph lives.
            unsafe { launch.push_device_ptr(address) };
        }
        launch.push_arg(u32::try_from(lanes).map_err(backend_error)?);
        launch.set_launch_config(LaunchConfig {
            grid_dim: (GRID_BLOCKS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        });
        for buffer in [state, output, qkv, normalized, betas, decays] {
            recording.retain(buffer);
        }
        recording.retain(metadata);
        let flat = |buffer: Buffer<'_, f32>| borrowed(buffer, ordinal, &[buffer.len()]);
        scope
            .record(Recorded {
                launch,
                ordinal,
                state: borrowed(state, ordinal, &[VALUE_HEADS, DIM, DIM])?,
                output: borrowed(output, ordinal, &[VALUE_HEADS, lanes, DIM])?,
                inputs: [flat(qkv)?, flat(normalized)?, flat(betas)?, flat(decays)?],
                metadata: borrowed(metadata, ordinal, &[metadata.len()])?,
            })
            .map_err(backend_error)?;
        Ok(())
    }
}

/// One launch and the operands whose leases its graph node holds.
struct Recorded {
    launch: AsyncKernelLaunch,
    ordinal: usize,
    state: Tensor<f32>,
    output: Tensor<f32>,
    inputs: [Tensor<f32>; INPUTS],
    metadata: Tensor<i32>,
}

impl GraphNode for Recorded {}

impl DeviceOp for Recorded {
    type Output = ();

    unsafe fn execute(mut self, ctx: &ExecutionContext) -> StdResult<(), DeviceError> {
        if ctx.get_cuda_stream().device().ordinal() != self.ordinal {
            return Err(DeviceError::Internal(
                "SIMT recurrence captured on another device".into(),
            ));
        }
        for input in &self.inputs {
            KernelInputStored::retain(&input, ctx)?;
        }
        KernelInputStored::retain(&&self.metadata, ctx)?;
        KernelOutputStored::retain(&(&mut self.state).partition([1, DIM, BLOCK]), ctx)?;
        KernelOutputStored::retain(&(&mut self.output).partition([1, 1, BLOCK]), ctx)?;
        ctx.retain(Arc::clone(&self.launch.func))?;
        // SAFETY: the module belongs to this stream's device and every operand lease is retained.
        unsafe { self.launch.execute(ctx) }
    }
}

impl IntoFuture for Recorded {
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
