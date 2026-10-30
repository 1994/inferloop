//! One stream-ordered launch retains all checked operands until the device fence.
use super::{
    geometry::{CHANNELS, Geometry},
    tables::Tables,
};
use cuda_core::{Function, LaunchConfig};
use cutile::{
    cuda_async::{
        device_context::with_default_device_policy, device_future::DeviceFuture,
        launch::AsyncKernelLaunch,
    },
    prelude::*,
};
use std::result::Result as StdResult;
use std::sync::Arc;

/// Threads per one-dimensional image kernel block.
const THREADS: u32 = 256;
/// Canonical RGB rescaling and symmetric normalization both use one half.
const CHANNEL_NORMALIZATION: f32 = 0.5;
/// Mean and standard deviation for each channel.
const NORMALIZATION_FIELDS: usize = CHANNELS * 2;

pub(super) struct Kernels {
    pub input: Tensor<u8>,
    pub coefficients: Tensor<i32>,
    pub intermediate: Tensor<u8>,
    pub output: Tensor<f32>,
    pub geometry: Geometry,
    pub tables: Tables,
    pub horizontal: Arc<Function>,
    pub vertical: Arc<Function>,
    pub stream: usize,
}

#[expect(
    unsafe_code,
    reason = "The checked PTX ABI is retained and submitted on its owning stream"
)]
impl DeviceOp for Kernels {
    type Output = Tensor<f32>;

    unsafe fn execute(mut self, ctx: &ExecutionContext) -> StdResult<Tensor<f32>, DeviceError> {
        if ctx.get_cuda_stream().cu_stream() as usize != self.stream {
            return Err(DeviceError::Internal(
                "image preprocessing stream mismatch".into(),
            ));
        }
        let (horizontal, vertical) = self.launches()?;
        KernelInputStored::retain(&&self.input, ctx)?;
        KernelInputStored::retain(&&self.coefficients, ctx)?;
        KernelOutputStored::retain(&(&mut self.intermediate).partition([1]), ctx)?;
        KernelOutputStored::retain(&(&mut self.output).partition([1]), ctx)?;
        ctx.retain(Arc::clone(&self.horizontal))?;
        ctx.retain(Arc::clone(&self.vertical))?;
        // SAFETY: all pointers and argument ranges were checked before launch. The execution
        // context retains every allocation and both kernel functions until the stream fences;
        // the second kernel follows the first on the same checked CUDA stream.
        unsafe { horizontal.execute(ctx)? };
        // SAFETY: the same retained operands remain live and this launch is ordered after the
        // horizontal kernel on the same stream.
        unsafe { vertical.execute(ctx)? };
        Ok(self.output)
    }
}

impl IntoFuture for Kernels {
    type Output = StdResult<Tensor<f32>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<f32>, Self>;

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

impl Kernels {
    #[expect(unsafe_code, reason = "Pointers follow the validated image kernel ABI")]
    fn launches(&self) -> StdResult<(AsyncKernelLaunch, AsyncKernelLaunch), DeviceError> {
        let base = self.coefficients.device_pointer().cu_deviceptr();
        let offsets = self.tables.offsets.map(|offset| base.checked_add(offset));
        let [
            Some(x_indices),
            Some(x_weights),
            Some(y_indices),
            Some(y_weights),
        ] = offsets
        else {
            return Err(DeviceError::Internal(
                "image coefficient address overflow".into(),
            ));
        };
        let input = self.input.device_pointer().cu_deviceptr();
        let intermediate = self.intermediate.device_pointer().cu_deviceptr();
        let output = self.output.device_pointer().cu_deviceptr();
        let mut horizontal = AsyncKernelLaunch::new(Arc::clone(&self.horizontal));
        // SAFETY: the input, table and intermediate allocations are checked and retained in
        // `execute` before either launch. The pointers match the PTX's four pointer parameters.
        unsafe {
            for pointer in [input, x_indices, x_weights, intermediate] {
                horizontal.push_device_ptr(pointer);
            }
        }
        for value in [
            self.geometry.batch,
            self.geometry.source_height,
            self.geometry.source_width,
            self.geometry.target_width,
            self.tables.horizontal_taps,
            self.tables.horizontal_denominator,
        ] {
            horizontal.push_arg(value);
        }
        horizontal.set_launch_config(config(self.geometry.horizontal_elements)?);
        let mut vertical = AsyncKernelLaunch::new(Arc::clone(&self.vertical));
        // SAFETY: all four pointers address the retained allocations, with table offsets checked
        // at construction and an output sized for every padded patch.
        unsafe {
            for pointer in [intermediate, y_indices, y_weights, output] {
                vertical.push_device_ptr(pointer);
            }
        }
        for value in [
            self.geometry.batch,
            self.geometry.source_height,
            self.geometry.target_height,
            self.geometry.target_width,
            self.tables.vertical_taps,
            self.tables.vertical_denominator,
            self.geometry.padded_patches,
        ] {
            vertical.push_arg(value);
        }
        for _ in 0..NORMALIZATION_FIELDS {
            vertical.push_arg(CHANNEL_NORMALIZATION);
        }
        vertical.set_launch_config(config(self.geometry.output_elements)?);
        Ok((horizontal, vertical))
    }
}

fn config(elements: usize) -> StdResult<LaunchConfig, DeviceError> {
    let threads =
        usize::try_from(THREADS).map_err(|error| DeviceError::Internal(error.to_string()))?;
    let blocks = u32::try_from(elements.div_ceil(threads))
        .map_err(|error| DeviceError::Internal(error.to_string()))?;
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}
