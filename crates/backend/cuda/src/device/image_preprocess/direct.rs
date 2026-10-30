//! Stream-ordered patch construction when the source geometry needs no resize.
use super::{
    geometry::Geometry,
    launch::{config, overwritten},
    source::Source,
};
use crate::device::CudaDevice;
use cuda_core::Function;
use cutile::{
    cuda_async::{
        device_context::with_default_device_policy, device_future::DeviceFuture,
        launch::AsyncKernelLaunch,
    },
    prelude::*,
};
use std::result::Result as StdResult;
use std::sync::Arc;

/// Mean and standard deviation in the supported symmetric normalization.
const NORMALIZATION: f32 = 0.5;
/// Each still-image patch repeats its pixels for two temporal positions.
const TEMPORAL: usize = 2;

pub(super) struct Direct {
    pub device: CudaDevice,
    pub source: Source,
    pub geometry: Geometry,
    pub kernel: Arc<Function>,
    pub stream: usize,
}

#[expect(
    unsafe_code,
    reason = "The checked PTX ABI and its buffers remain owned through the stream fence"
)]
impl DeviceOp for Direct {
    type Output = Tensor<f32>;

    unsafe fn execute(self, ctx: &ExecutionContext) -> StdResult<Tensor<f32>, DeviceError> {
        if ctx.get_cuda_stream().cu_stream() as usize != self.stream {
            return Err(DeviceError::Internal(
                "image preprocessing stream mismatch".into(),
            ));
        }
        // SAFETY: the pinned copy initializes the input before the kernel reads it.
        let mut input = unsafe { overwritten::<u8>(ctx, self.geometry.input_elements)? };
        // SAFETY: each kernel thread writes both temporal positions, including padding,
        // before the output may escape its fence-retained execution context.
        let output = unsafe { overwritten::<f32>(ctx, self.geometry.output_elements)? };
        let mut launch = AsyncKernelLaunch::new(Arc::clone(&self.kernel));
        // SAFETY: the input pointer refers to the checked allocation retained by the context.
        unsafe { launch.push_device_ptr(input.device_pointer().cu_deviceptr()) };
        // SAFETY: the output pointer refers to the checked allocation retained by the context.
        unsafe { launch.push_device_ptr(output.device_pointer().cu_deviceptr()) };
        for value in [
            self.geometry.batch,
            self.geometry.source_height,
            self.geometry.source_width,
            self.geometry.padded_patches,
        ] {
            launch.push_arg(value);
        }
        launch.push_arg(NORMALIZATION);
        launch.push_arg(NORMALIZATION);
        launch.set_launch_config(config(self.geometry.output_elements / TEMPORAL)?);
        self.source.retain(ctx)?;
        ctx.retain(Arc::clone(&self.kernel))?;
        self.source.enqueue(ctx, &self.device, &mut input)?;
        // SAFETY: all arguments match the kernel ABI and stay live until stream completion;
        // the pinned input copy precedes this launch on the same validated stream.
        unsafe { launch.execute(ctx)? };
        Ok(output)
    }
}

impl IntoFuture for Direct {
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
