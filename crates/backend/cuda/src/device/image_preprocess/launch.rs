//! One stream-ordered launch retains all checked operands until the device fence.
use super::{
    geometry::{CHANNELS, Geometry},
    staging::Staged,
    tables::Tables,
};
use crate::device::CudaDevice;
use cuda_core::{DType, Function, LaunchConfig};
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
    pub device: CudaDevice,
    pub pixels: Arc<Staged<u8>>,
    pub coefficients: Arc<Staged<i32>>,
    pub geometry: Geometry,
    pub tables: Arc<Tables>,
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

    unsafe fn execute(self, ctx: &ExecutionContext) -> StdResult<Tensor<f32>, DeviceError> {
        if ctx.get_cuda_stream().cu_stream() as usize != self.stream {
            return Err(DeviceError::Internal(
                "image preprocessing stream mismatch".into(),
            ));
        }
        // SAFETY: both inputs are completely initialized by the pinned copies below
        // before either kernel can read them. Their allocations are fence-retained.
        let mut input = unsafe { overwritten::<u8>(ctx, self.geometry.input_elements)? };
        // SAFETY: as above, the coefficients are initialized before the kernels read them.
        let mut coefficients = unsafe { overwritten::<i32>(ctx, self.tables.values.len())? };
        // SAFETY: the horizontal kernel writes every intermediate element, and the vertical
        // kernel writes every output element (including padded rows) before either is read.
        // The allocations are private to this operation and retained until its stream fences.
        let intermediate = unsafe { overwritten::<u8>(ctx, self.geometry.horizontal_elements)? };
        // SAFETY: as above, the vertical kernel initializes the complete output before the
        // tensor can escape the future or be read by a later device operation.
        let output = unsafe { overwritten::<f32>(ctx, self.geometry.output_elements)? };
        let (horizontal, vertical) =
            self.launches(&input, &coefficients, &intermediate, &output)?;
        ctx.retain(Arc::clone(&self.pixels))?;
        ctx.retain(Arc::clone(&self.coefficients))?;
        ctx.retain(Arc::clone(&self.horizontal))?;
        ctx.retain(Arc::clone(&self.vertical))?;
        self.device
            .copy_h2d_pinned(&mut input, self.pixels.buffer()?, self.pixels.len())
            .map_err(|error| DeviceError::Internal(error.to_string()))?;
        self.device
            .copy_h2d_pinned(
                &mut coefficients,
                self.coefficients.buffer()?,
                self.coefficients.len(),
            )
            .map_err(|error| DeviceError::Internal(error.to_string()))?;
        // SAFETY: all pointers and argument ranges were checked before launch. The execution
        // context retains every allocation and both kernel functions until the stream fences;
        // the second kernel follows the first on the same checked CUDA stream.
        unsafe { horizontal.execute(ctx)? };
        // SAFETY: the same retained operands remain live and this launch is ordered after the
        // horizontal kernel on the same stream.
        unsafe { vertical.execute(ctx)? };
        Ok(output)
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
    fn launches(
        &self,
        input: &Tensor<u8>,
        coefficients: &Tensor<i32>,
        intermediate: &Tensor<u8>,
        output: &Tensor<f32>,
    ) -> StdResult<(AsyncKernelLaunch, AsyncKernelLaunch), DeviceError> {
        let base = coefficients.device_pointer().cu_deviceptr();
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
        let input = input.device_pointer().cu_deviceptr();
        let intermediate = intermediate.device_pointer().cu_deviceptr();
        let output = output.device_pointer().cu_deviceptr();
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

/// Allocate storage that the image kernels will fully overwrite before any read.
#[expect(
    unsafe_code,
    reason = "The private tensor is initialized by the checked image kernels"
)]
pub(super) unsafe fn overwritten<T: DType>(
    ctx: &ExecutionContext,
    elements: usize,
) -> StdResult<Tensor<T>, DeviceError> {
    let count = i32::try_from(elements)
        .map_err(|_| DeviceError::Internal("image allocation exceeds tensor ABI".into()))?;
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or_else(|| DeviceError::Internal("image allocation size overflow".into()))?;
    // SAFETY: the execution context owns a valid stream, and the checked non-empty
    // image geometry guarantees the allocation has enough room for every element.
    let pointer = unsafe { ctx.alloc_async(bytes)? };
    // SAFETY: the pointer is the unique allocation just returned by this context; the
    // checked 1-D shape and stride describe exactly `bytes` on its device. No data is
    // read before the caller's kernels initialize the complete allocation.
    let mut tensor = unsafe {
        Tensor::<T>::from_raw_parts(pointer, bytes, ctx.get_device_id(), vec![count], vec![1])
    };
    KernelOutputStored::retain(&(&mut tensor).partition([1]), ctx)?;
    Ok(tensor)
}

pub(super) fn config(elements: usize) -> StdResult<LaunchConfig, DeviceError> {
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
