//! Stream-ordered patch construction when the source geometry needs no resize.
use super::{
    geometry::Geometry,
    launch::{THREADS, blocks},
    source::Source,
};
use cuda_core::Function;
use infer_core::Result;
use infer_cuda_kernels::tile::runtime::{Allocation, Argument, KernelLaunch, Operation};
use std::sync::Arc;

/// Mean and standard deviation in the supported symmetric normalization.
const NORMALIZATION: f32 = 0.5;
/// The kernel takes the mean, then the standard deviation.
const NORMALIZATION_FIELDS: usize = 2;
/// Each still-image patch repeats its pixels for two temporal positions.
const TEMPORAL: usize = 2;

pub(super) struct Direct {
    pub source: Source,
    pub geometry: Geometry,
    pub kernel: Arc<Function>,
    pub stream: usize,
}

impl Direct {
    /// Turn the source pixels straight into padded patches.
    #[expect(
        unsafe_code,
        reason = "The private allocations are fully written before their first read, and the launch follows the checked PTX ABI"
    )]
    pub fn enqueue(self, operation: &Operation<'_>) -> Result<Allocation<f32>> {
        // SAFETY: the source copy or decode initializes the input before the kernel reads it.
        let input = unsafe { operation.overwritten::<u8>(self.geometry.input_elements)? };
        // SAFETY: each kernel thread writes both temporal positions, padding included, before
        // the output can escape the operation.
        let output = unsafe { operation.overwritten::<f32>(self.geometry.output_elements)? };
        let mut arguments = vec![
            Argument::buffer(input.buffer()),
            Argument::buffer(output.buffer()),
        ];
        arguments.extend(
            [
                self.geometry.batch,
                self.geometry.source_height,
                self.geometry.source_width,
                self.geometry.padded_patches,
            ]
            .map(Argument::I32),
        );
        arguments.extend([Argument::F32(NORMALIZATION); NORMALIZATION_FIELDS]);
        // SAFETY: both pointers address allocations the operation retains until the stream passes
        // the launch, and the arguments follow the kernel's parameters in order and type.
        let launch = unsafe {
            KernelLaunch::new(
                Arc::clone(&self.kernel),
                self.stream,
                blocks(self.geometry.output_elements / TEMPORAL)?,
                THREADS,
                arguments,
            )
        };
        self.source.retain(operation)?;
        self.source.enqueue(operation, &input)?;
        operation.launch(launch)?;
        Ok(output)
    }
}
