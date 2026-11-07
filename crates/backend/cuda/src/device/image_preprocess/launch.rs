//! One stream-ordered operation retains all checked operands until the device has passed it.
use super::{
    geometry::{CHANNELS, Geometry},
    source::Source,
    staging::Staged,
    tables::Tables,
};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Function;
use infer_cuda_kernels::tile::runtime::{Allocation, Argument, KernelLaunch, Operation};
use std::sync::Arc;

/// Threads per one-dimensional image kernel block.
pub(super) const THREADS: u32 = 256;
/// Canonical RGB rescaling and symmetric normalization both use one half.
const CHANNEL_NORMALIZATION: f32 = 0.5;
/// Mean and standard deviation for each channel.
const NORMALIZATION_FIELDS: usize = CHANNELS * 2;

pub(super) struct Kernels {
    pub source: Source,
    pub coefficients: Arc<Staged<i32>>,
    pub geometry: Geometry,
    pub tables: Arc<Tables>,
    pub horizontal: Arc<Function>,
    pub vertical: Arc<Function>,
    pub stream: usize,
}

impl Kernels {
    /// Resize through the intermediate rows into padded patches.
    #[expect(
        unsafe_code,
        reason = "Each private allocation is fully written by the copy or kernel ordered before its first read"
    )]
    pub fn enqueue(self, operation: &Operation<'_>) -> Result<Allocation<f32>> {
        // SAFETY: the source copy or decode initializes the input before either kernel reads it.
        let input = unsafe { operation.overwritten::<u8>(self.geometry.input_elements)? };
        // SAFETY: the pinned upload below initializes the coefficients before the kernels run.
        let coefficients = unsafe { operation.overwritten::<i32>(self.tables.values.len())? };
        // SAFETY: the horizontal kernel writes every intermediate element before the vertical
        // kernel reads it.
        let intermediate =
            unsafe { operation.overwritten::<u8>(self.geometry.horizontal_elements)? };
        // SAFETY: the vertical kernel writes every output element, padded rows included, before
        // the output can escape the operation.
        let output = unsafe { operation.overwritten::<f32>(self.geometry.output_elements)? };
        let (horizontal, vertical) =
            self.launches(&input, &coefficients, &intermediate, &output)?;
        self.source.retain(operation)?;
        operation.retain(Arc::clone(&self.coefficients))?;
        self.source.enqueue(operation, &input)?;
        // SAFETY: the staged coefficients are retained above and return to their pool, the only
        // writer, after the operation completes.
        unsafe {
            operation.upload(
                &coefficients,
                self.coefficients.buffer()?,
                self.coefficients.len(),
            )?;
        }
        operation.launch(horizontal)?;
        operation.launch(vertical)?;
        Ok(output)
    }

    #[expect(
        unsafe_code,
        reason = "Arguments follow the validated image kernel ABI"
    )]
    fn launches(
        &self,
        input: &Allocation<u8>,
        coefficients: &Allocation<i32>,
        intermediate: &Allocation<u8>,
        output: &Allocation<f32>,
    ) -> Result<(KernelLaunch, KernelLaunch)> {
        let base = coefficients.buffer().address();
        let offsets = self.tables.offsets.map(|offset| base.checked_add(offset));
        let [
            Some(x_indices),
            Some(x_weights),
            Some(y_indices),
            Some(y_weights),
        ] = offsets
        else {
            return Err(Error::invalid("image coefficient address overflow"));
        };
        let input = Argument::buffer(input.buffer());
        let intermediate = Argument::buffer(intermediate.buffer());
        let output = Argument::buffer(output.buffer());
        let geometry = &self.geometry;
        let mut arguments = vec![
            input,
            Argument::Address(x_indices),
            Argument::Address(x_weights),
            intermediate,
        ];
        arguments.extend(
            [
                geometry.batch,
                geometry.source_height,
                geometry.source_width,
                geometry.target_width,
                self.tables.horizontal_taps,
                self.tables.horizontal_denominator,
            ]
            .map(Argument::I32),
        );
        // SAFETY: the four pointers address the input, both horizontal tables and the
        // intermediate, which the operation retains until the stream passes the launch; the
        // integers follow them in the PTX's order and types.
        let horizontal = unsafe {
            KernelLaunch::new(
                Arc::clone(&self.horizontal),
                self.stream,
                blocks(geometry.horizontal_elements)?,
                THREADS,
                arguments,
            )
        };
        let mut arguments = vec![
            intermediate,
            Argument::Address(y_indices),
            Argument::Address(y_weights),
            output,
        ];
        arguments.extend(
            [
                geometry.batch,
                geometry.source_height,
                geometry.target_height,
                geometry.target_width,
                self.tables.vertical_taps,
                self.tables.vertical_denominator,
                geometry.padded_patches,
            ]
            .map(Argument::I32),
        );
        arguments.extend([Argument::F32(CHANNEL_NORMALIZATION); NORMALIZATION_FIELDS]);
        // SAFETY: as above for the intermediate, vertical tables and an output sized for every
        // padded patch, with table offsets checked at construction.
        let vertical = unsafe {
            KernelLaunch::new(
                Arc::clone(&self.vertical),
                self.stream,
                blocks(geometry.output_elements)?,
                THREADS,
                arguments,
            )
        };
        Ok((horizontal, vertical))
    }
}

/// Blocks of [`THREADS`] covering `elements` threads.
pub(super) fn blocks(elements: usize) -> Result<u32> {
    let threads = usize::try_from(THREADS).map_err(|_| Error::invalid("image launch threads"))?;
    u32::try_from(elements.div_ceil(threads)).map_err(|_| Error::invalid("image launch grid"))
}
