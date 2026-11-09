//! Model-declared W4A4 projection with block-16 E4M3 activation scales.
#[cfg(test)]
#[cfg(feature = "cuda")]
pub use infer_cuda_kernels::tile::projection::nvfp4_gemm as kernels;
mod quant_reuse;
mod workspace;
use super::projection_lowering::Quantized;
use crate::mlp::ProjectionWeight as MlpProjectionWeight;
use crate::resident::buffers::RowMajor;
use infer_core::Result;
use infer_cuda_kernels::contract::Recording;
use infer_cuda_kernels::contract::{Buffer, Kernels};
pub use quant_reuse::QuantPairPlan;
#[cfg(test)]
pub use quant_reuse::with_quant_reuse_disabled;
pub use workspace::Workspace;

/// Quantize the activations and record the matching low-precision GEMM. A sibling projection
/// cannot reuse the activations.
pub(super) fn record(
    target: (&Kernels, &Recording<'_>),
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    quantized: Quantized,
    input: Buffer<'_, f32>,
    output: RowMajor<'_, f32>,
    columns: usize,
) -> Result<()> {
    workspace.clear_quant_reuse();
    record_input(
        target,
        workspace,
        weight,
        quantized,
        (input, output, columns),
        false,
    )
}

/// [`record`] for a whole matrix, whose quantized activations a sibling projection may reuse.
pub(super) fn record_certified(
    target: (&Kernels, &Recording<'_>),
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    quantized: Quantized,
    input: Buffer<'_, f32>,
    output: RowMajor<'_, f32>,
    columns: usize,
) -> Result<()> {
    record_input(
        target,
        workspace,
        weight,
        quantized,
        (input, output, columns),
        true,
    )
}

fn record_input(
    target: (&Kernels, &Recording<'_>),
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    quantized: Quantized,
    (input, output, columns): (Buffer<'_, f32>, RowMajor<'_, f32>, usize),
    certified: bool,
) -> Result<()> {
    match quantized {
        Quantized::Fp4(scale) => {
            workspace.record_input(target, weight, scale, input, output, columns, certified)
        }
        Quantized::Fp8Token => {
            workspace.record_fp8_input(target, weight, input, output, columns, certified)
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_nvfp4_gemm.rs"]
mod tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_nvfp4_pipeline.rs"]
mod pipeline_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_vendor_fp8.rs"]
mod vendor_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_quant_reuse.rs"]
mod quant_reuse_tests;
