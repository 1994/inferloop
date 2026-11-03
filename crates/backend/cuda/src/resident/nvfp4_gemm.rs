//! Model-declared W4A4 projection with block-16 E4M3 activation scales.
#[cfg(test)]
pub use infer_cuda_kernels::tile::projection::nvfp4_gemm as kernels;
mod quant_reuse;
mod workspace;
use crate::mlp::ProjectionWeight as MlpProjectionWeight;
use cutile::prelude::DeviceError;
use cutile::prelude::Scope;
use cutile::prelude::Tensor as PreludeTensor;
use infer_cuda_kernels::contract::{Buffer, Kernels};
pub use quant_reuse::QuantPairPlan;
#[cfg(test)]
pub use quant_reuse::with_quant_reuse_disabled;
pub use workspace::Workspace;

/// Record only when the checkpoint declares an activation scale for packed FP4 weights.
pub(super) fn record(
    target: (&Kernels, &Scope),
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: Buffer<'_, f32>,
    output: &PreludeTensor<f32>,
    columns: usize,
) -> Result<bool, DeviceError> {
    workspace.clear_quant_reuse();
    record_input(
        target,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        false,
    )
}

/// [`record`] for a whole matrix, whose quantized activations a sibling projection may reuse.
pub(super) fn record_certified(
    target: (&Kernels, &Scope),
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: Buffer<'_, f32>,
    output: &PreludeTensor<f32>,
    columns: usize,
) -> Result<bool, DeviceError> {
    record_input(
        target,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        true,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "One dispatcher preserves both certified and legacy projection contracts"
)]
fn record_input(
    target: (&Kernels, &Scope),
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: Buffer<'_, f32>,
    output: &PreludeTensor<f32>,
    columns: usize,
    certified: bool,
) -> Result<bool, DeviceError> {
    use super::program::ActivationQuantization;
    use crate::mlp::ProjectionWeight;
    match (weight, input_scale) {
        (ProjectionWeight::Fp4(..), Some(ActivationQuantization::Fp4(scale))) => {
            workspace.record_input(target, weight, scale, input, output, columns, certified)?;
        }
        (
            ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..),
            Some(ActivationQuantization::Fp8Token),
        ) => {
            workspace.record_fp8_input(target, weight, input, output, columns, certified)?;
        }
        _ => {
            workspace.clear_quant_reuse();
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "../../tests/unit/resident_nvfp4_gemm.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_nvfp4_pipeline.rs"]
mod pipeline_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_vendor_fp8.rs"]
mod vendor_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_quant_reuse.rs"]
mod quant_reuse_tests;
