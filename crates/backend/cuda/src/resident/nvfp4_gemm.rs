//! Model-declared W4A4 projection with block-16 E4M3 activation scales.
mod quant_reuse;
mod workspace;
use crate::mlp::ProjectionWeight as MlpProjectionWeight;
use cutile::prelude::DeviceError;
use cutile::prelude::Scope;
use cutile::prelude::Tensor as PreludeTensor;
use cutile::prelude::TensorView;
pub use infer_cuda_kernels::tile::nvfp4_gemm as kernels;
pub use quant_reuse::QuantPairPlan;
#[cfg(test)]
pub use quant_reuse::with_quant_reuse_disabled;
pub use workspace::Workspace;

/// Record only when the checkpoint declares an activation scale for packed FP4 weights.
pub(super) fn record(
    scope: &Scope,
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &TensorView<'_, f32>,
    output: &mut PreludeTensor<f32>,
    columns: usize,
) -> Result<bool, DeviceError> {
    workspace.clear_quant_reuse();
    record_input(
        scope,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        None,
    )
}

/// The caller supplies the pointer of the tensor from which this exact view was made.
#[expect(
    clippy::too_many_arguments,
    reason = "The certified view adds its actual device pointer to the unchanged projection contract"
)]
pub(super) fn record_with_input_pointer(
    scope: &Scope,
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &TensorView<'_, f32>,
    output: &mut PreludeTensor<f32>,
    columns: usize,
    input_pointer: u64,
) -> Result<bool, DeviceError> {
    record_input(
        scope,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        Some(input_pointer),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "One dispatcher preserves both certified and legacy projection contracts"
)]
fn record_input(
    scope: &Scope,
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &TensorView<'_, f32>,
    output: &mut PreludeTensor<f32>,
    columns: usize,
    input_pointer: Option<u64>,
) -> Result<bool, DeviceError> {
    use super::program::ActivationQuantization;
    use crate::mlp::ProjectionWeight;
    match (weight, input_scale) {
        (ProjectionWeight::Fp4(..), Some(ActivationQuantization::Fp4(scale))) => {
            workspace.record_input(scope, weight, scale, input, output, columns, input_pointer)?;
        }
        (
            ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..),
            Some(ActivationQuantization::Fp8Token),
        ) => {
            workspace.record_fp8_input(scope, weight, input, output, columns, input_pointer)?;
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
