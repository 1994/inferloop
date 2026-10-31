//! Audited host/device boundary for one PDL edge: `SwiGLU` -> down projection.
#![expect(
    unsafe_code,
    reason = "Audited PDL boundary: producer signals after stores; every consumer waits with a token before loads and graph owns all storage"
)]
use super::{ProjectionWeight, pdl_consumers::consumers};
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::strategy::LinearTiling;
use cutile::{half::bf16, prelude::*};

pub use infer_cuda_kernels::tile::mlp::pdl as producer;

pub(super) fn record_product(
    scope: &Scope,
    output: &mut Tensor<f32>,
    gate: &Tensor<f32>,
    up: &Tensor<f32>,
) -> Result<(), DeviceError> {
    scope.record(producer::silu_mul(
        output.partition([AUX_KERNEL_TILE]),
        gate,
        up,
    ))?;
    Ok(())
}

pub(super) fn record_down(
    scope: &Scope,
    weights: &ProjectionWeight,
    output: &mut Tensor<f32>,
    input: &Tensor<f32>,
    columns: usize,
    tile: LinearTiling,
) -> Result<(), DeviceError> {
    let mut generics = vec![
        tile.rows().to_string(),
        tile.columns().to_string(),
        columns.to_string(),
    ];
    match weights {
        ProjectionWeight::Dense(w) => {
            generics.insert(0, bf16::DTYPE.as_str().into());
            let launch =
                consumers::dense(output.partition([tile.rows()]), input, w).generics(generics);
            // SAFETY: this kernel waits before reading the producer's activation buffer.
            scope.record(unsafe { launch.programmatic_dependent_launch() })?;
        }
        ProjectionWeight::Fp8(w, s) => {
            let launch =
                consumers::fp8(output.partition([tile.rows()]), input, w, s).generics(generics);
            // SAFETY: same token-ordered activation reads; weights and scales are immutable.
            scope.record(unsafe { launch.programmatic_dependent_launch() })?;
        }
        ProjectionWeight::Fp8Block(..) => {
            return Err(DeviceError::Launch(
                "block FP8 requires the resident FP8 GEMM".to_string(),
            ));
        }
        ProjectionWeight::Fp4(w, s, global) => {
            generics.extend([
                (tile.columns() / 2).to_string(),
                (tile.columns() / NVFP4_GROUP_SIZE).to_string(),
            ]);
            let launch =
                consumers::nvfp4(output.partition([tile.rows()]), input, w, s, global.recip())
                    .generics(generics);
            // SAFETY: same token-ordered activation reads; graph retains all storage.
            scope.record(unsafe { launch.programmatic_dependent_launch() })?;
        }
    }
    Ok(())
}
