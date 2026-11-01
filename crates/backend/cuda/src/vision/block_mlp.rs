//! Device-resident projection, activation and residual for one vision block.
use super::kernels;
use super::program::{COLUMN_TILE, TOKEN_TILE, activation, projection_output};
use super::weights::MlpLayer;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use std::sync::Arc;

#[cfg(test)]
#[path = "../../tests/unit/vision_block_mlp.rs"]
mod tests;

/// Keep both projections and the GELU output on the device until the residual is added.
pub(super) fn run(
    device: &CudaDevice,
    weights: &MlpLayer,
    input: &[f32],
    residual: &[f32],
    rows: usize,
    hidden: usize,
    width: usize,
) -> Result<Vec<f32>> {
    let elements = rows
        .checked_mul(hidden)
        .ok_or_else(|| Error::invalid("vision MLP input overflow"))?;
    if rows == 0
        || hidden == 0
        || width == 0
        || !hidden.is_multiple_of(COLUMN_TILE)
        || !width.is_multiple_of(COLUMN_TILE)
        || input.len() != elements
        || residual.len() != elements
    {
        return Err(Error::invalid("vision MLP input shape"));
    }
    let padded = rows
        .checked_next_multiple_of(TOKEN_TILE)
        .ok_or_else(|| Error::invalid("vision MLP row padding overflow"))?;
    let input = upload(device, input, padded, hidden)?;
    let residual = upload(device, residual, padded, hidden)?;
    let fc1_weight = weights.fc1.0.view(&[width, hidden]).map_err(device_error)?;
    let fc2_weight = weights.fc2.0.view(&[hidden, width]).map_err(device_error)?;
    let mut fc1 = activation(device, padded, width)?;
    let mut activated = activation(device, padded, width)?;
    let mut fc2 = activation(device, padded, hidden)?;
    let mut output = activation(device, padded, hidden)?;
    let graph = Graph::scope(&device.stream, |scope| {
        scope.record(
            kernels::encoder::dense_bias(
                (&mut fc1).partition([TOKEN_TILE, COLUMN_TILE]),
                &input,
                &fc1_weight,
                &weights.fc1.1,
            )
            .generics(vec![hidden.to_string()]),
        )?;
        scope.record(
            kernels::encoder::gelu_tanh(
                (&mut activated).partition([TOKEN_TILE, COLUMN_TILE]),
                &fc1,
            )
            .generics(vec![width.to_string()]),
        )?;
        scope.record(
            kernels::encoder::dense_bias(
                (&mut fc2).partition([TOKEN_TILE, COLUMN_TILE]),
                &activated,
                &fc2_weight,
                &weights.fc2.1,
            )
            .generics(vec![width.to_string()]),
        )?;
        scope.record(
            kernels::encoder::add(
                (&mut output).partition([TOKEN_TILE, COLUMN_TILE]),
                &residual,
                &fc2,
            )
            .generics(vec![hidden.to_string()]),
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    projection_output(device.read(&Arc::new(output))?, rows, hidden)
}

fn upload(
    device: &CudaDevice,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Arc<Tensor<f32>>> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::invalid("vision MLP padded input overflow"))?;
    let mut values = vec![0.0; elements];
    values[..input.len()].copy_from_slice(input);
    device.upload(values, &[rows, columns])
}
