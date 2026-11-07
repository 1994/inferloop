//! Device-resident projection, activation and residual for one vision block.
use super::launches::capture;
use super::program::{COLUMN_TILE, TOKEN_TILE, activation, projection_output};
use super::weights::MlpLayer;
use crate::device::CudaDevice;
use crate::resident::buffers::buffer;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{Gelu, VisionDense};
use infer_cuda_kernels::runtime::Allocation;

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
    let fc1_weight = &weights.fc1.0;
    let fc2_weight = &weights.fc2.0;
    let fc1 = activation(device, padded, width)?;
    let activated = activation(device, padded, width)?;
    let fc2 = activation(device, padded, hidden)?;
    let output = activation(device, padded, hidden)?;
    let graph = capture(device, |launches| {
        launches.dense_bias(VisionDense {
            out: buffer(&fc1)?,
            input: buffer(&input)?,
            weight: buffer(fc1_weight)?,
            bias: buffer(&weights.fc1.1)?,
            columns: hidden,
        })?;
        launches.gelu(Gelu::Tanh, width, buffer(&activated)?, buffer(&fc1)?)?;
        launches.dense_bias(VisionDense {
            out: buffer(&fc2)?,
            input: buffer(&activated)?,
            weight: buffer(fc2_weight)?,
            bias: buffer(&weights.fc2.1)?,
            columns: width,
        })?;
        launches.add(hidden, buffer(&output)?, buffer(&residual)?, buffer(&fc2)?)?;
        Ok(())
    })?;
    device.tile().sync(graph.launch())?;
    projection_output(device.runtime().read(&output)?, rows, hidden)
}

fn upload(
    device: &CudaDevice,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Allocation<f32>> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::invalid("vision MLP padded input overflow"))?;
    let mut values = vec![0.0; elements];
    values[..input.len()].copy_from_slice(input);
    device.runtime().upload(values)
}
