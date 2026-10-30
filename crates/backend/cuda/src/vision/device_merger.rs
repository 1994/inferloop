//! Keep the vision merger's intermediate activations on the owning device.
use super::kernels;
use super::program::{COLUMN_TILE, TOKEN_TILE, activation, projection_output};
use super::weights::VisionWeights;
use crate::device::{CudaDevice, device_error};
use cutile::cuda_async::error::DeviceError;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_models::vision::{merge_unit, merged_width, row_tile};
use infer_spi::ModalityEncoder;
use std::sync::Arc;

pub(super) fn supports(encoder: &ModalityEncoder) -> Result<bool> {
    let hidden = encoder.hidden_size;
    let merged = merged_width(encoder)?;
    Ok(hidden > 0
        && encoder.out_hidden_size > 0
        && row_tile(encoder)? == hidden
        && hidden.is_multiple_of(COLUMN_TILE)
        && merged.is_multiple_of(COLUMN_TILE)
        && encoder.out_hidden_size.is_multiple_of(COLUMN_TILE))
}

struct Geometry {
    hidden: usize,
    merged: usize,
    out: usize,
    rows: usize,
    padded: usize,
    padded_patches: usize,
    elements: usize,
}

impl Geometry {
    fn new(encoder: &ModalityEncoder, input_len: usize, tokens: usize) -> Result<Self> {
        let hidden = encoder.hidden_size;
        let unit = merge_unit(encoder)?;
        let merged = merged_width(encoder)?;
        let out = encoder.out_hidden_size;
        let expected = tokens
            .checked_mul(hidden)
            .ok_or_else(|| Error::invalid("vision merger input overflow"))?;
        if tokens == 0
            || !tokens.is_multiple_of(unit)
            || input_len != expected
            || !supports(encoder)?
        {
            return Err(Error::invalid("vision merger input shape"));
        }
        let rows = tokens / unit;
        let padded = rows
            .checked_next_multiple_of(TOKEN_TILE)
            .ok_or_else(|| Error::invalid("vision merger row padding overflow"))?;
        let padded_patches = padded
            .checked_mul(unit)
            .ok_or_else(|| Error::invalid("vision merger patch padding overflow"))?;
        let elements = padded
            .checked_mul(merged)
            .ok_or_else(|| Error::invalid("vision merger activation overflow"))?;
        elements
            .checked_mul(size_of::<f32>())
            .and_then(|_| padded.checked_mul(out))
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .ok_or_else(|| Error::invalid("vision merger activation overflow"))?;
        Ok(Self {
            hidden,
            merged,
            out,
            rows,
            padded,
            padded_patches,
            elements,
        })
    }
}

/// Run normalization and both projections without reading intermediate rows back to the host.
pub(super) fn run(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    input: &[f32],
    tokens: usize,
) -> Result<Vec<f32>> {
    let geometry = Geometry::new(encoder, input.len(), tokens)?;
    let Geometry {
        hidden,
        merged,
        out,
        rows,
        padded,
        padded_patches,
        elements,
    } = geometry;
    let mut values = vec![0.0_f32; elements];
    values[..input.len()].copy_from_slice(input);
    let input = device.upload(values, &[padded_patches, hidden])?;
    let mut norm = api::zeros::<f32>(&[elements])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut fc1 = activation(device, padded, merged)?;
    let mut activated = activation(device, padded, merged)?;
    let mut output = activation(device, padded, out)?;
    let norm_weight = weights.get("merger.norm.weight")?;
    let norm_bias = weights.get("merger.norm.bias")?;
    let projections = weights.merger()?;
    let fc1_weight = projections
        .fc1
        .0
        .view(&[merged, merged])
        .map_err(device_error)?;
    let fc2_weight = projections
        .fc2
        .0
        .view(&[out, merged])
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        // Padded patches form complete independent merge rows; their outputs are discarded.
        scope.record(
            kernels::encoder::layernorm(
                (&mut norm).partition([hidden]),
                &input,
                norm_weight,
                norm_bias,
                encoder.norm_epsilon,
            )
            .generics(vec![hidden.to_string(), hidden.to_string()]),
        )?;
        let grouped = norm
            .view(&[padded, merged])
            .map_err(|error| DeviceError::Launch(error.to_string()))?;
        scope.record(
            kernels::encoder::dense_bias(
                (&mut fc1).partition([TOKEN_TILE, COLUMN_TILE]),
                &grouped,
                &fc1_weight,
                &projections.fc1.1,
            )
            .generics(vec![merged.to_string()]),
        )?;
        scope.record(
            kernels::encoder::gelu_erf((&mut activated).partition([TOKEN_TILE, COLUMN_TILE]), &fc1)
                .generics(vec![merged.to_string()]),
        )?;
        scope.record(
            kernels::encoder::dense_bias(
                (&mut output).partition([TOKEN_TILE, COLUMN_TILE]),
                &activated,
                &fc2_weight,
                &projections.fc2.1,
            )
            .generics(vec![merged.to_string()]),
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    projection_output(device.read(&Arc::new(output))?, rows, out)
}
