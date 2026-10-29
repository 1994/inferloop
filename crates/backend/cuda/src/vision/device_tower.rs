//! Execute compatible vision towers without staging each block on the host.
use super::kernels;
use super::program::{
    ATTENTION_CONFIG_FIELDS, AttentionMode, COLUMN_TILE, TOKEN_TILE, activation,
    attention_generics, attention_scale, attention_tiles, frame_token_count, mlp_width,
    projection_output, rope_tables,
};
use super::weights::{AttentionLayer, MlpLayer, VisionWeights, half_tile};
use crate::attention::kernels::sdpa::attention as sdpa_attention;
use crate::device::{CudaDevice, device_error};
use cutile::cuda_async::cuda_graph::Scope;
use cutile::cuda_async::error::DeviceError;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_models::vision::row_tile;
use infer_spi::ModalityEncoder;
use std::mem::swap;
use std::result::Result as StdResult;

/// Compact head packing and padded norm rows use the separately qualified block path.
const COMPACT_HALF: usize = 48;

pub(super) fn supports(encoder: &ModalityEncoder) -> Result<bool> {
    Ok(encoder.depth > 0
        && encoder.hidden_size.is_multiple_of(COLUMN_TILE)
        && row_tile(encoder)? == encoder.hidden_size
        && half_tile(encoder)? != COMPACT_HALF)
}

struct Geometry {
    rows: usize,
    hidden: usize,
    intermediate: usize,
    half: usize,
    width: usize,
    projections: usize,
    tokens: i32,
    frame_tokens: i32,
    tiles: [usize; ATTENTION_CONFIG_FIELDS],
    scale: f32,
    epsilon: f32,
}

impl Geometry {
    fn new(encoder: &ModalityEncoder, grid: (usize, usize, usize), tokens: usize) -> Result<Self> {
        let half = half_tile(encoder)?;
        let width = half
            .checked_mul(2)
            .ok_or_else(|| Error::invalid("vision half-head overflow"))?;
        let projections = encoder
            .heads
            .checked_mul(width)
            .ok_or_else(|| Error::invalid("vision projection overflow"))?;
        if !projections.is_multiple_of(COLUMN_TILE) {
            return Err(Error::unsupported(
                "vision projection geometry must tile by 64",
            ));
        }
        let tiles = attention_tiles(tokens);
        let rows = tokens
            .checked_next_multiple_of(tiles[0])
            .ok_or_else(|| Error::invalid("vision row padding overflow"))?;
        Ok(Self {
            rows,
            hidden: encoder.hidden_size,
            intermediate: mlp_width(encoder)?,
            half,
            width,
            projections,
            tokens: i32::try_from(tokens).map_err(device_error)?,
            frame_tokens: frame_token_count(grid, tokens)?,
            tiles,
            scale: attention_scale(encoder)?,
            epsilon: encoder.norm_epsilon,
        })
    }
}

struct Layer<'a> {
    norm1_weight: &'a Arc<Tensor<bf16>>,
    norm1_bias: &'a Arc<Tensor<bf16>>,
    norm2_weight: &'a Arc<Tensor<bf16>>,
    norm2_bias: &'a Arc<Tensor<bf16>>,
    attention: &'a AttentionLayer,
    mlp: &'a MlpLayer,
    fc1: TensorView<'a, bf16>,
    fc2: TensorView<'a, bf16>,
}

impl<'a> Layer<'a> {
    fn new(weights: &'a VisionWeights, layer: usize, geometry: &Geometry) -> Result<Self> {
        let prefix = format!("blocks.{layer}.");
        let mlp = weights.mlp(layer)?;
        Ok(Self {
            norm1_weight: weights.get(&format!("{prefix}norm1.weight"))?,
            norm1_bias: weights.get(&format!("{prefix}norm1.bias"))?,
            norm2_weight: weights.get(&format!("{prefix}norm2.weight"))?,
            norm2_bias: weights.get(&format!("{prefix}norm2.bias"))?,
            attention: weights.attention(layer)?,
            mlp,
            fc1: mlp
                .fc1
                .0
                .view(&[geometry.intermediate, geometry.hidden])
                .map_err(device_error)?,
            fc2: mlp
                .fc2
                .0
                .view(&[geometry.hidden, geometry.intermediate])
                .map_err(device_error)?,
        })
    }
}

struct Buffers {
    state: Tensor<f32>,
    next: Tensor<f32>,
    norm: Tensor<f32>,
    q: Tensor<f32>,
    k: Tensor<f32>,
    v: Tensor<f32>,
    q_rot: Tensor<f32>,
    k_rot: Tensor<f32>,
    attended: Tensor<f32>,
    attention_out: Tensor<f32>,
    residual: Tensor<f32>,
    fc1: Tensor<f32>,
    activated: Tensor<f32>,
    fc2: Tensor<f32>,
}

impl Buffers {
    fn new(device: &CudaDevice, geometry: &Geometry, input: &[f32]) -> Result<Self> {
        let elements = geometry
            .rows
            .checked_mul(geometry.hidden)
            .ok_or_else(|| Error::invalid("vision state overflow"))?;
        let mut padded = vec![0.0; elements];
        padded[..input.len()].copy_from_slice(input);
        let state = Arc::try_unwrap(device.upload(padded, &[geometry.rows, geometry.hidden])?)
            .map_err(|_| Error::invariant("vision state upload unexpectedly shared"))?;
        let hidden = || activation(device, geometry.rows, geometry.hidden);
        let projection = || activation(device, geometry.rows, geometry.projections);
        let intermediate = || activation(device, geometry.rows, geometry.intermediate);
        Ok(Self {
            state,
            next: hidden()?,
            norm: api::zeros::<f32>(&[elements])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            q: projection()?,
            k: projection()?,
            v: projection()?,
            q_rot: projection()?,
            k_rot: projection()?,
            attended: projection()?,
            attention_out: hidden()?,
            residual: hidden()?,
            fc1: intermediate()?,
            activated: intermediate()?,
            fc2: hidden()?,
        })
    }

    fn record(
        &mut self,
        scope: &Scope,
        layer: &Layer<'_>,
        geometry: &Geometry,
        cosines: &Arc<Tensor<f32>>,
        sines: &Arc<Tensor<f32>>,
    ) -> StdResult<(), DeviceError> {
        record_norm(
            scope,
            &self.state,
            &mut self.norm,
            layer.norm1_weight,
            layer.norm1_bias,
            geometry,
        )?;
        self.record_attention(scope, layer, geometry, cosines, sines)?;
        scope.record(
            kernels::encoder::add(
                (&mut self.residual).partition([TOKEN_TILE, COLUMN_TILE]),
                &self.state,
                &self.attention_out,
            )
            .generics(vec![geometry.hidden.to_string()]),
        )?;
        record_norm(
            scope,
            &self.residual,
            &mut self.norm,
            layer.norm2_weight,
            layer.norm2_bias,
            geometry,
        )?;
        self.record_mlp(scope, layer, geometry)?;
        swap(&mut self.state, &mut self.next);
        Ok(())
    }

    fn record_attention(
        &mut self,
        scope: &Scope,
        layer: &Layer<'_>,
        geometry: &Geometry,
        cosines: &Arc<Tensor<f32>>,
        sines: &Arc<Tensor<f32>>,
    ) -> StdResult<(), DeviceError> {
        let norm = self
            .norm
            .view(&[geometry.rows, geometry.hidden])
            .map_err(|error| DeviceError::Launch(error.to_string()))?;
        for (weight, bias, target) in [
            (&layer.attention.q.0, &layer.attention.q.1, &mut self.q),
            (&layer.attention.k.0, &layer.attention.k.1, &mut self.k),
            (&layer.attention.v.0, &layer.attention.v.1, &mut self.v),
        ] {
            scope.record(
                kernels::encoder::dense_bias(
                    target.partition([TOKEN_TILE, COLUMN_TILE]),
                    &norm,
                    weight,
                    bias,
                )
                .generics(vec![geometry.hidden.to_string()]),
            )?;
        }
        for (input, output) in [(&self.q, &mut self.q_rot), (&self.k, &mut self.k_rot)] {
            scope.record(
                kernels::encoder::rope(
                    output.partition([TOKEN_TILE, geometry.half]),
                    input,
                    cosines,
                    sines,
                )
                .generics(vec![geometry.half.to_string()]),
            )?;
        }
        scope.record(
            sdpa_attention(
                (&mut self.attended).partition([geometry.tiles[0], geometry.width]),
                &self.q_rot,
                &self.k_rot,
                &self.v,
                geometry.tokens,
                geometry.scale,
                geometry.frame_tokens,
                geometry.tokens,
                0,
                0,
                0,
            )
            .generics(attention_generics(
                geometry.width,
                AttentionMode::Online,
                geometry.tiles,
            )),
        )?;
        scope.record(
            kernels::encoder::dense_bias(
                (&mut self.attention_out).partition([TOKEN_TILE, COLUMN_TILE]),
                &self.attended,
                &layer.attention.proj.0,
                &layer.attention.proj.1,
            )
            .generics(vec![geometry.projections.to_string()]),
        )?;
        Ok(())
    }

    fn record_mlp(
        &mut self,
        scope: &Scope,
        layer: &Layer<'_>,
        geometry: &Geometry,
    ) -> StdResult<(), DeviceError> {
        let norm = self
            .norm
            .view(&[geometry.rows, geometry.hidden])
            .map_err(|error| DeviceError::Launch(error.to_string()))?;
        scope.record(
            kernels::encoder::dense_bias(
                (&mut self.fc1).partition([TOKEN_TILE, COLUMN_TILE]),
                &norm,
                &layer.fc1,
                &layer.mlp.fc1.1,
            )
            .generics(vec![geometry.hidden.to_string()]),
        )?;
        scope.record(
            kernels::encoder::gelu_tanh(
                (&mut self.activated).partition([TOKEN_TILE, COLUMN_TILE]),
                &self.fc1,
            )
            .generics(vec![geometry.intermediate.to_string()]),
        )?;
        scope.record(
            kernels::encoder::dense_bias(
                (&mut self.fc2).partition([TOKEN_TILE, COLUMN_TILE]),
                &self.activated,
                &layer.fc2,
                &layer.mlp.fc2.1,
            )
            .generics(vec![geometry.intermediate.to_string()]),
        )?;
        scope.record(
            kernels::encoder::add(
                (&mut self.next).partition([TOKEN_TILE, COLUMN_TILE]),
                &self.residual,
                &self.fc2,
            )
            .generics(vec![geometry.hidden.to_string()]),
        )?;
        Ok(())
    }
}

fn record_norm(
    scope: &Scope,
    input: &Tensor<f32>,
    output: &mut Tensor<f32>,
    weight: &Arc<Tensor<bf16>>,
    bias: &Arc<Tensor<bf16>>,
    geometry: &Geometry,
) -> StdResult<(), DeviceError> {
    scope.record(
        kernels::encoder::layernorm(
            output.partition([geometry.hidden]),
            input,
            weight,
            bias,
            geometry.epsilon,
        )
        .generics(vec![
            geometry.hidden.to_string(),
            geometry.hidden.to_string(),
        ]),
    )?;
    Ok(())
}

pub(super) fn run(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    input: &[f32],
    grid: (usize, usize, usize),
    tokens: usize,
) -> Result<Vec<f32>> {
    let elements = tokens
        .checked_mul(encoder.hidden_size)
        .ok_or_else(|| Error::invalid("vision input overflow"))?;
    if tokens == 0 || input.len() != elements {
        return Err(Error::invalid("vision tower input shape"));
    }
    let geometry = Geometry::new(encoder, grid, tokens)?;
    let layers = (0..encoder.depth)
        .map(|index| Layer::new(weights, index, &geometry))
        .collect::<Result<Vec<_>>>()?;
    let (cosines, sines) = rope_tables(encoder, grid, geometry.rows)?;
    let cosines = device.upload(cosines, &[geometry.rows, geometry.width])?;
    let sines = device.upload(sines, &[geometry.rows, geometry.width])?;
    let mut buffers = Buffers::new(device, &geometry, input)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        for layer in &layers {
            buffers.record(scope, layer, &geometry, &cosines, &sines)?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    projection_output(
        device.read_borrowed(&buffers.state)?,
        tokens,
        geometry.hidden,
    )
}
