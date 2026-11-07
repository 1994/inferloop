//! Execute compatible vision towers without staging each block on the host.
use super::launches::{Launches, capture};
use super::program::{
    ATTENTION_CONFIG_FIELDS, AttentionMode, COLUMN_TILE, activation, attention_scale,
    attention_tiles, frame_token_count, mlp_width, projection_output, rope_tables,
};
use super::weights::{AttentionLayer, MlpLayer, VisionWeights, half_tile};
use crate::device::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{DenseBuffers, Gelu, VisionDense, VisionNorm, VisionRope};
use infer_cuda_kernels::runtime::Allocation;
use infer_cuda_kernels::runtime::bf16;
use infer_models::vision::row_tile;
use infer_spi::ModalityEncoder;
use std::mem::swap;

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
    norm1_weight: &'a Allocation<bf16>,
    norm1_bias: &'a Allocation<bf16>,
    norm2_weight: &'a Allocation<bf16>,
    norm2_bias: &'a Allocation<bf16>,
    attention: &'a AttentionLayer,
    mlp: &'a MlpLayer,
    fc1: &'a Allocation<bf16>,
    fc2: &'a Allocation<bf16>,
}

impl<'a> Layer<'a> {
    fn new(weights: &'a VisionWeights, layer: usize) -> Result<Self> {
        let prefix = format!("blocks.{layer}.");
        let mlp = weights.mlp(layer)?;
        Ok(Self {
            norm1_weight: weights.get(&format!("{prefix}norm1.weight"))?,
            norm1_bias: weights.get(&format!("{prefix}norm1.bias"))?,
            norm2_weight: weights.get(&format!("{prefix}norm2.weight"))?,
            norm2_bias: weights.get(&format!("{prefix}norm2.bias"))?,
            attention: weights.attention(layer)?,
            mlp,
            fc1: &mlp.fc1.0,
            fc2: &mlp.fc2.0,
        })
    }
}

struct Buffers {
    state: Allocation<f32>,
    next: Allocation<f32>,
    norm: Allocation<f32>,
    q: Allocation<f32>,
    k: Allocation<f32>,
    v: Allocation<f32>,
    q_rot: Allocation<f32>,
    k_rot: Allocation<f32>,
    attended: Allocation<f32>,
    attention_out: Allocation<f32>,
    residual: Allocation<f32>,
    fc1: Allocation<f32>,
    activated: Allocation<f32>,
    fc2: Allocation<f32>,
}

impl Buffers {
    fn new(device: &CudaDevice, geometry: &Geometry, input: &[f32]) -> Result<Self> {
        let elements = geometry
            .rows
            .checked_mul(geometry.hidden)
            .ok_or_else(|| Error::invalid("vision state overflow"))?;
        let mut padded = vec![0.0; elements];
        padded[..input.len()].copy_from_slice(input);
        let state = device.runtime().upload(padded)?;
        let hidden = || activation(device, geometry.rows, geometry.hidden);
        let projection = || activation(device, geometry.rows, geometry.projections);
        let intermediate = || activation(device, geometry.rows, geometry.intermediate);
        Ok(Self {
            state,
            next: hidden()?,
            norm: device.runtime().zeros::<f32>(elements)?,
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
        launches: &Launches<'_>,
        layer: &Layer<'_>,
        geometry: &Geometry,
        cosines: &Allocation<f32>,
        sines: &Allocation<f32>,
    ) -> Result<()> {
        record_norm(
            launches,
            &self.state,
            &self.norm,
            layer.norm1_weight,
            layer.norm1_bias,
            geometry,
        )?;
        self.record_attention(launches, layer, geometry, cosines, sines)?;
        launches.add(
            geometry.hidden,
            buffer(&self.residual)?,
            buffer(&self.state)?,
            buffer(&self.attention_out)?,
        )?;
        record_norm(
            launches,
            &self.residual,
            &self.norm,
            layer.norm2_weight,
            layer.norm2_bias,
            geometry,
        )?;
        self.record_mlp(launches, layer, geometry)?;
        swap(&mut self.state, &mut self.next);
        Ok(())
    }

    fn record_attention(
        &mut self,
        launches: &Launches<'_>,
        layer: &Layer<'_>,
        geometry: &Geometry,
        cosines: &Allocation<f32>,
        sines: &Allocation<f32>,
    ) -> Result<()> {
        let norm = &self.norm;
        for (weight, bias, target) in [
            (&layer.attention.q.0, &layer.attention.q.1, &mut self.q),
            (&layer.attention.k.0, &layer.attention.k.1, &mut self.k),
            (&layer.attention.v.0, &layer.attention.v.1, &mut self.v),
        ] {
            launches.dense_bias(VisionDense {
                out: buffer(target)?,
                input: buffer(norm)?,
                weight: buffer(weight)?,
                bias: buffer(bias)?,
                columns: geometry.hidden,
            })?;
        }
        for (input, output) in [(&self.q, &mut self.q_rot), (&self.k, &mut self.k_rot)] {
            launches.rope(VisionRope {
                out: buffer(output)?,
                input: buffer(input)?,
                cosines: buffer(cosines)?,
                sines: buffer(sines)?,
                rows: geometry.rows,
                half: geometry.half,
                real_half: None,
            })?;
        }
        launches.dense(
            super::program::dense_plan(
                geometry.width,
                AttentionMode::Online,
                geometry.tiles,
                geometry.rows,
                geometry.tokens,
                geometry.scale,
                geometry.frame_tokens,
            ),
            DenseBuffers {
                out: buffer(&self.attended)?,
                query: buffer(&self.q_rot)?,
                key: buffer(&self.k_rot)?,
                value: buffer(&self.v)?,
            },
        )?;
        launches.dense_bias(VisionDense {
            out: buffer(&self.attention_out)?,
            input: buffer(&self.attended)?,
            weight: buffer(&layer.attention.proj.0)?,
            bias: buffer(&layer.attention.proj.1)?,
            columns: geometry.projections,
        })?;
        Ok(())
    }

    fn record_mlp(
        &self,
        launches: &Launches<'_>,
        layer: &Layer<'_>,
        geometry: &Geometry,
    ) -> Result<()> {
        let norm = &self.norm;
        launches.dense_bias(VisionDense {
            out: buffer(&self.fc1)?,
            input: buffer(norm)?,
            weight: buffer(layer.fc1)?,
            bias: buffer(&layer.mlp.fc1.1)?,
            columns: geometry.hidden,
        })?;
        launches.gelu(
            Gelu::Tanh,
            geometry.intermediate,
            buffer(&self.activated)?,
            buffer(&self.fc1)?,
        )?;
        launches.dense_bias(VisionDense {
            out: buffer(&self.fc2)?,
            input: buffer(&self.activated)?,
            weight: buffer(layer.fc2)?,
            bias: buffer(&layer.mlp.fc2.1)?,
            columns: geometry.intermediate,
        })?;
        launches.add(
            geometry.hidden,
            buffer(&self.next)?,
            buffer(&self.residual)?,
            buffer(&self.fc2)?,
        )?;
        Ok(())
    }
}

fn record_norm(
    launches: &Launches<'_>,
    input: &Allocation<f32>,
    output: &Allocation<f32>,
    weight: &Allocation<bf16>,
    bias: &Allocation<bf16>,
    geometry: &Geometry,
) -> Result<()> {
    launches.layernorm(VisionNorm {
        out: buffer(output)?,
        input: buffer(input)?,
        weight: buffer(weight)?,
        bias: buffer(bias)?,
        dim: geometry.hidden,
        tile: geometry.hidden,
        epsilon: geometry.epsilon,
    })?;
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
        .map(|index| Layer::new(weights, index))
        .collect::<Result<Vec<_>>>()?;
    let (cosines, sines) = rope_tables(encoder, grid, geometry.rows)?;
    let cosines = device.runtime().upload(cosines)?;
    let sines = device.runtime().upload(sines)?;
    let mut buffers = Buffers::new(device, &geometry, input)?;
    let graph = capture(device, |launches| {
        for layer in &layers {
            buffers.record(launches, layer, &geometry, &cosines, &sines)?;
        }
        Ok(())
    })?;
    device.tile().sync(graph.launch())?;
    projection_output(
        device.runtime().read(&buffers.state)?,
        tokens,
        geometry.hidden,
    )
}
