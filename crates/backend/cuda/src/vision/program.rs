//! Captured vision-encoder programs.
use super::block_mlp;
use super::device_tower;
use super::launches::{Launches, capture};
use super::{VisionWeights, weights::AttentionLayer, weights::half_tile};
#[cfg(test)]
use crate::attention::benchmark::REPEATS;
#[cfg(test)]
use crate::attention::benchmark::measure;
use crate::attention::tile_rows;
use crate::device::vision_position::DeviceTaps;
use crate::device::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use cutile::cuda_async::error::DeviceError;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    DenseBuffers, DensePlan, Gelu, Pack, VisionDense, VisionNorm, VisionRope,
};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_models::vision::PositionTaps;
use infer_models::vision::head_dim;
use infer_models::vision::merge_unit;
use infer_models::vision::merged_width;
use infer_models::vision::position_ids;
use infer_models::vision::rope_frequencies;
use infer_models::vision::row_tile;
use infer_spi::ModalityEncoder;
use std::env::var;
use std::result::Result as StdResult;

#[cfg(test)]
#[path = "../../tests/unit/vision_projection_output.rs"]
mod projection_output_tests;

/// Token tile of every vision projection, matching the prompt GEMM.
pub(super) const TOKEN_TILE: usize = 32;
/// Column tile of every vision projection, matching the prompt GEMM.
pub(super) const COLUMN_TILE: usize = 64;

/// Column tile of every vision projection; test-only audit hook.
#[cfg(test)]
#[must_use]
pub(super) const fn column_tile() -> usize {
    COLUMN_TILE
}
pub(super) const ATTENTION_CONFIG_FIELDS: usize = 3;
/// Compact physical half/full head and qualified wide pack width.
const COMPACT_HALF: usize = 48;
const COMPACT_REAL_HALF: usize = 36;
const COMPACT_WIDTH: usize = 96;
const WIDE_WIDTH: usize = 128;

/// Flattened width of one patch: `channels × temporal × patch × patch`.
/// # Errors
/// Rejects overflowing patch geometry.
pub fn patch_width(encoder: &ModalityEncoder) -> Result<usize> {
    encoder
        .in_channels
        .checked_mul(encoder.temporal_patch_size)
        .and_then(|n| n.checked_mul(encoder.patch_size))
        .and_then(|n| n.checked_mul(encoder.patch_size))
        .ok_or_else(|| Error::invalid("patch width overflow"))
}

/// Patch-embed flattened patches and add the gathered position embeddings.
///
/// `positions` may be empty, in which case only the projection runs; otherwise it must hold
/// `patch_count × hidden_size` values from `infer_models::vision::gather_positions`.
///
/// # Errors
/// Rejects patch or position input that disagrees with the declared geometry, or failed CUDA
/// capture/launch.
pub fn patch_embed(
    device: &CudaDevice,
    encoder: &ModalityEncoder,
    weights: &VisionWeights,
    patches: &[f32],
    positions: &[f32],
    patch_count: usize,
) -> Result<Vec<f32>> {
    patch_embed_with_positions(
        device,
        encoder,
        weights,
        PatchInput::Host(patches),
        PositionInput::Host(positions),
        patch_count,
    )
}

/// Embed patches while interpolating the learned position table on the device.
///
/// # Errors
/// Rejects mismatched input geometry or a failed CUDA operation.
pub fn patch_embed_device_positions(
    device: &CudaDevice,
    encoder: &ModalityEncoder,
    weights: &VisionWeights,
    patches: &[f32],
    positions: &PositionTaps,
    patch_count: usize,
) -> Result<Vec<f32>> {
    patch_embed_with_positions(
        device,
        encoder,
        weights,
        PatchInput::Host(patches),
        PositionInput::Device(positions),
        patch_count,
    )
}

/// Embed already-preprocessed device patches without reading them back or uploading them again.
pub fn patch_embed_device_pixels(
    device: &CudaDevice,
    encoder: &ModalityEncoder,
    weights: &VisionWeights,
    patches: &Allocation<f32>,
    positions: &PositionTaps,
    patch_count: usize,
) -> Result<Vec<f32>> {
    patch_embed_with_positions(
        device,
        encoder,
        weights,
        PatchInput::Device(patches),
        PositionInput::Device(positions),
        patch_count,
    )
}

#[derive(Clone, Copy)]
enum PatchInput<'a> {
    Host(&'a [f32]),
    Device(&'a Allocation<f32>),
}

#[derive(Clone, Copy)]
enum PositionInput<'a> {
    Host(&'a [f32]),
    Device(&'a PositionTaps),
}

struct StagedPositions {
    host: Option<Allocation<f32>>,
    device: Option<DeviceTaps>,
}

impl StagedPositions {
    fn new(
        device: &CudaDevice,
        encoder: &ModalityEncoder,
        weights: &VisionWeights,
        positions: PositionInput<'_>,
        patch_count: usize,
        padded: usize,
    ) -> Result<Self> {
        let hidden = encoder.hidden_size;
        let (host, device_positions) = match positions {
            PositionInput::Host([]) => (None, None),
            PositionInput::Host(positions) => {
                let elements = patch_count
                    .checked_mul(hidden)
                    .ok_or_else(|| Error::invalid("position input overflow"))?;
                let padded_elements = padded
                    .checked_mul(hidden)
                    .ok_or_else(|| Error::invalid("position padding overflow"))?;
                if positions.len() != elements {
                    return Err(Error::invalid("position input shape"));
                }
                let mut values = vec![0.0f32; padded_elements];
                values[..positions.len()].copy_from_slice(positions);
                (Some(device.runtime().upload(values)?), None)
            }
            PositionInput::Device(taps) => {
                if taps.indices.len() != patch_count {
                    return Err(Error::invalid("position input shape"));
                }
                let table = weights.get("pos_embed.weight")?;
                let staged = DeviceTaps::new(
                    device,
                    &taps.indices,
                    &taps.weights,
                    table,
                    encoder.position_embeddings,
                    hidden,
                    padded,
                )?;
                (None, Some(staged))
            }
        };
        Ok(Self {
            host,
            device: device_positions,
        })
    }

    const fn present(&self) -> bool {
        self.host.is_some() || self.device.is_some()
    }

    fn record(
        &self,
        launches: &Launches<'_>,
        embedded: &Allocation<f32>,
        output: &Allocation<f32>,
        hidden: usize,
    ) -> StdResult<(), DeviceError> {
        if let Some(positions) = &self.host {
            launches.add(
                hidden,
                buffer(output)?,
                buffer(embedded)?,
                buffer(positions)?,
            )?;
        }
        if let Some(taps) = &self.device {
            launches.add_positions(buffer(output)?, buffer(embedded)?, taps.taps())?;
        }
        Ok(())
    }
}

fn patch_embed_with_positions(
    device: &CudaDevice,
    encoder: &ModalityEncoder,
    weights: &VisionWeights,
    patches: PatchInput<'_>,
    positions: PositionInput<'_>,
    patch_count: usize,
) -> Result<Vec<f32>> {
    let width = patch_width(encoder)?;
    let hidden = encoder.hidden_size;
    let patch_elements = patch_count
        .checked_mul(width)
        .ok_or_else(|| Error::invalid("patch input overflow"))?;
    if patch_count == 0 {
        return Err(Error::invalid("patch input shape"));
    }
    if !hidden.is_multiple_of(COLUMN_TILE) || !width.is_multiple_of(COLUMN_TILE) {
        return Err(Error::unsupported(
            "vision patch/hidden widths must tile by the GEMM column size",
        ));
    }
    let padded = patch_count
        .checked_next_multiple_of(TOKEN_TILE)
        .ok_or_else(|| Error::invalid("vision patch padding overflow"))?;
    let padded_elements = padded
        .checked_mul(width)
        .ok_or_else(|| Error::invalid("vision patch padding overflow"))?;
    validate_patch_input(&patches, patch_elements, padded, width)?;
    patch_count
        .checked_mul(hidden)
        .and_then(|_| padded.checked_mul(hidden))
        .ok_or_else(|| Error::invalid("vision patch output overflow"))?;
    let positions = StagedPositions::new(device, encoder, weights, positions, patch_count, padded)?;

    let input = stage_patches(device, patches, padded_elements)?;
    let embedded = device.runtime().zeros::<f32>(padded * hidden)?;
    let output = if positions.present() {
        Some(device.runtime().zeros::<f32>(padded * hidden)?)
    } else {
        None
    };
    let weight = weights.get("patch_embed.proj.weight")?;
    let bias = weights.get("patch_embed.proj.bias")?;

    let graph = capture(device, |launches| {
        launches.dense_bias(VisionDense {
            out: buffer(&embedded)?,
            input: buffer(&input)?,
            weight: buffer(weight)?,
            bias: buffer(bias)?,
            columns: width,
        })?;
        if let Some(output) = output.as_ref() {
            positions.record(launches, &embedded, output, hidden)?;
        }
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;

    let values = match output {
        Some(output) => device.runtime().read(&output)?,
        None => device.runtime().read(&embedded)?,
    };
    Ok(values[..patch_count * hidden].to_vec())
}

fn validate_patch_input(
    patches: &PatchInput<'_>,
    elements: usize,
    rows: usize,
    width: usize,
) -> Result<()> {
    match patches {
        PatchInput::Host(values) if values.len() != elements => {
            Err(Error::invalid("patch input shape"))
        }
        PatchInput::Device(values) => {
            if rows.checked_mul(width) != Some(values.len()) {
                return Err(Error::invalid("device patch input shape"));
            }
            Ok(())
        }
        PatchInput::Host(_) => Ok(()),
    }
}

fn stage_patches(
    device: &CudaDevice,
    patches: PatchInput<'_>,
    elements: usize,
) -> Result<Allocation<f32>> {
    match patches {
        PatchInput::Host(patches) => {
            let mut values = vec![0.0f32; elements];
            values[..patches.len()].copy_from_slice(patches);
            device.runtime().upload(values)
        }
        PatchInput::Device(patches) => Ok(patches.clone()),
    }
}

/// Linear projection with bias: `out = input @ weight^T + bias` for `[rows, in_features]`.
///
/// # Errors
/// Rejects shapes that disagree with the declared weights or failed CUDA capture/launch.
pub fn project(
    device: &CudaDevice,
    weights: &VisionWeights,
    weight_slot: &str,
    bias_slot: &str,
    input: &[f32],
    rows: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    project_with(
        device,
        weights.get(weight_slot)?,
        weights.get(bias_slot)?,
        input,
        rows,
        out_features,
    )
}

/// Linear projection with an explicit weight and bias pair.
///
/// # Errors
/// Rejects shapes that disagree with the weights or failed CUDA capture/launch.
pub fn project_with(
    device: &CudaDevice,
    weight: &Allocation<bf16>,
    bias: &Allocation<bf16>,
    input: &[f32],
    rows: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    if rows == 0 || !input.len().is_multiple_of(rows) {
        return Err(Error::invalid("vision projection shape"));
    }
    let in_features = input.len() / rows;
    if !in_features.is_multiple_of(COLUMN_TILE) || !out_features.is_multiple_of(COLUMN_TILE) {
        return Err(Error::unsupported(
            "vision projection widths must tile by the GEMM column size",
        ));
    }
    let padded = rows.div_ceil(TOKEN_TILE) * TOKEN_TILE;
    let mut values = vec![0.0f32; padded * in_features];
    values[..input.len()].copy_from_slice(input);
    let tensor = device.runtime().upload(values)?;
    let output = device.runtime().zeros::<f32>(padded * out_features)?;
    let view = weight;
    let graph = capture(device, |launches| {
        launches.dense_bias(VisionDense {
            out: buffer(&output)?,
            input: buffer(&tensor)?,
            weight: buffer(view)?,
            bias: buffer(bias)?,
            columns: in_features,
        })?;
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    projection_output(device.runtime().read(&output)?, rows, out_features)
}

pub(super) fn projection_output(
    values: Vec<f32>,
    rows: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    let len = rows
        .checked_mul(out_features)
        .ok_or_else(|| Error::invalid("vision projection output overflow"))?;
    if values.len() < len {
        return Err(Error::invariant("vision projection output truncated"));
    }
    if values.len() == len {
        Ok(values)
    } else {
        // Do not retain the extra padded rows in long-lived host results.
        Ok(values[..len].to_vec())
    }
}

/// Padded intermediate width of the block MLPs.
/// # Errors
/// Rejects overflowing geometry.
pub fn mlp_width(encoder: &ModalityEncoder) -> Result<usize> {
    encoder
        .intermediate_size
        .checked_next_multiple_of(COLUMN_TILE)
        .ok_or_else(|| Error::invalid("padded intermediate overflow"))
}

/// Element-wise `GELU` over a `[rows, columns]` activation; `erf` selects the merger's exact form.
///
/// # Errors
/// Rejects shapes that disagree with the geometry or failed CUDA capture/launch.
pub fn gelu(
    device: &CudaDevice,
    erf: bool,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    elementwise(device, erf, input, rows, columns)
}

/// Element-wise sum of two equally shaped activations.
///
/// # Errors
/// Rejects mismatched shapes or failed CUDA capture/launch.
pub fn add(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Err(Error::invalid("vision residual shape"));
    }
    elementwise_sum(device, left, right, rows, columns)
}

/// Shared plumbing of the element-wise kernels: pad rows, upload, capture, trim.
fn elementwise(
    device: &CudaDevice,
    erf: bool,
    input: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    if rows == 0 || input.len() != rows * columns || !columns.is_multiple_of(COLUMN_TILE) {
        return Err(Error::invalid("vision elementwise shape"));
    }
    let padded = rows.div_ceil(TOKEN_TILE) * TOKEN_TILE;
    let mut values = vec![0.0f32; padded * columns];
    values[..input.len()].copy_from_slice(input);
    let tensor = device.runtime().upload(values)?;
    let output = device.runtime().zeros::<f32>(padded * columns)?;
    let graph = capture(device, |launches| {
        if erf {
            launches.gelu(Gelu::Erf, columns, buffer(&output)?, buffer(&tensor)?)?;
        } else {
            launches.gelu(Gelu::Tanh, columns, buffer(&output)?, buffer(&tensor)?)?;
        }
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.runtime().read(&output)?;
    Ok(values[..rows * columns].to_vec())
}

/// Element-wise sum plumbing, mirroring [`elementwise`].
fn elementwise_sum(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    if rows == 0 || left.len() != rows * columns || !columns.is_multiple_of(COLUMN_TILE) {
        return Err(Error::invalid("vision elementwise shape"));
    }
    let padded = rows.div_ceil(TOKEN_TILE) * TOKEN_TILE;
    let mut values = vec![0.0f32; padded * columns];
    values[..left.len()].copy_from_slice(left);
    let mut other = vec![0.0f32; padded * columns];
    other[..right.len()].copy_from_slice(right);
    let first = device.runtime().upload(values)?;
    let second = device.runtime().upload(other)?;
    let output = device.runtime().zeros::<f32>(padded * columns)?;
    let graph = capture(device, |launches| {
        launches.add(columns, buffer(&output)?, buffer(&first)?, buffer(&second)?)?;
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.runtime().read(&output)?;
    Ok(values[..rows * columns].to_vec())
}

/// Row-wise `LayerNorm` with bias over `hidden` columns.
///
/// # Errors
/// Rejects shapes that disagree with the declared weights or failed CUDA capture/launch.
pub fn layernorm(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    weight_slot: &str,
    bias_slot: &str,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>> {
    let hidden = encoder.hidden_size;
    let tile = row_tile(encoder)?;
    if rows == 0 || input.len() != rows * hidden {
        return Err(Error::invalid("vision layernorm shape"));
    }
    let mut values = vec![0.0f32; rows * tile];
    for row in 0..rows {
        values[row * tile..row * tile + hidden]
            .copy_from_slice(&input[row * hidden..(row + 1) * hidden]);
    }
    let tensor = device.runtime().upload(values)?;
    let output = device.runtime().zeros::<f32>(rows * tile)?;
    let weight = weights.get(weight_slot)?;
    let bias = weights.get(bias_slot)?;
    let graph = capture(device, |launches| {
        launches.layernorm(VisionNorm {
            out: buffer(&output)?,
            input: buffer(&tensor)?,
            weight: buffer(weight)?,
            bias: buffer(bias)?,
            dim: hidden,
            tile,
            epsilon: encoder.norm_epsilon,
        })?;
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let padded = device.runtime().read(&output)?;
    let mut result = Vec::with_capacity(rows * hidden);
    for row in 0..rows {
        result.extend_from_slice(&padded[row * tile..row * tile + hidden]);
    }
    Ok(result)
}

/// Attention scaling of the real head width: `head_dim ** -0.5`.
/// # Errors
/// Rejects a zero head width.
#[expect(
    clippy::cast_precision_loss,
    reason = "Head widths are tiny exact integers"
)]
pub(super) fn attention_scale(encoder: &ModalityEncoder) -> Result<f32> {
    let head = head_dim(encoder)?;
    if head == 0 {
        return Err(Error::invalid("encoder head width must be nonzero"));
    }
    Ok(1.0 / (head as f32).sqrt())
}

/// Axial `RoPE` tables of one image: `[padded, 2, half]` cosine and sine tiles.
///
/// Lanes past the real half-head stay at `cos = 1, sin = 0`, which leaves the zero padding of the
/// projections untouched.
/// # Errors
/// Rejects grid or frequency geometry that disagrees with the encoder.
#[expect(
    clippy::cast_possible_truncation,
    reason = "RoPE angles are computed in f64 and stored in the f32 activation dtype"
)]
pub(super) fn rope_tables(
    encoder: &ModalityEncoder,
    grid: (usize, usize, usize),
    padded: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let ids = position_ids(grid, encoder)?;
    let frequencies = rope_frequencies(encoder)?;
    let half = half_tile(encoder)?;
    let real = head_dim(encoder)? / 2;
    if !real.is_multiple_of(2) || real / 2 > frequencies.len() {
        return Err(Error::invalid("RoPE axis geometry"));
    }
    let mut cosines = vec![1.0f32; padded * 2 * half];
    let mut sines = vec![0.0f32; padded * 2 * half];
    for (token, id) in ids.iter().enumerate() {
        for lane in 0..real {
            let (axis, index) = if lane < real / 2 {
                (id[0], lane)
            } else {
                (id[1], lane - real / 2)
            };
            let angle = f64::from(axis) * f64::from(frequencies[index]);
            for half_index in 0..2 {
                let slot = token * 2 * half + half_index * half + lane;
                cosines[slot] = angle.cos() as f32;
                sines[slot] = angle.sin() as f32;
            }
        }
    }
    Ok((cosines, sines))
}

/// Key-streaming form of the attention kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    /// One pass with a running maximum and accumulator rescaling (flash-attention form).
    Online,
    /// Global maximum first, then a rescale-free accumulation; the reference form.
    Exact,
}

impl AttentionMode {
    /// Mode selected by `INFER_ATTENTION_MODE`, defaulting to the streaming form.
    #[must_use]
    pub fn from_environment() -> Self {
        match var("INFER_ATTENTION_MODE").as_deref() {
            Ok("exact") => Self::Exact,
            _ => Self::Online,
        }
    }
}

/// Axial `RoPE` and packed non-causal attention of one block, projected back to hidden states.
///
/// # Errors
/// Rejects input that disagrees with the declared geometry or failed CUDA capture/launch.
pub fn attention(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    layer: usize,
    input: &[f32],
    grid: (usize, usize, usize),
    tokens: usize,
) -> Result<Vec<f32>> {
    attention_with(
        device,
        weights,
        encoder,
        layer,
        input,
        grid,
        tokens,
        AttentionMode::Online,
    )
}

/// Attention with an explicit key-streaming form.
///
/// # Errors
/// Rejects input that disagrees with the declared geometry or failed CUDA capture/launch.
#[expect(
    clippy::too_many_arguments,
    reason = "The caller owns the layer, tensor, grid, token count and streaming form"
)]
pub fn attention_with(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    layer: usize,
    input: &[f32],
    grid: (usize, usize, usize),
    tokens: usize,
    mode: AttentionMode,
) -> Result<Vec<f32>> {
    let frame_tokens = frame_token_count(grid, tokens)?;
    let hidden = encoder.hidden_size;
    let heads = encoder.heads;
    let half = half_tile(encoder)?;
    if half == COMPACT_HALF {
        return compact_attention_layer(device, weights, encoder, layer, input, grid, tokens, mode);
    }
    let width = half
        .checked_mul(2)
        .ok_or_else(|| Error::invalid("padded head overflow"))?;
    let projections = heads
        .checked_mul(width)
        .ok_or_else(|| Error::invalid("padded projection overflow"))?;
    if tokens == 0 || input.len() != tokens * hidden {
        return Err(Error::invalid("vision attention input shape"));
    }
    if !hidden.is_multiple_of(COLUMN_TILE) || !projections.is_multiple_of(COLUMN_TILE) {
        return Err(Error::unsupported(
            "vision attention geometry must tile by 64",
        ));
    }
    let tiles = attention_tiles(tokens);
    let padded = tokens.div_ceil(tiles[0]) * tiles[0];
    let mut values = vec![0.0f32; padded * hidden];
    values[..input.len()].copy_from_slice(input);
    let tensor = device.runtime().upload(values)?;
    let (cosines, sines) = rope_tables(encoder, grid, padded)?;
    let cosines = device.runtime().upload(cosines)?;
    let sines = device.runtime().upload(sines)?;
    let layer_weights = weights.attention(layer)?;
    let mut q = activation(device, padded, projections)?;
    let mut k = activation(device, padded, projections)?;
    let mut v = activation(device, padded, projections)?;
    let mut rotated_query = activation(device, padded, projections)?;
    let mut rotated_key = activation(device, padded, projections)?;
    let attended = activation(device, padded, projections)?;
    let output = activation(device, padded, hidden)?;
    let tokens = i32::try_from(tokens).map_err(device_error)?;
    let scale = attention_scale(encoder)?;
    let graph = capture(device, |launches| {
        for (weight, bias, target) in [
            (&layer_weights.q.0, &layer_weights.q.1, &mut q),
            (&layer_weights.k.0, &layer_weights.k.1, &mut k),
            (&layer_weights.v.0, &layer_weights.v.1, &mut v),
        ] {
            launches.dense_bias(VisionDense {
                out: buffer(target)?,
                input: buffer(&tensor)?,
                weight: buffer(weight)?,
                bias: buffer(bias)?,
                columns: hidden,
            })?;
        }
        for (input, output) in [(&q, &mut rotated_query), (&k, &mut rotated_key)] {
            launches.rope(VisionRope {
                out: buffer(output)?,
                input: buffer(input)?,
                cosines: buffer(&cosines)?,
                sines: buffer(&sines)?,
                rows: padded,
                half,
                real_half: None,
            })?;
        }
        launches.dense(
            dense_plan(width, mode, tiles, padded, tokens, scale, frame_tokens),
            DenseBuffers {
                out: buffer(&attended)?,
                query: buffer(&rotated_query)?,
                key: buffer(&rotated_key)?,
                value: buffer(&v)?,
            },
        )?;
        launches.dense_bias(VisionDense {
            out: buffer(&output)?,
            input: buffer(&attended)?,
            weight: buffer(&layer_weights.proj.0)?,
            bias: buffer(&layer_weights.proj.1)?,
            columns: projections,
        })?;
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.runtime().read(&output)?;
    Ok(values[..input.len()].to_vec())
}

struct CompactBuffers {
    q: Allocation<f32>,
    k: Allocation<f32>,
    v: Allocation<f32>,
    q_rot: Allocation<f32>,
    k_rot: Allocation<f32>,
    q_wide: Allocation<f32>,
    k_wide: Allocation<f32>,
    v_wide: Allocation<f32>,
    attended: Allocation<f32>,
    compact_out: Allocation<f32>,
    output: Option<Allocation<f32>>,
}

fn compact_buffers(
    device: &CudaDevice,
    padded: usize,
    heads: usize,
    hidden: usize,
) -> Result<CompactBuffers> {
    let projections = heads * COMPACT_WIDTH;
    let wide = heads * WIDE_WIDTH;
    Ok(CompactBuffers {
        q: activation(device, padded, projections)?,
        k: activation(device, padded, projections)?,
        v: activation(device, padded, projections)?,
        q_rot: activation(device, padded, projections)?,
        k_rot: activation(device, padded, projections)?,
        q_wide: activation(device, padded, wide)?,
        k_wide: activation(device, padded, wide)?,
        v_wide: activation(device, padded, wide)?,
        attended: activation(device, padded, wide)?,
        compact_out: activation(device, padded, projections)?,
        output: Some(activation(device, padded, hidden)?),
    })
}

/// Production compact head72 attention: projected Q/K/V stay 96-wide, pack to the qualified
/// 128-wide SDPA, then pack back for the compact output projection.
#[expect(
    clippy::too_many_arguments,
    reason = "The production layer owns the same explicit geometry as attention_with"
)]
fn compact_attention_layer(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    layer: usize,
    input: &[f32],
    grid: (usize, usize, usize),
    tokens: usize,
    mode: AttentionMode,
) -> Result<Vec<f32>> {
    let frame_tokens = frame_token_count(grid, tokens)?;
    let hidden = encoder.hidden_size;
    let heads = encoder.heads;
    let projections = heads
        .checked_mul(COMPACT_WIDTH)
        .ok_or_else(|| Error::invalid("compact projection overflow"))?;
    if tokens == 0 || input.len() != tokens * hidden {
        return Err(Error::invalid("vision attention input shape"));
    }
    if !hidden.is_multiple_of(COLUMN_TILE) || !projections.is_multiple_of(COLUMN_TILE) {
        return Err(Error::unsupported(
            "vision compact geometry must tile by 64",
        ));
    }
    let tiles = attention_tiles(tokens);
    let padded = tokens.div_ceil(tiles[0]) * tiles[0];
    let mut values = vec![0.0f32; padded * hidden];
    values[..input.len()].copy_from_slice(input);
    let tensor = device.runtime().upload(values)?;
    let (cosines, sines) = rope_tables(encoder, grid, padded)?;
    let cosines = device.runtime().upload(cosines)?;
    let sines = device.runtime().upload(sines)?;
    let layer_weights = weights.attention(layer)?;
    let mut buffers = compact_buffers(device, padded, heads, hidden)?;
    let tokens_i = i32::try_from(tokens).map_err(device_error)?;
    let scale = attention_scale(encoder)?;
    let values = compact_attention_launch(
        device,
        &mut buffers,
        &tensor,
        &cosines,
        &sines,
        layer_weights,
        tokens_i,
        frame_tokens,
        scale,
        tiles,
        hidden,
        projections,
        mode,
    )?;
    Ok(values[..input.len()].to_vec())
}

#[expect(
    clippy::too_many_arguments,
    reason = "The staged launch helper receives its explicit device buffers and geometry"
)]
fn compact_attention_launch(
    device: &CudaDevice,
    buffers: &mut CompactBuffers,
    tensor: &Allocation<f32>,
    cosines: &Allocation<f32>,
    sines: &Allocation<f32>,
    layer_weights: &AttentionLayer,
    tokens: i32,
    frame_tokens: i32,
    scale: f32,
    tiles: [usize; ATTENTION_CONFIG_FIELDS],
    hidden: usize,
    projections: usize,
    mode: AttentionMode,
) -> Result<Vec<f32>> {
    let padded = tensor
        .len()
        .checked_div(hidden)
        .ok_or_else(|| Error::invalid("compact attention hidden width"))?;
    let output = buffers
        .output
        .take()
        .ok_or_else(|| Error::invalid("compact attention output buffer"))?;
    let graph = capture(device, |launches| {
        for (weight, bias, target) in [
            (&layer_weights.q.0, &layer_weights.q.1, &mut buffers.q),
            (&layer_weights.k.0, &layer_weights.k.1, &mut buffers.k),
            (&layer_weights.v.0, &layer_weights.v.1, &mut buffers.v),
        ] {
            launches.dense_bias(VisionDense {
                out: buffer(target)?,
                input: buffer(tensor)?,
                weight: buffer(weight)?,
                bias: buffer(bias)?,
                columns: hidden,
            })?;
        }
        for (input, output) in [
            (&buffers.q, &mut buffers.q_rot),
            (&buffers.k, &mut buffers.k_rot),
        ] {
            launches.rope(VisionRope {
                out: buffer(output)?,
                input: buffer(input)?,
                cosines: buffer(cosines)?,
                sines: buffer(sines)?,
                rows: padded,
                half: COMPACT_HALF,
                real_half: Some(COMPACT_REAL_HALF),
            })?;
        }
        for (input, output) in [
            (&buffers.q_rot, &mut buffers.q_wide),
            (&buffers.k_rot, &mut buffers.k_wide),
            (&buffers.v, &mut buffers.v_wide),
        ] {
            launches.pack(Pack::Wide, padded, buffer(output)?, buffer(input)?)?;
        }
        launches.dense(
            dense_plan(WIDE_WIDTH, mode, tiles, padded, tokens, scale, frame_tokens),
            DenseBuffers {
                out: buffer(&buffers.attended)?,
                query: buffer(&buffers.q_wide)?,
                key: buffer(&buffers.k_wide)?,
                value: buffer(&buffers.v_wide)?,
            },
        )?;
        launches.pack(
            Pack::Compact,
            padded,
            buffer(&buffers.compact_out)?,
            buffer(&buffers.attended)?,
        )?;
        launches.dense_bias(VisionDense {
            out: buffer(&output)?,
            input: buffer(&buffers.compact_out)?,
            weight: buffer(&layer_weights.proj.0)?,
            bias: buffer(&layer_weights.proj.1)?,
            columns: projections,
        })?;
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.runtime().read(&output)?;
    Ok(values)
}

pub(super) fn frame_token_count(grid: (usize, usize, usize), tokens: usize) -> Result<i32> {
    let count = grid
        .1
        .checked_mul(grid.2)
        .filter(|n| *n > 0 && n.checked_mul(grid.0) == Some(tokens))
        .ok_or_else(|| Error::invalid("vision attention grid disagrees with tokens"))?;
    i32::try_from(count).map_err(device_error)
}

/// Allocate a zeroed activation matrix on the device stream.
pub(super) fn activation(
    device: &CudaDevice,
    rows: usize,
    columns: usize,
) -> Result<Allocation<f32>> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::invalid("vision activation overflow"))?;
    device.runtime().zeros(elements)
}

/// One vision block: attention and MLP sublayers with their residual adds.
///
/// # Errors
/// Rejects input that disagrees with the declared geometry or failed CUDA capture/launch.
pub fn block(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    layer: usize,
    input: &[f32],
    grid: (usize, usize, usize),
    tokens: usize,
) -> Result<Vec<f32>> {
    let hidden = encoder.hidden_size;
    if tokens == 0 || input.len() != tokens * hidden {
        return Err(Error::invalid("vision block input shape"));
    }
    let prefix = format!("blocks.{layer}.");
    let normed = layernorm(
        device,
        weights,
        encoder,
        &format!("{prefix}norm1.weight"),
        &format!("{prefix}norm1.bias"),
        input,
        tokens,
    )?;
    let attended = attention(device, weights, encoder, layer, &normed, grid, tokens)?;
    let residual = add(device, input, &attended, tokens, hidden)?;
    let normed = layernorm(
        device,
        weights,
        encoder,
        &format!("{prefix}norm2.weight"),
        &format!("{prefix}norm2.bias"),
        &residual,
        tokens,
    )?;
    let width = mlp_width(encoder)?;
    let layer_weights = weights.mlp(layer)?;
    block_mlp::run(
        device,
        layer_weights,
        &normed,
        &residual,
        tokens,
        hidden,
        width,
    )
}

/// The whole vision tower: every block in declaration order.
///
/// # Errors
/// Rejects input that disagrees with the declared geometry or failed CUDA computation.
pub fn tower(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    input: &[f32],
    grid: (usize, usize, usize),
    tokens: usize,
) -> Result<Vec<f32>> {
    if device_tower::supports(encoder)? {
        return device_tower::run(device, weights, encoder, input, grid, tokens);
    }
    let mut state = input.to_vec();
    for layer in 0..encoder.depth {
        state = block(device, weights, encoder, layer, &state, grid, tokens)?;
    }
    Ok(state)
}

/// Spatial merge and merger MLP: `LayerNorm`, four-patch concatenation, `fc1`, erf `GELU`, `fc2`.
///
/// The tower already emits patches in spatial-merge-block order, so the concatenation is a plain
/// regrouping of consecutive rows.
///
/// # Errors
/// Rejects token counts that do not divide by the merge unit or failed CUDA computation.
pub fn merger(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    hidden_states: &[f32],
    tokens: usize,
) -> Result<Vec<f32>> {
    let hidden = encoder.hidden_size;
    let merged = merged_width(encoder)?;
    let unit = merge_unit(encoder)?;
    if tokens == 0 || !tokens.is_multiple_of(unit) || hidden_states.len() != tokens * hidden {
        return Err(Error::invalid("vision merger input shape"));
    }
    let rows = tokens / unit;
    let normed = layernorm(
        device,
        weights,
        encoder,
        "merger.norm.weight",
        "merger.norm.bias",
        hidden_states,
        tokens,
    )?;
    let layer_weights = weights.merger()?;
    let fc1 = project_with(
        device,
        &layer_weights.fc1.0,
        &layer_weights.fc1.1,
        &normed,
        rows,
        merged,
    )?;
    let activated = gelu(device, true, &fc1, rows, merged)?;
    project_with(
        device,
        &layer_weights.fc2.0,
        &layer_weights.fc2.1,
        &activated,
        rows,
        encoder.out_hidden_size,
    )
}

/// Run the attention kernel alone on caller-supplied tensors.
///
/// `query`, `key` and `value` use the padded half-head layout the projections produce
/// (`tokens × heads × 2 × half`), and `cos`/`sin` are `tokens × 2 × half`. Exposed so the kernel
/// regression test can compare the device result against an independent host computation.
///
/// # Errors
/// Rejects shapes that do not tile by the kernel's block sizes or failed CUDA execution.
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "The regression test supplies every kernel operand explicitly"
)]
pub fn attention_kernel(
    device: &CudaDevice,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cos: &[f32],
    sin: &[f32],
    tokens: usize,
    heads: usize,
    half: usize,
    mode: AttentionMode,
    frame_tokens: usize,
    real_width: usize,
    tiles: [usize; ATTENTION_CONFIG_FIELDS],
    profile: bool,
) -> Result<Vec<f32>> {
    if frame_tokens == 0 || frame_tokens > tokens {
        return Err(Error::invalid("attention probe frame size"));
    }
    let frame_tokens = i32::try_from(frame_tokens).map_err(device_error)?;
    let width = half
        .checked_mul(2)
        .ok_or_else(|| Error::invalid("padded head overflow"))?;
    let projections = heads
        .checked_mul(width)
        .ok_or_else(|| Error::invalid("padded projection overflow"))?;
    let expected = tokens
        .checked_mul(projections)
        .ok_or_else(|| Error::invalid("attention probe overflow"))?;
    let table = tokens
        .checked_mul(width)
        .ok_or_else(|| Error::invalid("attention probe table overflow"))?;
    if tokens == 0 || query.len() != expected || key.len() != expected || value.len() != expected {
        return Err(Error::invalid("attention probe tensor shape"));
    }
    if cos.len() != table || sin.len() != table {
        return Err(Error::invalid("attention probe rope table shape"));
    }
    let align = TOKEN_TILE.max(tiles[0]).max(tiles[1]);
    let padded = tokens.div_ceil(align) * align;
    let pad = |values: &[f32]| -> Result<Allocation<f32>> {
        let mut padded_values = vec![0.0f32; padded * projections];
        padded_values[..values.len()].copy_from_slice(values);
        device.runtime().upload(padded_values)
    };
    let table_pad = |values: &[f32]| -> Result<Allocation<f32>> {
        let mut padded_values = vec![0.0f32; padded * width];
        for token in 0..tokens {
            padded_values[token * width..(token + 1) * width]
                .copy_from_slice(&values[token * width..(token + 1) * width]);
        }
        device.runtime().upload(padded_values)
    };
    let query = pad(query)?;
    let key = pad(key)?;
    let value = pad(value)?;
    let cosines = table_pad(cos)?;
    let sines = table_pad(sin)?;
    let mut rotated_query = device.runtime().zeros::<f32>(padded * projections)?;
    let mut rotated_key = device.runtime().zeros::<f32>(padded * projections)?;
    let output = device.runtime().zeros::<f32>(padded * projections)?;
    let scale = attention_scale_from_width(real_width);
    let tokens = i32::try_from(tokens).map_err(device_error)?;
    let graph = capture(device, |launches| {
        let repeats = if profile { REPEATS } else { 1 };
        for _ in 0..repeats {
            for (input, output) in [(&query, &mut rotated_query), (&key, &mut rotated_key)] {
                launches.rope(VisionRope {
                    out: buffer(output)?,
                    input: buffer(input)?,
                    cosines: buffer(&cosines)?,
                    sines: buffer(&sines)?,
                    rows: padded,
                    half,
                    real_half: None,
                })?;
            }
            launches.dense(
                dense_plan(width, mode, tiles, padded, tokens, scale, frame_tokens),
                DenseBuffers {
                    out: buffer(&output)?,
                    query: buffer(&rotated_query)?,
                    key: buffer(&rotated_key)?,
                    value: buffer(&value)?,
                },
            )?;
        }
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    if profile {
        measure(device, &graph, tokens, heads, real_width, tiles)?;
    }
    let values = device.runtime().read(&output)?;
    Ok(values[..expected].to_vec())
}

/// Attention scaling of one padded head width: the real head is `head_dim`, not the padded one.
#[expect(
    clippy::cast_precision_loss,
    reason = "Head widths are tiny exact integers"
)]
fn attention_scale_from_width(width: usize) -> f32 {
    1.0 / (width as f32).sqrt()
}

/// Mask form of a vision block: every token attends every token of its frame.
const NONCAUSAL_MASK: i32 = 3;

/// Full-head attention of one padded head width.
pub(super) const fn dense_plan(
    width: usize,
    mode: AttentionMode,
    tiles: [usize; ATTENTION_CONFIG_FIELDS],
    rows: usize,
    tokens: i32,
    scale: f32,
    frame_tokens: i32,
) -> DensePlan {
    DensePlan {
        head_width: width,
        online: matches!(mode, AttentionMode::Online),
        tiles,
        group: 1,
        value_width: width,
        mask: NONCAUSAL_MASK,
        query_rows: rows,
        key_rows: rows,
        tokens,
        scale,
        frame_tokens,
        queries: tokens,
        query_start: 0,
        left: 0,
        right: 0,
    }
}

/// Measured crossover for compensated full-head attention. Pipelining did not improve timings.
pub const fn attention_tiles(tokens: usize) -> [usize; ATTENTION_CONFIG_FIELDS] {
    let tile = tile_rows(tokens);
    [tile, tile, 0]
}

/// Test-only launch of the indexed compact `RoPE` kernel.
///
/// # Errors
/// Rejects invalid half-width/row geometry or failed CUDA execution.
#[cfg(test)]
pub fn rope_compact_kernel(
    device: &CudaDevice,
    query: &[f32],
    cos: &[f32],
    sin: &[f32],
    tokens: usize,
    half: usize,
    real: usize,
) -> Result<Vec<f32>> {
    if half < 16
        || !half.is_multiple_of(16)
        || real > half
        || tokens == 0
        || cos.len() != tokens * half * 2
        || sin.len() != tokens * half * 2
    {
        return Err(Error::invalid("indexed RoPE fixture geometry"));
    }
    let width = half * 2;
    let heads = query
        .len()
        .checked_div(
            tokens
                .checked_mul(width)
                .ok_or_else(|| Error::invalid("indexed RoPE fixture geometry"))?,
        )
        .filter(|heads| *heads > 0 && query.len() == tokens * heads * width)
        .ok_or_else(|| Error::invalid("indexed RoPE fixture geometry"))?;
    let padded = tokens.div_ceil(TOKEN_TILE) * TOKEN_TILE;
    let projections = heads * width;
    let mut values = vec![0.0f32; padded * projections];
    values[..query.len()].copy_from_slice(query);
    let input = device.runtime().upload(values)?;
    let mut cosines = vec![0.0f32; padded * width];
    cosines[..cos.len()].copy_from_slice(cos);
    let mut sines = vec![0.0f32; padded * width];
    sines[..sin.len()].copy_from_slice(sin);
    let cosines = device.runtime().upload(cosines)?;
    let sines = device.runtime().upload(sines)?;
    let output = device.runtime().zeros::<f32>(padded * projections)?;
    let graph = capture(device, |launches| {
        launches.rope(VisionRope {
            out: buffer(&output)?,
            input: buffer(&input)?,
            cosines: buffer(&cosines)?,
            sines: buffer(&sines)?,
            rows: padded,
            half,
            real_half: Some(real),
        })?;
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.runtime().read(&output)?;
    Ok(values[..query.len()].to_vec())
}

/// Compact attention path: physical 96-wide heads -> power-of-two 128 pack -> existing
/// qualified SDPA with the true 72-wide scale -> pack back to 96.
///
/// The path is production-capable but not selected by [`attention_with`] until the full compact
/// encoder is enabled. The caller owns the intermediate wide buffers; their footprint is part of
/// the media reservation.
///
/// # Errors
/// Rejects invalid fixture geometry or failed CUDA execution.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "The compact attention block is staged until the full compact encoder selects it"
    )
)]
#[expect(
    clippy::too_many_arguments,
    reason = "The staged path exposes Q/K/V, RoPE tables, geometry and launch tiles explicitly"
)]
pub fn compact_attention(
    device: &CudaDevice,
    qkv: (&[f32], &[f32], &[f32]),
    cos: &[f32],
    sin: &[f32],
    tokens: usize,
    heads: usize,
    frame_tokens: usize,
    tiles: [usize; ATTENTION_CONFIG_FIELDS],
) -> Result<Vec<f32>> {
    const REAL_HEAD: usize = 72;
    let (query, key, value) = qkv;
    if tokens == 0
        || heads == 0
        || frame_tokens == 0
        || frame_tokens > tokens
        || query.len() != tokens * heads * COMPACT_WIDTH
        || key.len() != query.len()
        || value.len() != query.len()
        || cos.len() != tokens * COMPACT_WIDTH
        || sin.len() != tokens * COMPACT_WIDTH
    {
        return Err(Error::invalid("compact attention fixture geometry"));
    }
    let align = TOKEN_TILE.max(tiles[0]).max(tiles[1]);
    let padded = tokens.div_ceil(align) * align;
    let compact_projection = heads * COMPACT_WIDTH;
    let wide_projection = heads * WIDE_WIDTH;
    let pad = |values: &[f32], width: usize| -> Result<Allocation<f32>> {
        let projection = heads * width;
        let mut padded_values = vec![0.0f32; padded * projection];
        padded_values[..values.len()].copy_from_slice(values);
        device.runtime().upload(padded_values)
    };
    let table = |values: &[f32]| -> Result<Allocation<f32>> {
        let mut padded_values = vec![0.0f32; padded * COMPACT_WIDTH];
        padded_values[..values.len()].copy_from_slice(values);
        device.runtime().upload(padded_values)
    };
    let q = pad(query, COMPACT_WIDTH)?;
    let k = pad(key, COMPACT_WIDTH)?;
    let v = pad(value, COMPACT_WIDTH)?;
    let cos = table(cos)?;
    let sin = table(sin)?;
    let mut q_rot = activation(device, padded, compact_projection)?;
    let mut k_rot = activation(device, padded, compact_projection)?;
    let mut q_wide = activation(device, padded, wide_projection)?;
    let mut k_wide = activation(device, padded, wide_projection)?;
    let mut v_wide = activation(device, padded, wide_projection)?;
    let attended = activation(device, padded, wide_projection)?;
    let compact_out = activation(device, padded, compact_projection)?;
    let tokens_i = i32::try_from(tokens).map_err(device_error)?;
    let frame_i = i32::try_from(frame_tokens).map_err(device_error)?;
    let scale = attention_scale_from_width(REAL_HEAD);
    let graph = capture(device, |launches| {
        for (input, output) in [(&q, &mut q_rot), (&k, &mut k_rot)] {
            launches.rope(VisionRope {
                out: buffer(output)?,
                input: buffer(input)?,
                cosines: buffer(&cos)?,
                sines: buffer(&sin)?,
                rows: padded,
                half: COMPACT_HALF,
                real_half: Some(COMPACT_REAL_HALF),
            })?;
        }
        for (input, output) in [
            (&q_rot, &mut q_wide),
            (&k_rot, &mut k_wide),
            (&v, &mut v_wide),
        ] {
            launches.pack(Pack::Wide, padded, buffer(output)?, buffer(input)?)?;
        }
        launches.dense(
            dense_plan(
                WIDE_WIDTH,
                AttentionMode::Online,
                tiles,
                padded,
                tokens_i,
                scale,
                frame_i,
            ),
            DenseBuffers {
                out: buffer(&attended)?,
                query: buffer(&q_wide)?,
                key: buffer(&k_wide)?,
                value: buffer(&v_wide)?,
            },
        )?;
        launches.pack(
            Pack::Compact,
            padded,
            buffer(&compact_out)?,
            buffer(&attended)?,
        )?;
        Ok(())
    })?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.runtime().read(&compact_out)?;
    Ok(values[..query.len()].to_vec())
}
