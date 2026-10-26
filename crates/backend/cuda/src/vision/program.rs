//! Captured vision-encoder programs.
use super::{VisionWeights, kernels, weights::half_tile};
#[cfg(test)]
use crate::attention::benchmark::REPEATS;
#[cfg(test)]
use crate::attention::benchmark::measure;
use crate::attention::kernels::sdpa::attention as sdpa_attention;
use crate::attention::tile_rows;
use crate::device::{CudaDevice, device_error};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_models::vision::head_dim;
use infer_models::vision::merge_unit;
use infer_models::vision::merged_width;
use infer_models::vision::position_ids;
use infer_models::vision::rope_frequencies;
use infer_models::vision::row_tile;
use infer_spi::ModalityEncoder;
use std::env::var;
use std::sync::Arc;

/// Token tile of every vision projection, matching the prompt GEMM.
const TOKEN_TILE: usize = 32;
/// Column tile of every vision projection, matching the prompt GEMM.
const COLUMN_TILE: usize = 64;

/// Column tile of every vision projection; test-only audit hook.
#[cfg(test)]
#[must_use]
pub(super) const fn column_tile() -> usize {
    COLUMN_TILE
}
const ATTENTION_CONFIG_FIELDS: usize = 3;

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
    let width = patch_width(encoder)?;
    let hidden = encoder.hidden_size;
    if patch_count == 0 || patches.len() != patch_count * width {
        return Err(Error::invalid("patch input shape"));
    }
    if !hidden.is_multiple_of(COLUMN_TILE) || !width.is_multiple_of(COLUMN_TILE) {
        return Err(Error::unsupported(
            "vision patch/hidden widths must tile by the GEMM column size",
        ));
    }
    let with_positions = !positions.is_empty();
    if with_positions && positions.len() != patch_count * hidden {
        return Err(Error::invalid("position input shape"));
    }
    let padded = patch_count.div_ceil(TOKEN_TILE) * TOKEN_TILE;

    let mut input_values = vec![0.0f32; padded * width];
    input_values[..patches.len()].copy_from_slice(patches);
    let input = device.upload(input_values, &[padded, width])?;
    let mut embedded = api::zeros::<f32>(&[padded, hidden])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let position_tensor = if with_positions {
        let mut values = vec![0.0f32; padded * hidden];
        values[..positions.len()].copy_from_slice(positions);
        Some(device.upload(values, &[padded, hidden])?)
    } else {
        None
    };
    let mut output = if with_positions {
        Some(
            api::zeros::<f32>(&[padded, hidden])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        )
    } else {
        None
    };
    let weight = weights
        .get("patch_embed.proj.weight")?
        .view(&[hidden, width])
        .map_err(device_error)?;
    let bias = weights.get("patch_embed.proj.bias")?;

    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::encoder::dense_bias(
                (&mut embedded).partition([TOKEN_TILE, COLUMN_TILE]),
                &input,
                &weight,
                bias,
            )
            .generics(vec![width.to_string()]),
        )?;
        if let (Some(positions), Some(output)) = (&position_tensor, output.as_mut()) {
            scope.record(
                kernels::encoder::add(
                    output.partition([TOKEN_TILE, COLUMN_TILE]),
                    &embedded,
                    positions,
                )
                .generics(vec![hidden.to_string()]),
            )?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;

    let values = match output {
        Some(output) => device.read(&Arc::new(output))?,
        None => device.read(&Arc::new(embedded))?,
    };
    Ok(values[..patch_count * hidden].to_vec())
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
    weight: &Arc<Tensor<bf16>>,
    bias: &Arc<Tensor<bf16>>,
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
    let tensor = device.upload(values, &[padded, in_features])?;
    let mut output = api::zeros::<f32>(&[padded, out_features])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let view = weight
        .view(&[out_features, in_features])
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::encoder::dense_bias(
                (&mut output).partition([TOKEN_TILE, COLUMN_TILE]),
                &tensor,
                &view,
                bias,
            )
            .generics(vec![in_features.to_string()]),
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.read(&Arc::new(output))?;
    Ok(values[..rows * out_features].to_vec())
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
    let tensor = device.upload(values, &[padded, columns])?;
    let mut output = api::zeros::<f32>(&[padded, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        let generics = vec![columns.to_string()];
        if erf {
            scope.record(
                kernels::encoder::gelu_erf(
                    (&mut output).partition([TOKEN_TILE, COLUMN_TILE]),
                    &tensor,
                )
                .generics(generics),
            )?;
        } else {
            scope.record(
                kernels::encoder::gelu_tanh(
                    (&mut output).partition([TOKEN_TILE, COLUMN_TILE]),
                    &tensor,
                )
                .generics(generics),
            )?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.read(&Arc::new(output))?;
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
    let first = device.upload(values, &[padded, columns])?;
    let second = device.upload(other, &[padded, columns])?;
    let mut output = api::zeros::<f32>(&[padded, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::encoder::add(
                (&mut output).partition([TOKEN_TILE, COLUMN_TILE]),
                &first,
                &second,
            )
            .generics(vec![columns.to_string()]),
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.read(&Arc::new(output))?;
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
    let tensor = device.upload(values, &[rows, tile])?;
    let mut output = api::zeros::<f32>(&[rows * tile])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight = weights.get(weight_slot)?;
    let bias = weights.get(bias_slot)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::encoder::layernorm(
                (&mut output).partition([tile]),
                &tensor,
                weight,
                bias,
                encoder.norm_epsilon,
            )
            .generics(vec![hidden.to_string(), tile.to_string()]),
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let padded = device.read(&Arc::new(output))?;
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
fn attention_scale(encoder: &ModalityEncoder) -> Result<f32> {
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
fn rope_tables(
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
    const fn generic(self) -> &'static str {
        match self {
            Self::Online => "1",
            Self::Exact => "0",
        }
    }

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
    let tensor = device.upload(values, &[padded, hidden])?;
    let (cosines, sines) = rope_tables(encoder, grid, padded)?;
    let cosines = device.upload(cosines, &[padded, width])?;
    let sines = device.upload(sines, &[padded, width])?;
    let layer_weights = weights.attention(layer)?;
    let mut q = activation(device, padded, projections)?;
    let mut k = activation(device, padded, projections)?;
    let mut v = activation(device, padded, projections)?;
    let mut rotated_query = activation(device, padded, projections)?;
    let mut rotated_key = activation(device, padded, projections)?;
    let mut attended = activation(device, padded, projections)?;
    let mut output = activation(device, padded, hidden)?;
    let tokens = i32::try_from(tokens).map_err(device_error)?;
    let scale = attention_scale(encoder)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        for (weight, bias, target) in [
            (&layer_weights.q.0, &layer_weights.q.1, &mut q),
            (&layer_weights.k.0, &layer_weights.k.1, &mut k),
            (&layer_weights.v.0, &layer_weights.v.1, &mut v),
        ] {
            scope.record(
                kernels::encoder::dense_bias(
                    target.partition([TOKEN_TILE, COLUMN_TILE]),
                    &tensor,
                    weight,
                    bias,
                )
                .generics(vec![hidden.to_string()]),
            )?;
        }
        for (input, output) in [(&q, &mut rotated_query), (&k, &mut rotated_key)] {
            scope.record(
                kernels::encoder::rope(
                    output.partition([TOKEN_TILE, half]),
                    input,
                    &cosines,
                    &sines,
                )
                .generics(vec![half.to_string()]),
            )?;
        }
        scope.record(
            sdpa_attention(
                (&mut attended).partition([tiles[0], width]),
                &rotated_query,
                &rotated_key,
                &v,
                tokens,
                scale,
                frame_tokens,
                tokens,
                0,
                0,
                0,
            )
            .generics(attention_generics(width, mode, tiles)),
        )?;
        scope.record(
            kernels::encoder::dense_bias(
                (&mut output).partition([TOKEN_TILE, COLUMN_TILE]),
                &attended,
                &layer_weights.proj.0,
                &layer_weights.proj.1,
            )
            .generics(vec![projections.to_string()]),
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.read(&Arc::new(output))?;
    Ok(values[..input.len()].to_vec())
}

fn frame_token_count(grid: (usize, usize, usize), tokens: usize) -> Result<i32> {
    let count = grid
        .1
        .checked_mul(grid.2)
        .filter(|n| *n > 0 && n.checked_mul(grid.0) == Some(tokens))
        .ok_or_else(|| Error::invalid("vision attention grid disagrees with tokens"))?;
    i32::try_from(count).map_err(device_error)
}

/// Allocate a zeroed activation matrix on the device stream.
fn activation(device: &CudaDevice, rows: usize, columns: usize) -> Result<Tensor<f32>> {
    api::zeros::<f32>(&[rows, columns])
        .sync_on(&device.stream)
        .map_err(device_error)
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
    let fc1 = project_with(
        device,
        &layer_weights.fc1.0,
        &layer_weights.fc1.1,
        &normed,
        tokens,
        width,
    )?;
    let activated = gelu(device, false, &fc1, tokens, width)?;
    let fc2 = project_with(
        device,
        &layer_weights.fc2.0,
        &layer_weights.fc2.1,
        &activated,
        tokens,
        hidden,
    )?;
    add(device, &residual, &fc2, tokens, hidden)
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
    let pad = |values: &[f32]| -> Result<Arc<Tensor<f32>>> {
        let mut padded_values = vec![0.0f32; padded * projections];
        padded_values[..values.len()].copy_from_slice(values);
        device.upload(padded_values, &[padded, projections])
    };
    let table_pad = |values: &[f32]| -> Result<Arc<Tensor<f32>>> {
        let mut padded_values = vec![0.0f32; padded * width];
        for token in 0..tokens {
            padded_values[token * width..(token + 1) * width]
                .copy_from_slice(&values[token * width..(token + 1) * width]);
        }
        device.upload(padded_values, &[padded, width])
    };
    let query = pad(query)?;
    let key = pad(key)?;
    let value = pad(value)?;
    let cosines = table_pad(cos)?;
    let sines = table_pad(sin)?;
    let mut rotated_query = api::zeros::<f32>(&[padded, projections])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut rotated_key = api::zeros::<f32>(&[padded, projections])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::zeros::<f32>(&[padded, projections])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let scale = attention_scale_from_width(real_width);
    let tokens = i32::try_from(tokens).map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        let repeats = if profile { REPEATS } else { 1 };
        for _ in 0..repeats {
            for (input, output) in [(&query, &mut rotated_query), (&key, &mut rotated_key)] {
                scope.record(
                    kernels::encoder::rope(
                        output.partition([TOKEN_TILE, half]),
                        input,
                        &cosines,
                        &sines,
                    )
                    .generics(vec![half.to_string()]),
                )?;
            }
            scope.record(
                sdpa_attention(
                    (&mut output).partition([tiles[0], width]),
                    &rotated_query,
                    &rotated_key,
                    &value,
                    tokens,
                    scale,
                    frame_tokens,
                    tokens,
                    0,
                    0,
                    0,
                )
                .generics(attention_generics(width, mode, tiles)),
            )?;
        }
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    if profile {
        measure(device, &graph, tokens, heads, real_width, tiles)?;
    }
    let values = device.read(&Arc::new(output))?;
    Ok(values[..expected].to_vec())
}

/// Attention scaling of one padded head width: the real head is `head_dim`, not the padded one.
#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Head widths are tiny exact integers"
)]
fn attention_scale_from_width(width: usize) -> f32 {
    1.0 / (width as f32).sqrt()
}

fn attention_generics(
    width: usize,
    mode: AttentionMode,
    tiles: [usize; ATTENTION_CONFIG_FIELDS],
) -> Vec<String> {
    let mut values = vec![width.to_string(), mode.generic().to_owned()];
    values.extend(tiles.map(|n| n.to_string()));
    values.extend(["1".into(), width.to_string(), "3".into()]);
    values
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
    let input = device.upload(values, &[padded, projections])?;
    let mut cosines = vec![0.0f32; padded * width];
    cosines[..cos.len()].copy_from_slice(cos);
    let mut sines = vec![0.0f32; padded * width];
    sines[..sin.len()].copy_from_slice(sin);
    let cosines = device.upload(cosines, &[padded, width])?;
    let sines = device.upload(sines, &[padded, width])?;
    let mut output = api::zeros::<f32>(&[padded, projections])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::encoder::rope_indexed(
                (&mut output).partition([TOKEN_TILE, 16]),
                &input,
                &cosines,
                &sines,
            )
            .generics(vec![half.to_string(), real.to_string()]),
        )?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let values = device.read(&Arc::new(output))?;
    Ok(values[..query.len()].to_vec())
}
