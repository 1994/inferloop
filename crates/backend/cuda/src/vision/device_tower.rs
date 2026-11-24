//! Execute compatible vision towers without staging each block on the host.
use super::launches::{Launches, capture};
use super::program;
use super::program::{
    ATTENTION_CONFIG_FIELDS, AttentionMode, COLUMN_TILE, activation, attention_scale,
    attention_tiles, frame_token_count, mlp_width, projection_output, rope_tables,
};
use super::weights::{AttentionLayer, MergerLayer, MlpLayer, VisionWeights, half_tile};
use crate::device::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{DenseBuffers, Gelu, VisionDense, VisionNorm, VisionRope};
use infer_cuda_kernels::runtime::Allocation;
use infer_cuda_kernels::runtime::bf16;
use infer_models::vision::{merge_unit, merged_width, row_tile};
use infer_spi::ModalityEncoder;
use std::iter::once;
use std::mem::{replace, swap};

/// Compact head packing and padded norm rows use the separately qualified block path.
const COMPACT_HALF: usize = 48;
/// Row tile of the vision projections: merger rows are padded to it.
const TOKEN_TILE: usize = 32;
/// Element tile of the copy that stages a tower state as merger input.
const COPY_TILE: usize = 256;

pub(super) fn supports(encoder: &ModalityEncoder) -> Result<bool> {
    Ok(encoder.depth > 0
        && encoder.hidden_size.is_multiple_of(COLUMN_TILE)
        && row_tile(encoder)? == encoder.hidden_size
        && half_tile(encoder)? != COMPACT_HALF)
}

/// Whether [`encode`] can run the mergers in the tower's graph: unpadded merger norms and widths
/// that the projection and copy kernels tile. Staged rows are whole merges, so a merged width the
/// copy tile divides makes every staged state divide too.
pub(super) fn supports_mergers(encoder: &ModalityEncoder) -> Result<bool> {
    let merged = merged_width(encoder)?;
    Ok(supports(encoder)?
        && merged.is_multiple_of(COPY_TILE)
        && merged.is_power_of_two()
        && merged.is_multiple_of(COLUMN_TILE)
        && encoder.out_hidden_size.is_multiple_of(COLUMN_TILE))
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
            program::dense_plan(
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

/// The tower, its merger and every deepstack merger in one captured graph: the merged
/// embeddings and the feature rows per deepstack level, both `tokens / unit` rows wide.
///
/// A tapped layer's state stays in its own buffer: once the next layer has read it, that buffer
/// leaves the swap and a fresh one takes its place, so no copy or graph split is needed.
pub(super) fn encode(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    (input, grid, tokens): (&[f32], (usize, usize, usize), usize),
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let elements = tokens
        .checked_mul(encoder.hidden_size)
        .ok_or_else(|| Error::invalid("vision input overflow"))?;
    let taps = &encoder.deepstack_layers;
    if tokens == 0
        || input.len() != elements
        || taps.windows(2).any(|pair| pair[0] >= pair[1])
        || taps.iter().any(|&tap| tap >= encoder.depth)
    {
        return Err(Error::invalid("vision encode input or deepstack layers"));
    }
    let geometry = Geometry::new(encoder, grid, tokens)?;
    let layers = (0..encoder.depth)
        .map(|index| Layer::new(weights, index))
        .collect::<Result<Vec<_>>>()?;
    let (cosines, sines) = rope_tables(encoder, grid, geometry.rows)?;
    let cosines = device.runtime().upload(cosines)?;
    let sines = device.runtime().upload(sines)?;
    let mut buffers = Buffers::new(device, &geometry, input)?;
    let mut fresh = (0..taps.len())
        .map(|_| activation(device, geometry.rows, geometry.hidden))
        .collect::<Result<Vec<_>>>()?;
    let mergers = Mergers::new(device, encoder, tokens, taps.len())?;
    let mut tapped = Vec::with_capacity(taps.len());
    let graph = capture(device, |launches| {
        let mut pending = false;
        for (index, layer) in layers.iter().enumerate() {
            buffers.record(launches, layer, &geometry, &cosines, &sines)?;
            if pending {
                // The previous layer's output sits in `next` now that this layer has read it.
                let replacement = fresh
                    .pop()
                    .ok_or_else(|| Error::invariant("vision tap buffers"))?;
                tapped.push(replace(&mut buffers.next, replacement));
            }
            pending = taps.contains(&index);
        }
        mergers.record(launches, weights.merger()?, &buffers.state, 0)?;
        for (level, state) in tapped
            .iter()
            .chain(pending.then_some(&buffers.state))
            .enumerate()
        {
            mergers.record(launches, weights.deepstack_merger(level)?, state, level + 1)?;
        }
        Ok(())
    })?;
    device.tile().sync(graph.launch())?;
    let mut outputs = mergers.read(device)?.into_iter();
    let merged = outputs
        .next()
        .ok_or_else(|| Error::invariant("vision merger output"))?;
    Ok((merged, outputs.collect()))
}

/// Device buffers of the mergers [`encode`] records, shared by every level in graph order.
struct Mergers {
    tokens: usize,
    hidden: usize,
    merged: usize,
    out_hidden: usize,
    rows: usize,
    epsilon: f32,
    input: Allocation<f32>,
    normed: Allocation<f32>,
    fc1: Allocation<f32>,
    activated: Allocation<f32>,
    outputs: Vec<Allocation<f32>>,
}

impl Mergers {
    fn new(
        device: &CudaDevice,
        encoder: &ModalityEncoder,
        tokens: usize,
        levels: usize,
    ) -> Result<Self> {
        let unit = merge_unit(encoder)?;
        let merged = merged_width(encoder)?;
        if !tokens.is_multiple_of(unit) {
            return Err(Error::invalid("vision merger geometry"));
        }
        let rows = tokens / unit;
        let padded = rows
            .checked_next_multiple_of(TOKEN_TILE)
            .ok_or_else(|| Error::invalid("vision merger row padding overflow"))?;
        let wide = padded
            .checked_mul(merged)
            .ok_or_else(|| Error::invalid("vision merger buffer overflow"))?;
        let output = padded
            .checked_mul(encoder.out_hidden_size)
            .ok_or_else(|| Error::invalid("vision merger output overflow"))?;
        let zeros = || device.runtime().zeros::<f32>(wide);
        Ok(Self {
            tokens,
            hidden: encoder.hidden_size,
            merged,
            out_hidden: encoder.out_hidden_size,
            rows,
            epsilon: encoder.norm_epsilon,
            input: zeros()?,
            normed: zeros()?,
            fc1: zeros()?,
            activated: zeros()?,
            outputs: (0..=levels)
                .map(|_| device.runtime().zeros::<f32>(output))
                .collect::<Result<_>>()?,
        })
    }

    /// One merger over `state`'s first `tokens` rows into output `index`. A norm as wide as the
    /// tower normalizes each patch before the merge; one as wide as the merge normalizes the
    /// merged row after it.
    fn record(
        &self,
        launches: &Launches<'_>,
        layer: &MergerLayer,
        state: &Allocation<f32>,
        index: usize,
    ) -> Result<()> {
        let staged = self.tokens * self.hidden;
        launches.copy(
            buffer(&self.input)?
                .slice(0..staged)
                .map_err(device_error)?,
            buffer(state)?.slice(0..staged).map_err(device_error)?,
        )?;
        let dim = if layer.norm.0.len() == self.hidden {
            self.hidden
        } else if layer.norm.0.len() == self.merged {
            self.merged
        } else {
            return Err(Error::invalid("vision merger norm width"));
        };
        launches.layernorm(VisionNorm {
            out: buffer(&self.normed)?,
            input: buffer(&self.input)?,
            weight: buffer(&layer.norm.0)?,
            bias: buffer(&layer.norm.1)?,
            dim,
            tile: dim,
            epsilon: self.epsilon,
        })?;
        launches.dense_bias(VisionDense {
            out: buffer(&self.fc1)?,
            input: buffer(&self.normed)?,
            weight: buffer(&layer.fc1.0)?,
            bias: buffer(&layer.fc1.1)?,
            columns: self.merged,
        })?;
        launches.gelu(
            Gelu::Erf,
            self.merged,
            buffer(&self.activated)?,
            buffer(&self.fc1)?,
        )?;
        let output = self
            .outputs
            .get(index)
            .ok_or_else(|| Error::invariant("vision merger output index"))?;
        launches.dense_bias(VisionDense {
            out: buffer(output)?,
            input: buffer(&self.activated)?,
            weight: buffer(&layer.fc2.0)?,
            bias: buffer(&layer.fc2.1)?,
            columns: self.merged,
        })
    }

    fn read(&self, device: &CudaDevice) -> Result<Vec<Vec<f32>>> {
        self.outputs
            .iter()
            .map(|output| {
                projection_output(device.runtime().read(output)?, self.rows, self.out_hidden)
            })
            .collect()
    }
}

pub(super) fn run(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    input: &[f32],
    grid: (usize, usize, usize),
    tokens: usize,
) -> Result<Vec<f32>> {
    run_tapped(device, weights, encoder, (input, grid, tokens), &[]).map(|(state, _)| state)
}

/// Run the tower and also return the state after each layer of `taps`, in ascending order.
///
/// Each tap ends one captured segment, so the device runs whole segments and the host reads only
/// the tapped states; with no taps this is one graph, as before.
pub(super) fn run_tapped(
    device: &CudaDevice,
    weights: &VisionWeights,
    encoder: &ModalityEncoder,
    (input, grid, tokens): (&[f32], (usize, usize, usize), usize),
    taps: &[usize],
) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
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
    let mut tapped = Vec::with_capacity(taps.len());
    let mut start = 0;
    for end in taps.iter().map(|&tap| tap + 1).chain(once(encoder.depth)) {
        if end < start || end > encoder.depth {
            return Err(Error::invalid("vision taps must be ascending tower layers"));
        }
        if end > start {
            let graph = capture(device, |launches| {
                for layer in &layers[start..end] {
                    buffers.record(launches, layer, &geometry, &cosines, &sines)?;
                }
                Ok(())
            })?;
            device.tile().sync(graph.launch())?;
        }
        if tapped.len() < taps.len() {
            tapped.push(projection_output(
                device.runtime().read(&buffers.state)?,
                tokens,
                geometry.hidden,
            )?);
        }
        start = end;
    }
    let state = projection_output(
        device.runtime().read(&buffers.state)?,
        tokens,
        geometry.hidden,
    )?;
    Ok((state, tapped))
}
