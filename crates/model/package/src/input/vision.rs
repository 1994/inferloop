//! Checkpoint inventory of a modality encoder, derived from its declared geometry.
//!
//! This is the contract between a provider's [`ModalityEncoder`] and a backend that builds
//! kernels: a backend reads exactly these tensors, so a checkpoint whose tower geometry differs
//! from the declaration fails loudly instead of running a partially bound encoder.
use infer_core::{Error, Result};
use infer_spi::ModalityEncoder;
use serde::Serialize;

/// Q, K and V projections packed into one encoder tensor.
const QKV_WIDTH: usize = 3;

/// One encoder tensor: canonical slot, checkpoint name and expected shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VisionSlot {
    /// Slot relative to [`ModalityEncoder::prefix`], for example `blocks.3.attn.qkv.weight`.
    pub slot: String,
    /// Full checkpoint tensor name.
    pub tensor: String,
    /// Expected shape, most significant first.
    pub shape: Vec<usize>,
    /// Whether a whole-row kernel consumes this tensor, so it is padded to [`row_tile`].
    pub row: bool,
}

/// Attention head width of an encoder.
/// # Errors
/// Rejects a head count that does not divide the hidden width.
pub fn head_dim(encoder: &ModalityEncoder) -> Result<usize> {
    if encoder.heads == 0 || !encoder.hidden_size.is_multiple_of(encoder.heads) {
        return Err(Error::invalid("encoder heads must divide the hidden width"));
    }
    Ok(encoder.hidden_size / encoder.heads)
}

/// Side of the square learned position grid.
/// # Errors
/// Rejects a position table whose size is not a perfect square.
pub fn position_grid(encoder: &ModalityEncoder) -> Result<usize> {
    let side = integer_sqrt(encoder.position_embeddings);
    if side * side != encoder.position_embeddings || side == 0 {
        return Err(Error::invalid(
            "position embeddings must form a nonempty square grid",
        ));
    }
    Ok(side)
}

/// Tokens merged into one encoder output, and the width the merger concatenates.
/// # Errors
/// Rejects a zero merge size or overflowing merged width.
pub fn merged_width(encoder: &ModalityEncoder) -> Result<usize> {
    let unit = merge_unit(encoder)?;
    encoder
        .hidden_size
        .checked_mul(unit)
        .ok_or_else(|| Error::invalid("encoder merged width overflow"))
}

/// Tokens per merged output (`spatial_merge_size²`).
/// # Errors
/// Rejects a zero merge size or overflowing unit.
pub fn merge_unit(encoder: &ModalityEncoder) -> Result<usize> {
    if encoder.spatial_merge_size == 0 {
        return Err(Error::invalid("encoder merge size must be nonzero"));
    }
    encoder
        .spatial_merge_size
        .checked_mul(encoder.spatial_merge_size)
        .ok_or_else(|| Error::invalid("encoder merge unit overflow"))
}

/// Geometry-independent consistency checks shared by every backend.
/// # Errors
/// Rejects zero widths, non-square position tables, indivisible heads or overflowing projections.
pub fn validate(encoder: &ModalityEncoder) -> Result<()> {
    if encoder.prefix.is_empty()
        || encoder.depth == 0
        || encoder.hidden_size == 0
        || encoder.intermediate_size == 0
        || encoder.in_channels == 0
        || encoder.patch_size == 0
        || encoder.temporal_patch_size == 0
        || encoder.out_hidden_size == 0
    {
        return Err(Error::invalid("encoder geometry must be nonzero"));
    }
    head_dim(encoder)?;
    position_grid(encoder)?;
    merged_width(encoder)?;
    if !encoder.norm_epsilon.is_finite() || encoder.norm_epsilon <= 0.0 {
        return Err(Error::invalid("encoder norm epsilon must be positive"));
    }
    if !encoder.rope_theta.is_finite() || encoder.rope_theta <= 0.0 {
        return Err(Error::invalid("encoder rope theta must be positive"));
    }
    Ok(())
}

/// Complete tensor inventory of a vision encoder, in binding order.
///
/// # Errors
/// Rejects invalid geometry or arithmetic overflow while sizing a projection.
pub fn slots(encoder: &ModalityEncoder) -> Result<Vec<VisionSlot>> {
    validate(encoder)?;
    let hidden = encoder.hidden_size;
    let prefix = &encoder.prefix;
    let mut slots: Vec<VisionSlot> = Vec::new();
    let mut push = |slot: String, shape: Vec<usize>| {
        let tensor = format!("{prefix}{slot}");
        let row = slot.contains(".norm");
        slots.push(VisionSlot {
            slot,
            tensor,
            shape,
            row,
        });
    };

    let qkv = hidden
        .checked_mul(QKV_WIDTH)
        .ok_or_else(|| Error::invalid("encoder qkv width overflow"))?;
    let merged = merged_width(encoder)?;
    push(
        "patch_embed.proj.weight".into(),
        vec![
            hidden,
            encoder.in_channels,
            encoder.temporal_patch_size,
            encoder.patch_size,
            encoder.patch_size,
        ],
    );
    push("patch_embed.proj.bias".into(), vec![hidden]);
    push(
        "pos_embed.weight".into(),
        vec![encoder.position_embeddings, hidden],
    );
    for layer in 0..encoder.depth {
        let block = format!("blocks.{layer}");
        push(format!("{block}.norm1.weight"), vec![hidden]);
        push(format!("{block}.norm1.bias"), vec![hidden]);
        push(format!("{block}.norm2.weight"), vec![hidden]);
        push(format!("{block}.norm2.bias"), vec![hidden]);
        push(format!("{block}.attn.qkv.weight"), vec![qkv, hidden]);
        push(format!("{block}.attn.qkv.bias"), vec![qkv]);
        push(format!("{block}.attn.proj.weight"), vec![hidden, hidden]);
        push(format!("{block}.attn.proj.bias"), vec![hidden]);
        push(
            format!("{block}.mlp.linear_fc1.weight"),
            vec![encoder.intermediate_size, hidden],
        );
        push(
            format!("{block}.mlp.linear_fc1.bias"),
            vec![encoder.intermediate_size],
        );
        push(
            format!("{block}.mlp.linear_fc2.weight"),
            vec![hidden, encoder.intermediate_size],
        );
        push(format!("{block}.mlp.linear_fc2.bias"), vec![hidden]);
    }
    push("merger.norm.weight".into(), vec![hidden]);
    push("merger.norm.bias".into(), vec![hidden]);
    push("merger.linear_fc1.weight".into(), vec![merged, merged]);
    push("merger.linear_fc1.bias".into(), vec![merged]);
    push(
        "merger.linear_fc2.weight".into(),
        vec![encoder.out_hidden_size, merged],
    );
    push(
        "merger.linear_fc2.bias".into(),
        vec![encoder.out_hidden_size],
    );
    Ok(slots)
}

/// Integer square root by Newton iteration; exact for the sizes used here.
const fn integer_sqrt(value: usize) -> usize {
    if value < 2 {
        return value;
    }
    let mut guess = value;
    let mut next = guess.div_ceil(2);
    while next < guess {
        guess = next;
        next = guess.midpoint(value / guess);
    }
    guess
}

/// Weighted four-tap gather of the learned position table for one image.
///
/// This host reference depends only on the image grid and learned table. Device backends may
/// perform the same interpolation while producing patch embeddings.
///
/// # Errors
/// Rejects a table that disagrees with the declared geometry or overflowing shapes.
pub fn gather_positions(
    table: &[f32],
    encoder: &ModalityEncoder,
    taps: &PositionTaps,
) -> Result<Vec<f32>> {
    let hidden = encoder.hidden_size;
    let rows = encoder.position_embeddings;
    let table_len = rows
        .checked_mul(hidden)
        .ok_or_else(|| Error::invalid("position table overflow"))?;
    if table.len() != table_len {
        return Err(Error::invalid("position table shape"));
    }
    let out_len = taps
        .indices
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| Error::invalid("position output overflow"))?;
    let mut output = vec![0f32; out_len];
    for (patch, (indices, weights)) in taps.indices.iter().zip(&taps.weights).enumerate() {
        let target = &mut output[patch * hidden..(patch + 1) * hidden];
        for (tap, index) in indices.iter().enumerate() {
            let row = usize::try_from(*index).map_err(|_| Error::invalid("position index"))?;
            if row >= rows {
                return Err(Error::invalid("position index out of table"));
            }
            let source = &table[row * hidden..(row + 1) * hidden];
            let weight = weights[tap];
            for (value, row_value) in target.iter_mut().zip(source) {
                *value = weight.mul_add(*row_value, *value);
            }
        }
    }
    Ok(output)
}

/// Tile width of a whole-row kernel: the row length rounded up to a power of two.
///
/// cuTile tile dimensions must be powers of two, so tensors a row kernel consumes whole (the
/// `LayerNorm` weights and biases) are padded to this width with zeros on the device.
/// # Errors
/// Rejects a zero or overflowing row length.
pub fn row_tile(encoder: &ModalityEncoder) -> Result<usize> {
    if encoder.hidden_size == 0 {
        return Err(Error::invalid("encoder hidden size must be nonzero"));
    }
    encoder
        .hidden_size
        .checked_next_power_of_two()
        .ok_or_else(|| Error::invalid("encoder row tile overflow"))
}

/// Axial `(height, width)` grid coordinates of every patch, in spatial-merge-block order.
///
/// # Errors
/// Rejects zero axes, axes not divisible by the merge size, or overflowing grids.
pub fn position_ids(
    grid: (usize, usize, usize),
    encoder: &ModalityEncoder,
) -> Result<Vec<[u32; 2]>> {
    let (temporal, height, width) = grid;
    let merge = encoder.spatial_merge_size;
    if temporal == 0 || height == 0 || width == 0 {
        return Err(Error::invalid("patch grid axes must be nonzero"));
    }
    if !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
        return Err(Error::invalid("patch grid must divide by the merge size"));
    }
    let per_frame = height
        .checked_mul(width)
        .ok_or_else(|| Error::invalid("patch grid overflow"))?;
    let blocks_w = width / merge;
    let mut ids = Vec::with_capacity(per_frame.saturating_mul(temporal));
    for _ in 0..temporal {
        for within in 0..per_frame {
            let in_col = within % merge;
            let in_row = (within / merge) % merge;
            let block_col = (within / (merge * merge)) % blocks_w;
            let block_row = within / (merge * merge * blocks_w);
            ids.push([
                u32::try_from(block_row * merge + in_row)
                    .map_err(|_| Error::invalid("position overflow"))?,
                u32::try_from(block_col * merge + in_col)
                    .map_err(|_| Error::invalid("position overflow"))?,
            ]);
        }
    }
    Ok(ids)
}

/// Axial inverse frequencies of the vision `RoPE`: `head_dim / 4` values of stride two.
///
/// # Errors
/// Rejects an encoder whose head width is not divisible by four.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Head widths are tiny exact integers; the power is computed in f64 and stored as f32"
)]
pub fn rope_frequencies(encoder: &ModalityEncoder) -> Result<Vec<f32>> {
    let head = head_dim(encoder)?;
    let spatial = head / 2;
    if !spatial.is_multiple_of(FREQUENCY_STRIDE) {
        return Err(Error::invalid(
            "encoder head width must hold an even number of axial frequencies",
        ));
    }
    let mut frequencies = Vec::with_capacity(spatial / 2);
    for index in (0..spatial).step_by(FREQUENCY_STRIDE) {
        let exponent = index as f64 / spatial as f64;
        frequencies.push(encoder.rope_theta.powf(-exponent) as f32);
    }
    Ok(frequencies)
}

/// Every second axial lane carries a frequency, mirroring the reference `RoPE` layout.
const FREQUENCY_STRIDE: usize = 2;

/// Bilinear taps per patch: two rows times two columns.
const BILINEAR_TAPS: usize = 4;

/// Interpolation taps of one image grid into the square learned position table.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionTaps {
    /// Four table indices per patch, in spatial-merge-block order.
    pub indices: Vec<[u32; BILINEAR_TAPS]>,
    /// Bilinear weights matching [`Self::indices`].
    pub weights: Vec<[f32; BILINEAR_TAPS]>,
}

/// Position-table taps for one `(temporal, height, width)` patch grid.
///
/// Patches are emitted in spatial-merge-block order, exactly as the encoder's merger expects, and
/// the learned table is bilinearly resampled with `align_corners=true` and border clamping. Both
/// properties are load-bearing: a raster-ordered or half-pixel variant produces wrong values with
/// correct shapes.
///
/// # Errors
/// Rejects zero grid axes, axes not divisible by the merge size, or an invalid position table.
pub fn position_taps(
    grid: (usize, usize, usize),
    encoder: &ModalityEncoder,
) -> Result<PositionTaps> {
    let (temporal, height, width) = grid;
    let merge = encoder.spatial_merge_size;
    let table_side = position_grid(encoder)?;
    if temporal == 0 || height == 0 || width == 0 {
        return Err(Error::invalid("patch grid axes must be nonzero"));
    }
    if !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
        return Err(Error::invalid("patch grid must divide by the merge size"));
    }
    let per_frame = height
        .checked_mul(width)
        .ok_or_else(|| Error::invalid("patch grid overflow"))?;
    let blocks_w = width / merge;
    let mut indices = Vec::with_capacity(per_frame.saturating_mul(temporal));
    let mut weights = Vec::with_capacity(per_frame.saturating_mul(temporal));
    for _ in 0..temporal {
        for within in 0..per_frame {
            let in_col = within % merge;
            let in_row = (within / merge) % merge;
            let block_col = (within / (merge * merge)) % blocks_w;
            let block_row = within / (merge * merge * blocks_w);
            let row = block_row * merge + in_row;
            let col = block_col * merge + in_col;
            let (h_taps, h_weights) = axis_taps(row, height, table_side);
            let (w_taps, w_weights) = axis_taps(col, width, table_side);
            let mut patch_indices = [0u32; BILINEAR_TAPS];
            let mut patch_weights = [0f32; BILINEAR_TAPS];
            for (h_slot, h_tap) in h_taps.into_iter().enumerate() {
                for (w_slot, w_tap) in w_taps.into_iter().enumerate() {
                    let slot = h_slot * 2 + w_slot;
                    patch_indices[slot] = u32::try_from(h_tap * table_side + w_tap)
                        .map_err(|_| Error::invalid("position tap overflow"))?;
                    patch_weights[slot] = h_weights[h_slot] * w_weights[w_slot];
                }
            }
            indices.push(patch_indices);
            weights.push(patch_weights);
        }
    }
    Ok(PositionTaps { indices, weights })
}

/// Two bilinear taps and weights of one axis position.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "Table sides and weights are small exact values; the f64 -> f32 weight is the kernel's own precision"
)]
fn axis_taps(index: usize, size: usize, table_side: usize) -> ([usize; 2], [f32; 2]) {
    let denominator = size.saturating_sub(1).max(1) as f64;
    let source = index as f64 * (table_side - 1) as f64 / denominator;
    let floor = source.floor();
    let mut axis_taps = [0usize; 2];
    let mut axis_weights = [0f32; 2];
    for (slot, offset) in [0.0f64, 1.0].into_iter().enumerate() {
        let raw = floor + offset;
        axis_taps[slot] = raw.clamp(0.0, (table_side - 1) as f64) as usize;
        axis_weights[slot] = (1.0 - (source - raw).abs()).clamp(0.0, 1.0) as f32;
    }
    (axis_taps, axis_weights)
}

#[cfg(test)]
#[path = "../../tests/unit/input_vision.rs"]
mod tests;
