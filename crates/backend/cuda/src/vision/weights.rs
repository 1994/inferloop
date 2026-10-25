//! Vision-encoder weight binding, driven by the provider's declared geometry.
//!
//! The inventory comes from `infer_models::vision::slots`, so a checkpoint whose tower differs
//! from the declaration fails here — before any kernel runs — instead of producing a partially
//! bound encoder. Vision tensors are BF16 in this family, so no quantized path is involved.
use crate::device::{CudaDevice, device_error};
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use infer_models::vision::head_dim;
use infer_models::vision::merged_width;
use infer_models::vision::row_tile;
use infer_models::vision::slots as vision_slots;
use infer_models::{QuantizedPackage, TensorDtype};
use infer_spi::ModalityEncoder;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Encoded staging limit for one vision tensor; the largest is a 4608×4608 merger projection.
const STAGING_BYTES: u64 = 512 * 1024 * 1024;

/// Query, key and value packed into one checkpoint projection.
const QKV: usize = 3;
/// Vision GEMM column tile; padded widths are multiples of it.
const COLUMN_MULTIPLE: usize = 64;

/// Each prepared attention, MLP or merger group replaces two weight/bias pairs in the inventory.
const PREPARED_SLOTS: usize = 4;

/// Half-head tile width: the real half head rounded up to a power-of-two tile.
///
/// The installed cuTile attention/RoPE partition views require power-of-two tile extents. A
/// 72-wide head therefore uses 64/128 until a compact 48/96 kernel is device-validated.
pub fn half_tile(encoder: &ModalityEncoder) -> Result<usize> {
    let head = head_dim(encoder)?;
    if head < 2 || !head.is_multiple_of(2) {
        return Err(Error::invalid("encoder head width must be even"));
    }
    (head / 2)
        .checked_next_power_of_two()
        .ok_or_else(|| Error::invalid("encoder half-head overflow"))
}

/// Padded half-head layout mapped back to real source lanes.
///
/// The two halves stay contiguous and in order; each real half is followed by zero lanes.
pub(super) fn half_layout(real_head: usize, half: usize) -> Vec<Option<usize>> {
    let half_head = real_head / 2;
    let mut layout = Vec::with_capacity(2 * half);
    for half_index in 0..2 {
        for lane in 0..half {
            layout.push((lane < half_head).then_some(half_index * half_head + lane));
        }
    }
    layout
}

/// Prepared attention projections of one block.
///
/// `q`, `k`, `v` are stored with each head laid out as `[half, half]` half-heads padded to a
/// power-of-two tile, so a tile kernel can address either half through a partition axis; the
/// padding lanes are zero and the projections carry matching zero rows/columns.
pub type Prepared = (Arc<Tensor<bf16>>, Arc<Tensor<bf16>>);

/// Encoder slots the prepare passes upload in their own padded layout: block attention and
/// MLP projections and the merger projections. Runtime kernels read only the prepared copies.
#[must_use]
fn prepared_slot(slot: &str) -> bool {
    const BLOCK: [&str; 4] = [
        ".attn.qkv.",
        ".attn.proj.",
        ".mlp.linear_fc1.",
        ".mlp.linear_fc2.",
    ];
    (slot.starts_with("blocks.") && BLOCK.iter().any(|part| slot.contains(part)))
        || slot.starts_with("merger.linear_fc1.")
        || slot.starts_with("merger.linear_fc2.")
}

pub struct AttentionLayer {
    pub q: Prepared,
    pub k: Prepared,
    pub v: Prepared,
    pub proj: Prepared,
}

/// Prepared MLP projections of one block; the intermediate width is padded to a tile multiple
/// with zero rows and columns, which `gelu(0) = 0` keeps neutral.
pub struct MlpLayer {
    pub fc1: Prepared,
    pub fc2: Prepared,
}

/// Prepared merger projections.
pub struct MergerLayer {
    pub fc1: Prepared,
    pub fc2: Prepared,
}

/// Resident vision-encoder weights keyed by canonical slot.
pub struct VisionWeights {
    tensors: BTreeMap<String, Arc<Tensor<bf16>>>,
    attention: Vec<AttentionLayer>,
    mlp: Vec<MlpLayer>,
    merger: Option<MergerLayer>,
    /// Host copy of the learned position table, gathered per image without a device round trip.
    positions: Vec<f32>,
}

impl VisionWeights {
    /// Bind every declared encoder tensor and reject any mismatch.
    ///
    /// # Errors
    /// Rejects invalid geometry, missing tensors, non-BF16 storage, shape disagreement or failed
    /// device uploads.
    pub fn load(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        encoder: &ModalityEncoder,
    ) -> Result<Self> {
        let slots = vision_slots(encoder)?;
        let prepares = encoder.heads > 0 && encoder.depth > 0;
        let mut tensors = BTreeMap::new();
        let mut positions = Vec::new();
        for slot in slots {
            let source = package.source(&slot.tensor)?;
            if source.dtype != TensorDtype::BF16 {
                return Err(Error::unsupported(format!(
                    "{}: vision tensors must be BF16",
                    slot.tensor
                )));
            }
            if source.shape != slot.shape {
                return Err(Error::invalid(format!(
                    "{}: expected {:?}, found {:?}",
                    slot.tensor, slot.shape, source.shape
                )));
            }
            let bytes = package.read(&source, STAGING_BYTES)?;
            let mut values: Vec<bf16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)))
                .collect();
            drop(bytes);
            // Row kernels load a norm tensor whole, so it must span a power-of-two tile.
            let shape = if slot.row {
                let padded = row_tile(encoder)?;
                if padded < values.len() {
                    return Err(Error::invalid(format!(
                        "{}: row tile below row",
                        slot.tensor
                    )));
                }
                values.resize(padded, bf16::from_bits(0));
                vec![padded]
            } else {
                slot.shape.clone()
            };
            if slot.slot == "pos_embed.weight" {
                positions = values.iter().map(|value| value.to_f32()).collect();
            }
            // The prepare passes reread these from the package into their padded layout; a raw
            // device copy would only duplicate the encoder's largest weights.
            if prepares && prepared_slot(&slot.slot) {
                continue;
            }
            tensors.insert(slot.slot, device.upload(values, &shape)?);
        }
        let mut weights = Self {
            tensors,
            attention: Vec::new(),
            mlp: Vec::new(),
            merger: None,
            positions,
        };
        if encoder.heads > 0 && encoder.depth > 0 {
            weights.prepare_attention(device, package, encoder)?;
            weights.prepare_mlp(device, package, encoder)?;
            weights.prepare_merger(device, package, encoder)?;
        }
        Ok(weights)
    }

    /// Prepare the attention projections of every block.
    ///
    /// # Errors
    /// Rejects missing or mis-shaped attention weights, or failed device uploads.
    pub fn prepare_attention(
        &mut self,
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        encoder: &ModalityEncoder,
    ) -> Result<()> {
        let half = half_tile(encoder)?;
        let heads = encoder.heads;
        let width = half
            .checked_mul(2)
            .ok_or_else(|| Error::invalid("padded head overflow"))?;
        let hidden = encoder.hidden_size;
        let rows = heads
            .checked_mul(width)
            .ok_or_else(|| Error::invalid("padded projection overflow"))?;
        let mut layers = Vec::with_capacity(encoder.depth);
        for layer in 0..encoder.depth {
            let base = format!("{}blocks.{layer}.attn.", encoder.prefix);
            let qkv_weight = Self::read_values(package, &format!("{base}qkv.weight"))?;
            let qkv_bias = Self::read_values(package, &format!("{base}qkv.bias"))?;
            let proj_weight = Self::read_values(package, &format!("{base}proj.weight"))?;
            let proj_bias = Self::read_values(package, &format!("{base}proj.bias"))?;
            let real_head = head_dim(encoder)?;
            if qkv_weight.len() != QKV * heads * real_head * hidden {
                return Err(Error::invalid(format!("{base}qkv.weight shape")));
            }
            let zero_weight = || vec![bf16::from_bits(0); rows * hidden];
            let zero_bias = || vec![bf16::from_bits(0); rows];
            let mut packed: Vec<(Vec<bf16>, Vec<bf16>)> =
                (0..QKV).map(|_| (zero_weight(), zero_bias())).collect();
            let layout = half_layout(real_head, half);
            for (which, (weight, bias)) in packed.iter_mut().enumerate() {
                for head_index in 0..heads {
                    for (local, source_lane) in layout.iter().enumerate() {
                        let Some(source_lane) = source_lane else {
                            continue;
                        };
                        let target = head_index * width + local;
                        let source = (which * heads + head_index) * real_head + source_lane;
                        weight[target * hidden..(target + 1) * hidden]
                            .copy_from_slice(&qkv_weight[source * hidden..(source + 1) * hidden]);
                        bias[target] = qkv_bias[source];
                    }
                }
            }
            let mut proj = vec![bf16::from_bits(0); hidden * rows];
            for head_index in 0..heads {
                for (local, source_lane) in layout.iter().enumerate() {
                    let Some(source_lane) = source_lane else {
                        continue;
                    };
                    let column = head_index * width + local;
                    let source = head_index * real_head + source_lane;
                    for row in 0..hidden {
                        proj[row * rows + column] = proj_weight[row * heads * real_head + source];
                    }
                }
            }
            let mut uploaded = Vec::with_capacity(QKV);
            for (weight, bias) in packed {
                uploaded.push((
                    device.upload(weight, &[rows, hidden])?,
                    device.upload(bias, &[rows])?,
                ));
            }
            let [q, k, v]: [Prepared; QKV] = uploaded
                .try_into()
                .map_err(|_| Error::invariant("attention slots"))?;
            layers.push(AttentionLayer {
                q,
                k,
                v,
                proj: (
                    device.upload(proj, &[hidden, rows])?,
                    device.upload(proj_bias, &[hidden])?,
                ),
            });
        }
        self.attention = layers;
        Ok(())
    }

    /// Prepared attention projections of one block.
    /// # Errors
    /// Rejects a block index outside the prepared range.
    pub fn attention(&self, layer: usize) -> Result<&AttentionLayer> {
        self.attention
            .get(layer)
            .ok_or_else(|| Error::invalid("unprepared vision attention block"))
    }

    /// Prepare the MLP projections of every block, padding the intermediate width.
    ///
    /// # Errors
    /// Rejects missing or mis-shaped MLP weights, or failed device uploads.
    pub fn prepare_mlp(
        &mut self,
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        encoder: &ModalityEncoder,
    ) -> Result<()> {
        let hidden = encoder.hidden_size;
        let intermediate = encoder.intermediate_size;
        let padded = intermediate
            .checked_next_multiple_of(COLUMN_MULTIPLE)
            .ok_or_else(|| Error::invalid("padded intermediate overflow"))?;
        let mut layers = Vec::with_capacity(encoder.depth);
        for layer in 0..encoder.depth {
            let base = format!("{}blocks.{layer}.mlp.", encoder.prefix);
            let fc1_weight = Self::read_values(package, &format!("{base}linear_fc1.weight"))?;
            let fc1_bias = Self::read_values(package, &format!("{base}linear_fc1.bias"))?;
            let fc2_weight = Self::read_values(package, &format!("{base}linear_fc2.weight"))?;
            let fc2_bias = Self::read_values(package, &format!("{base}linear_fc2.bias"))?;
            if fc1_weight.len() != intermediate * hidden
                || fc2_weight.len() != hidden * intermediate
                || fc1_bias.len() != intermediate
                || fc2_bias.len() != hidden
            {
                return Err(Error::invalid(format!("{base}shape")));
            }
            let mut fc1 = vec![bf16::from_bits(0); padded * hidden];
            let mut bias1 = vec![bf16::from_bits(0); padded];
            for row in 0..intermediate {
                fc1[row * hidden..(row + 1) * hidden]
                    .copy_from_slice(&fc1_weight[row * hidden..(row + 1) * hidden]);
                bias1[row] = fc1_bias[row];
            }
            let mut fc2 = vec![bf16::from_bits(0); hidden * padded];
            for row in 0..hidden {
                fc2[row * padded..row * padded + intermediate]
                    .copy_from_slice(&fc2_weight[row * intermediate..(row + 1) * intermediate]);
            }
            layers.push(MlpLayer {
                fc1: (
                    device.upload(fc1, &[padded, hidden])?,
                    device.upload(bias1, &[padded])?,
                ),
                fc2: (
                    device.upload(fc2, &[hidden, padded])?,
                    device.upload(fc2_bias, &[hidden])?,
                ),
            });
        }
        self.mlp = layers;
        Ok(())
    }

    /// Prepare the merger projections and its norms.
    ///
    /// # Errors
    /// Rejects missing or mis-shaped merger weights, or failed device uploads.
    pub fn prepare_merger(
        &mut self,
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        encoder: &ModalityEncoder,
    ) -> Result<()> {
        let hidden = encoder.hidden_size;
        let merged = merged_width(encoder)?;
        let out = encoder.out_hidden_size;
        let base = format!("{}merger.", encoder.prefix);
        let fc1_weight = Self::read_values(package, &format!("{base}linear_fc1.weight"))?;
        let fc1_bias = Self::read_values(package, &format!("{base}linear_fc1.bias"))?;
        let fc2_weight = Self::read_values(package, &format!("{base}linear_fc2.weight"))?;
        let fc2_bias = Self::read_values(package, &format!("{base}linear_fc2.bias"))?;
        if fc1_weight.len() != merged * merged
            || fc2_weight.len() != out * merged
            || fc1_bias.len() != merged
            || fc2_bias.len() != out
        {
            return Err(Error::invalid("merger shape"));
        }
        let _ = hidden;
        self.merger = Some(MergerLayer {
            fc1: (
                device.upload(fc1_weight, &[merged, merged])?,
                device.upload(fc1_bias, &[merged])?,
            ),
            fc2: (
                device.upload(fc2_weight, &[out, merged])?,
                device.upload(fc2_bias, &[out])?,
            ),
        });
        Ok(())
    }

    /// Prepared MLP projections of one block.
    /// # Errors
    /// Rejects a block index outside the prepared range.
    pub fn mlp(&self, layer: usize) -> Result<&MlpLayer> {
        self.mlp
            .get(layer)
            .ok_or_else(|| Error::invalid("unprepared vision MLP block"))
    }

    /// Learned position table, row-major `[positions, hidden]`.
    #[must_use]
    pub fn position_table(&self) -> &[f32] {
        &self.positions
    }

    /// Prepared merger projections.
    /// # Errors
    /// Rejects a package without a prepared merger.
    pub fn merger(&self) -> Result<&MergerLayer> {
        self.merger
            .as_ref()
            .ok_or_else(|| Error::invalid("unprepared vision merger"))
    }

    /// Decoded BF16 values of one checkpoint tensor.
    fn read_values(package: &mut QuantizedPackage, name: &str) -> Result<Vec<bf16>> {
        let source = package.source(name)?;
        if source.dtype != TensorDtype::BF16 {
            return Err(Error::unsupported(format!("{name}: must be BF16")));
        }
        let bytes = package.read(&source, STAGING_BYTES)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)))
            .collect())
    }

    /// One raw bound tensor by canonical slot. Prepared projections are available through
    /// [`Self::attention`], [`Self::mlp`] and [`Self::merger`], without duplicate raw storage.
    /// # Errors
    /// Rejects an unknown slot or a projection retained only in its prepared layout.
    pub fn get(&self, slot: &str) -> Result<&Arc<Tensor<bf16>>> {
        self.tensors
            .get(slot)
            .ok_or_else(|| Error::invalid(format!("unbound vision tensor {slot}")))
    }

    /// Bound checkpoint slot count, including projections retained only in prepared layouts.
    /// This is the declared inventory size, not the number of physical device tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
            + PREPARED_SLOTS
                * (self.attention.len() + self.mlp.len() + usize::from(self.merger.is_some()))
    }

    /// Whether no tensor was bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Two-dimensional view of a bound tensor, for kernels that need explicit shapes.
/// # Errors
/// Rejects a tensor whose rank does not match.
pub fn matrix(tensor: &Arc<Tensor<bf16>>) -> Result<[usize; 2]> {
    let shape = tensor.shape();
    let [rows, columns] = shape else {
        return Err(Error::invalid("vision kernel requires a matrix"));
    };
    Ok([
        usize::try_from(*rows).map_err(device_error)?,
        usize::try_from(*columns).map_err(device_error)?,
    ])
}
