//! Native GGUF mmproj package: the vision inventory a projector adapter maps, and the two
//! temporal patch slices llama.cpp stores in place of one `Conv3D` weight.
use super::arch::GgufProjector;
use super::package::specs;
use super::tensor::VisionTensor;
use crate::{ModelRegistry, default_registry};
use infer_core::{Error, Result};
use infer_format::{GgufFile, GgufType, Metadata, dequantize};
use infer_spi::ModalityEncoder;
use std::collections::BTreeMap;
use std::path::Path;

/// One vision tensor in the existing HF inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionSlot {
    /// Canonical HF name.
    pub slot: String,
    /// Stored GGUF name.
    pub source: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Stored type.
    pub dtype: GgufType,
}

/// A directly opened mmproj package.
pub struct GgufVisionPackage {
    file: GgufFile,
    projector: &'static dyn GgufProjector,
    encoder: ModalityEncoder,
    slots: BTreeMap<String, VisionSlot>,
    patch: [Option<VisionSlot>; 2],
}

impl GgufVisionPackage {
    /// Open and validate the vision tensor set with the built-in adapters.
    /// # Errors
    /// See [`Self::open_in`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_in(default_registry(), path)
    }

    /// Open and validate the vision tensor set with the adapter `registry` holds for the file's
    /// `clip.projector_type`.
    /// # Errors
    /// Rejects a malformed container, a projector without an adapter, a missing/extra tensor, or
    /// invalid geometry.
    pub fn open_in(registry: &ModelRegistry, mmproj: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(mmproj)?;
        let projector = registry
            .gguf()
            .projector(file.metadata().require_str("clip.projector_type")?)?;
        projector.validate(file.metadata(), &specs(&file)?)?;
        let encoder = projector.encoder(file.metadata())?;
        let mut slots = BTreeMap::new();
        let mut patch = [None, None];
        for (source, info) in file.tensors() {
            let Some(mapped) = projector.tensor(source, &encoder) else {
                continue;
            };
            let slot = |name: String| -> Result<VisionSlot> {
                Ok(VisionSlot {
                    slot: name,
                    source: source.clone(),
                    shape: info.shape()?,
                    dtype: info.ty(),
                })
            };
            match mapped {
                VisionTensor::Slot(name) => {
                    slots.insert(name.clone(), slot(name)?);
                }
                VisionTensor::PatchSlice(index) => {
                    let target = patch
                        .get_mut(index)
                        .ok_or_else(|| Error::invalid(format!("patch slice {index}")))?;
                    *target = Some(slot(format!("patch.{index}"))?);
                }
            }
        }
        Ok(Self {
            file,
            projector,
            encoder,
            slots,
            patch,
        })
    }

    /// Stored metadata of the mmproj.
    #[must_use]
    pub const fn metadata(&self) -> &Metadata {
        self.file.metadata()
    }

    /// Existing vision-encoder declaration.
    #[must_use]
    pub const fn encoder(&self) -> &ModalityEncoder {
        &self.encoder
    }

    /// Standard HF vision slots.
    #[must_use]
    pub const fn slots(&self) -> &BTreeMap<String, VisionSlot> {
        &self.slots
    }

    /// The two temporal patch slices.
    #[must_use]
    pub const fn patch(&self) -> &[Option<VisionSlot>; 2] {
        &self.patch
    }

    /// Fuse the two temporal patch slices into the declared 5D convolution weight.
    /// # Errors
    /// Rejects a missing slice, a shape other than `[hidden, channels, 1, patch, patch]`, or an
    /// exceeded read budget.
    pub fn patch_weight(&self, budget: u64) -> Result<Vec<f32>> {
        let [left, right] = self.patch();
        let (left, right) = (
            left.as_ref()
                .ok_or_else(|| Error::invalid("missing patch slice 0"))?,
            right
                .as_ref()
                .ok_or_else(|| Error::invalid("missing patch slice 1"))?,
        );
        let [hidden, channels, patch, patch_width] = left.shape.as_slice() else {
            return Err(Error::invalid("patch slice rank"));
        };
        if left.shape != right.shape {
            return Err(Error::invalid("patch slice shape"));
        }
        let values = [self.dense(left, budget)?, self.dense(right, budget)?];
        let plane = hidden * channels * patch * patch_width;
        if values.iter().any(|slice| slice.len() != plane) {
            return Err(Error::invalid("patch slice payload"));
        }
        let per_channel = patch * patch_width;
        let mut fused = vec![0.0f32; plane * 2];
        for hidden_index in 0..*hidden {
            for channel in 0..*channels {
                let base = (hidden_index * channels + channel) * per_channel;
                let out = (hidden_index * channels + channel) * per_channel * 2;
                fused[out..out + per_channel].copy_from_slice(&values[0][base..base + per_channel]);
                fused[out + per_channel..out + per_channel * 2]
                    .copy_from_slice(&values[1][base..base + per_channel]);
            }
        }
        Ok(fused)
    }

    /// The fused patch weight as little-endian BF16 bytes for the existing encoder loader.
    /// # Errors
    /// Rejects a malformed patch pair or an exceeded budget.
    pub fn patch_bf16(&self, budget: u64) -> Result<Vec<u8>> {
        let values = self.patch_weight(budget)?;
        Ok(values
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect())
    }

    /// The `preprocessor_config.json` the projector synthesizes.
    /// # Errors
    /// Rejects missing image normalization metadata.
    pub fn preprocessor_config(&self) -> Result<Vec<u8>> {
        self.projector.preprocessor(self.metadata(), &self.encoder)
    }

    /// Decode one slot to F32.
    /// # Errors
    /// Rejects an unknown slot, an unsupported dtype, or an exceeded budget.
    pub fn dense(&self, slot: &VisionSlot, budget: u64) -> Result<Vec<f32>> {
        let elements = slot
            .shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| Error::invalid("vision tensor shape overflow"))?;
        let mut values = vec![0.0f32; elements];
        dequantize(
            slot.dtype,
            &self.file.read_tensor(&slot.source, budget)?,
            &mut values,
        )?;
        Ok(values)
    }
}

/// Round-to-nearest-even bias for narrowing F32 to BF16.
const BF16_ROUND: u32 = 0x7FFF;
/// Low BF16 mantissa bit used for tie breaking.
const BF16_TIE: u32 = 1;
/// Bits discarded when narrowing to BF16.
const BF16_SHIFT: u32 = 16;

const fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(BF16_ROUND + ((bits >> BF16_SHIFT) & BF16_TIE));
    (rounded >> BF16_SHIFT) as u16
}
