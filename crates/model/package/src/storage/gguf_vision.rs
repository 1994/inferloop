//! Native GGUF mmproj package: vision inventory and the two temporal patch slices.
use crate::providers::qwen::VisionTensor;
use crate::providers::qwen::gguf::encoder;
use crate::providers::qwen::gguf::validate::{TensorSpec, validate_vision};
use crate::providers::qwen::gguf::vision_tensor;
use infer_core::{Error, Result};
use infer_format::{GgufFile, GgufType, dequantize};
use infer_spi::ModalityEncoder;
use std::array::from_fn;
use std::collections::BTreeMap;
use std::path::Path;

/// Tensors in one deepstack merger: norm, fc1, fc2.
const DEEPSTACK_PARTS: usize = 3;

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
    encoder: ModalityEncoder,
    slots: BTreeMap<String, VisionSlot>,
    patch: [Option<VisionSlot>; 2],
    deepstack: BTreeMap<u64, [Option<VisionSlot>; DEEPSTACK_PARTS]>,
}

impl GgufVisionPackage {
    /// Open and validate the vision tensor set.
    /// # Errors
    /// Rejects a malformed container, a missing/extra tensor, or invalid geometry.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        let specs: Vec<TensorSpec> = file
            .tensors()
            .values()
            .map(|tensor| {
                Ok(TensorSpec {
                    name: tensor.name().to_owned(),
                    shape: tensor.shape()?,
                    ty: tensor.ty(),
                })
            })
            .collect::<Result<_>>()?;
        validate_vision(file.metadata(), &specs)?;
        let encoder = encoder(file.metadata())?;
        let deepstack_layers = encoder_deepstack(file.metadata())?;
        let mut slots = BTreeMap::new();
        let mut patch_slices = [None, None];
        let mut deepstack = BTreeMap::new();
        for (source, info) in file.tensors() {
            let Some(mapped) = vision_tensor(source, &deepstack_layers) else {
                continue;
            };
            let slot = VisionSlot {
                slot: String::new(),
                source: source.clone(),
                shape: info.shape()?,
                dtype: info.ty(),
            };
            match mapped {
                VisionTensor::Slot(name) => {
                    slots.insert(
                        name.clone(),
                        VisionSlot {
                            slot: name,
                            ..slot.clone()
                        },
                    );
                }
                VisionTensor::PatchSlice(index) => {
                    if let Some(target) = patch_slices.get_mut(index) {
                        *target = Some(VisionSlot {
                            slot: format!("patch.{index}"),
                            ..slot.clone()
                        });
                    }
                }
            }
            if let Some(rest) = source.strip_prefix("v.deepstack.") {
                let layer = rest
                    .split('.')
                    .next()
                    .and_then(|layer| layer.parse::<u64>().ok())
                    .ok_or_else(|| Error::invalid("deepstack layer"))?;
                let entry = deepstack.entry(layer).or_insert_with(|| from_fn(|_| None));
                let index = if rest.contains(".norm.") {
                    0
                } else if rest.contains(".fc1.") {
                    1
                } else if rest.contains(".fc2.") {
                    2
                } else {
                    continue;
                };
                entry[index] = Some(slot.clone());
            }
        }
        Ok(Self {
            file,
            encoder,
            slots,
            patch: patch_slices,
            deepstack,
        })
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

    /// Deepstack mergers, keyed by their absolute vision layer.
    #[must_use]
    pub const fn deepstack(&self) -> &BTreeMap<u64, [Option<VisionSlot>; DEEPSTACK_PARTS]> {
        &self.deepstack
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

fn encoder_deepstack(metadata: &infer_format::Metadata) -> Result<Vec<u64>> {
    metadata
        .get("clip.vision.is_deepstack_layers")
        .and_then(infer_format::MetadataValue::as_array)
        .ok_or_else(|| Error::invalid("clip deepstack layers"))?
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            value
                .as_bool()
                .filter(|enabled| *enabled)
                .map(|_| index as u64)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(Ok)
        .collect()
}
