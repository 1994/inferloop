//! Native GGUF text package: synthesized assets plus raw tensor blocks.
//!
//! The package keeps GGUF blocks in place and hands grouped-int tensors to the backend as raw
//! blocks; each backend packs them for its own kernels. Dense tensors are decoded on demand for
//! reference or small auxiliary tensors. No intermediate safetensors package is written.
use super::arch::{GgufArchitecture, LayerPlan};
use super::requirements::encoding_for;
use super::tensor::{TensorSpec, ValueTransform};
use super::tokenizer::{tokenizer_config, tokenizer_json};
use super::vision::GgufVisionPackage;
use crate::{ModelRegistry, default_registry};
use infer_core::{Error, Result};
use infer_format::{GgufFile, GgufType, dequantize};
use infer_ir::ExecutableEncoding;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One stored tensor, addressed by its canonical HF slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufSlot {
    /// Canonical HF name.
    pub slot: String,
    /// Stored GGUF name.
    pub source: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Stored type.
    pub dtype: GgufType,
    /// The converter's value change, undone when the slot is decoded.
    pub transform: ValueTransform,
}

impl GgufSlot {
    /// The type a float read returns: a transformed slot decodes to F32.
    #[must_use]
    pub fn float_type(&self) -> GgufType {
        if self.transform == ValueTransform::Identity {
            self.dtype
        } else {
            GgufType::F32
        }
    }

    /// Capability encoding this tensor needs.
    /// # Errors
    /// Rejects a type the capability model does not name.
    pub fn encoding(&self) -> Result<ExecutableEncoding> {
        encoding_for(self.dtype)
    }
}

/// A directly opened GGUF text package.
pub struct GgufPackage {
    file: GgufFile,
    architecture: &'static dyn GgufArchitecture,
    tensors: Vec<TensorSpec>,
    slots: BTreeMap<String, GgufSlot>,
    config: Vec<u8>,
    tokenizer: Vec<u8>,
    tokenizer_config: Vec<u8>,
    vision: Option<GgufVisionPackage>,
}

impl GgufPackage {
    /// Open and validate the text tensor set with the built-in adapters.
    /// # Errors
    /// See [`Self::open_in`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_in(default_registry(), path)
    }

    /// Open and validate the text tensor set with the adapter `registry` holds for the file's
    /// `general.architecture`.
    /// # Errors
    /// Rejects an invalid container, an architecture without an adapter, a tensor the adapter
    /// does not map, malformed geometry, or a tokenizer without the expected markers.
    pub fn open_in(registry: &ModelRegistry, path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        let tensors = specs(&file)?;
        let architecture = registry
            .gguf()
            .architecture(file.metadata().require_str("general.architecture")?)?;
        architecture.validate(file.metadata(), &tensors)?;
        let layers = LayerPlan::of(file.metadata(), architecture.name())?;
        let mut slots = BTreeMap::new();
        for tensor in &tensors {
            let mapped = architecture.tensor(tensor, &layers).ok_or_else(|| {
                Error::invalid(format!(
                    "unmapped {} GGUF tensor {}",
                    architecture.name(),
                    tensor.name
                ))
            })?;
            if mapped.transform != ValueTransform::Identity && tensor.ty.grouped_int().is_some() {
                return Err(Error::unsupported(format!(
                    "GGUF tensor {} needs a value transform its quantized type cannot take",
                    tensor.name
                )));
            }
            let slot = GgufSlot {
                slot: mapped.slot.clone(),
                source: tensor.name.clone(),
                shape: mapped.shape.unwrap_or_else(|| tensor.shape.clone()),
                dtype: tensor.ty,
                transform: mapped.transform,
            };
            if slots.insert(mapped.slot.clone(), slot).is_some() {
                return Err(Error::invalid(format!(
                    "two GGUF tensors map to {}",
                    mapped.slot
                )));
            }
        }
        Ok(Self {
            config: architecture.config(file.metadata(), &tensors, None)?,
            tokenizer: tokenizer_json(file.metadata())?,
            tokenizer_config: tokenizer_config(file.metadata())?,
            architecture,
            tensors,
            file,
            slots,
            vision: None,
        })
    }

    /// Open a text package together with a sibling mmproj.
    /// # Errors
    /// Rejects either malformed package, or a text and vision metadata pair whose config cannot
    /// be synthesized.
    pub fn open_with_vision(
        registry: &ModelRegistry,
        path: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut package = Self::open_in(registry, path)?;
        if !package.architecture.accepts_vision() {
            return Err(Error::unsupported(format!(
                "GGUF architecture {} takes no mmproj",
                package.architecture.name()
            )));
        }
        let vision = GgufVisionPackage::open_in(registry, mmproj)?;
        package.config = package.architecture.config(
            package.file.metadata(),
            &package.tensors,
            Some(vision.metadata()),
        )?;
        package.vision = Some(vision);
        Ok(package)
    }

    /// Open a text package with the built-in adapters and attach a sibling `mmproj-*.gguf`
    /// when present.
    /// # Errors
    /// See [`Self::open_auto_in`].
    pub fn open_auto(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_auto_in(default_registry(), path)
    }

    /// Open a text package and attach a sibling `mmproj-*.gguf` when present.
    /// # Errors
    /// Rejects a malformed text package or an ambiguous/malformed sibling.
    pub fn open_auto_in(registry: &ModelRegistry, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        sibling(path).map_or_else(
            || Self::open_in(registry, path),
            |mmproj| Self::open_with_vision(registry, path, mmproj),
        )
    }

    /// Attached mmproj, when this package is multimodal.
    #[must_use]
    pub const fn vision(&self) -> Option<&GgufVisionPackage> {
        self.vision.as_ref()
    }

    /// The `general.architecture` this package was read as.
    #[must_use]
    pub fn architecture(&self) -> &'static str {
        self.architecture.name()
    }

    /// Synthesized `config.json`.
    #[must_use]
    pub fn config(&self) -> &[u8] {
        &self.config
    }

    /// Synthesized `tokenizer.json`.
    #[must_use]
    pub fn tokenizer(&self) -> &[u8] {
        &self.tokenizer
    }

    /// Synthesized `tokenizer_config.json`.
    #[must_use]
    pub fn tokenizer_config(&self) -> &[u8] {
        &self.tokenizer_config
    }

    /// Canonical slots, keyed by HF name.
    #[must_use]
    pub const fn slots(&self) -> &BTreeMap<String, GgufSlot> {
        &self.slots
    }

    /// Raw stored blocks for one HF slot.
    /// # Errors
    /// Rejects an unknown slot or a read over budget.
    pub fn blocks(&self, slot: &str, budget: u64) -> Result<Vec<u8>> {
        let source = &self
            .slots
            .get(slot)
            .ok_or_else(|| Error::invalid(format!("missing GGUF slot {slot}")))?
            .source;
        self.file.read_tensor(source, budget)
    }

    /// Decode one dense slot to F32.
    /// # Errors
    /// Rejects a non-dense slot, an unknown type, or an exceeded budget.
    pub fn dense(&self, slot: &str, budget: u64) -> Result<Vec<f32>> {
        let header = self
            .slots
            .get(slot)
            .ok_or_else(|| Error::invalid(format!("missing GGUF slot {slot}")))?;
        let elements = header
            .shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| Error::invalid("GGUF slot shape overflow"))?;
        let mut values = vec![0.0f32; elements];
        dequantize(header.dtype, &self.blocks(slot, budget)?, &mut values)?;
        header.transform.undo(&mut values);
        Ok(values)
    }

    /// The stored bytes of one float slot, as [`GgufSlot::float_type`] reads them: as stored, or
    /// decoded to F32 with the converter's value change undone.
    /// # Errors
    /// Rejects an unknown or quantized slot, or an exceeded budget.
    pub fn float_bytes(&self, slot: &str, budget: u64) -> Result<Vec<u8>> {
        let header = self
            .slots
            .get(slot)
            .ok_or_else(|| Error::invalid(format!("missing GGUF slot {slot}")))?;
        if header.dtype.grouped_int().is_some() {
            return Err(Error::invalid(format!("GGUF slot {slot} is quantized")));
        }
        if header.transform == ValueTransform::Identity {
            return self.blocks(slot, budget);
        }
        Ok(self
            .dense(slot, budget)?
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect())
    }
}

/// The tensor inventory of `file`.
pub(super) fn specs(file: &GgufFile) -> Result<Vec<TensorSpec>> {
    file.tensors()
        .values()
        .map(|tensor| {
            Ok(TensorSpec {
                name: tensor.name().to_owned(),
                shape: tensor.shape()?,
                ty: tensor.ty(),
            })
        })
        .collect()
}

fn sibling(path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?;
    let mut found = None;
    for entry in parent.read_dir().ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("mmproj") && name.ends_with(".gguf") {
            if found.is_some() {
                return None;
            }
            found = Some(entry.path());
        }
    }
    found
}
