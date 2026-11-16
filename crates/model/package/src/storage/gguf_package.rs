//! Native GGUF text package: synthesized assets plus raw tensor blocks.
//!
//! The package keeps GGUF blocks in place. `Q4_K`/`Q6_K` tensors are repacked into the host layouts
//! the grouped-int kernels consume; dense tensors are decoded on demand for reference or small
//! auxiliary tensors. No intermediate safetensors package is written.
use crate::providers::qwen::gguf::config as gguf_config;
use crate::providers::qwen::gguf::config_text;
use crate::providers::qwen::gguf::text_tensor;
use crate::providers::qwen::gguf::tokenizer::{tokenizer_config, tokenizer_json};
use crate::providers::qwen::gguf::validate::{TensorSpec, validate_text};
use crate::storage::gguf_requirements::encoding_for;
use crate::storage::gguf_vision::GgufVisionPackage;
use crate::storage::grouped_int::Q4KDevice;
use crate::storage::grouped_int6::Q6KTiled;
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
}

impl GgufSlot {
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
    slots: BTreeMap<String, GgufSlot>,
    config: Vec<u8>,
    tokenizer: Vec<u8>,
    tokenizer_config: Vec<u8>,
    vision: Option<GgufVisionPackage>,
}

impl GgufPackage {
    /// Open and validate the text tensor set.
    /// # Errors
    /// Rejects an invalid container, a missing/extra tensor, malformed geometry, or a tokenizer
    /// without the expected markers.
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
        validate_text(file.metadata(), &specs)?;
        let mut slots = BTreeMap::new();
        for (source, info) in file.tensors() {
            let slot = text_tensor(source)
                .ok_or_else(|| Error::invalid(format!("unmapped GGUF tensor {source}")))?;
            slots.insert(
                slot.clone(),
                GgufSlot {
                    slot,
                    source: source.clone(),
                    shape: info.shape()?,
                    dtype: info.ty(),
                },
            );
        }
        Ok(Self {
            config: config_text(file.metadata())?,
            tokenizer: tokenizer_json(file.metadata())?,
            tokenizer_config: tokenizer_config(file.metadata())?,
            file,
            slots,
            vision: None,
        })
    }

    /// Open a text package together with a sibling mmproj.
    /// # Errors
    /// Rejects either malformed package, or a text and vision metadata pair whose config cannot
    /// be synthesized.
    pub fn open_with_vision(path: impl AsRef<Path>, mmproj: impl AsRef<Path>) -> Result<Self> {
        let mut package = Self::open(path)?;
        let vision = GgufVisionPackage::open(mmproj)?;
        package.config = gguf_config(package.file.metadata(), vision.metadata())?;
        package.vision = Some(vision);
        Ok(package)
    }

    /// Open a text package and attach a sibling `mmproj-*.gguf` when present.
    /// # Errors
    /// Rejects a malformed text package or an ambiguous/malformed sibling.
    pub fn open_auto(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        sibling(path).map_or_else(
            || Self::open(path),
            |mmproj| Self::open_with_vision(path, mmproj),
        )
    }

    /// Attached mmproj, when this package is multimodal.
    #[must_use]
    pub const fn vision(&self) -> Option<&GgufVisionPackage> {
        self.vision.as_ref()
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

    /// Repack one `Q4_K` matrix into the grouped-int4 host layout.
    /// # Errors
    /// Rejects a non-`Q4_K`/rank-2 slot or an exceeded budget.
    pub fn q4k(&self, slot: &str, budget: u64) -> Result<Q4KDevice> {
        let header = self
            .slots
            .get(slot)
            .ok_or_else(|| Error::invalid(format!("missing GGUF slot {slot}")))?;
        if header.dtype != GgufType::Q4K {
            return Err(Error::unsupported(format!("{slot} is not Q4_K")));
        }
        let [rows, columns] = header.shape.as_slice() else {
            return Err(Error::unsupported("Q4_K slot rank"));
        };
        Q4KDevice::from_blocks(*rows, *columns, &self.blocks(slot, budget)?)
    }

    /// Repack one `Q6_K` matrix into the grouped-int6 host layout.
    /// # Errors
    /// Rejects a non-`Q6_K`/rank-2 slot or an exceeded budget.
    pub fn q6k(&self, slot: &str, budget: u64) -> Result<Q6KTiled> {
        let header = self
            .slots
            .get(slot)
            .ok_or_else(|| Error::invalid(format!("missing GGUF slot {slot}")))?;
        if header.dtype != GgufType::Q6K {
            return Err(Error::unsupported(format!("{slot} is not Q6_K")));
        }
        let [rows, columns] = header.shape.as_slice() else {
            return Err(Error::unsupported("Q6_K slot rank"));
        };
        Q6KTiled::from_blocks(*rows, *columns, &self.blocks(slot, budget)?)
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
        Ok(values)
    }
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
