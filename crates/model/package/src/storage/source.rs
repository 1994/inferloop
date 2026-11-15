//! One bounded read of the metadata and tensor headers a package exposes.
//!
//! A package's format is an implementation detail of this module: callers ask the source for
//! bytes or typed assets and never join metadata names themselves. Adding a non-HF container
//! means adding a [`TensorFile`] variant and constructing a source for it, not editing every
//! reader.
use crate::constants::{CONFIG_MAX_BYTES, MIB_U64};
use crate::storage::package::read_bounded;
use crate::{HostTensor, ModelRegistry, default_registry, package_path};
use crate::{SafetensorsFile, SafetensorsIndex, TensorHeader};
use infer_core::{Error, ModelId, Result};
use infer_spi::{ImportedModel, ModelProvider};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::str;
use std::sync::{Arc, OnceLock};

/// Maximum accepted tokenizer JSON size in mebibytes.
const TOKENIZER_JSON_MAX_MIB: u64 = 64;
/// Maximum accepted tokenizer JSON size in bytes.
const TOKENIZER_JSON_MAX_BYTES: u64 = TOKENIZER_JSON_MAX_MIB * MIB_U64;
/// Maximum accepted tokenizer-config, chat-template, or card size in bytes.
const TEXT_ASSET_MAX_BYTES: u64 = 2 * MIB_U64;
/// Maximum accepted multimodal preprocessor metadata size in mebibytes.
const PREPROCESSOR_MAX_MIB: u64 = 16;
/// Maximum accepted multimodal preprocessor metadata size in bytes.
const PREPROCESSOR_MAX_BYTES: u64 = PREPROCESSOR_MAX_MIB * MIB_U64;
/// Maximum accepted `model.safetensors.index.json` size.
const INDEX_MAX_BYTES: u64 = 16 * MIB_U64;

/// Read an optional package asset within its size bound.
///
/// Absence is reported as `None`; an inaccessible or oversized file is still an error. The
/// canonicalized root makes the later [`package_path`] check cheap while keeping symlink escape
/// rejection identical to a required asset.
fn read_optional(root: &Path, name: &str, max: u64) -> Result<Option<Vec<u8>>> {
    if !root.join(name).exists() {
        return Ok(None);
    }
    read_bounded(&package_path(root, name)?, max).map(Some)
}

/// One opened tensor-bearing file in its container format.
pub enum TensorFile {
    /// A Hugging Face safetensors shard.
    Safetensors(SafetensorsFile),
}

impl TensorFile {
    /// # Errors
    /// Returns an I/O or invalid-input error for a missing, malformed, or oversized shard.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        SafetensorsFile::open(path).map(Self::Safetensors)
    }

    /// Validated tensor headers, keyed by their stored name.
    #[must_use]
    pub const fn tensors(&self) -> &BTreeMap<String, TensorHeader> {
        match self {
            Self::Safetensors(file) => &file.tensors,
        }
    }

    /// Bytes preceding the validated, contiguous tensor payload.
    #[must_use]
    pub const fn header_bytes(&self) -> u64 {
        match self {
            Self::Safetensors(file) => file.header_bytes(),
        }
    }

    /// Borrow stored tensor bytes without allocating a host staging buffer.
    /// # Errors
    /// Rejects missing tensors, invalid bounds or an exceeded byte budget.
    pub fn bytes(&self, name: &str, budget_bytes: u64) -> Result<&[u8]> {
        match self {
            Self::Safetensors(file) => file.bytes(name, budget_bytes),
        }
    }

    /// # Errors
    /// Returns an I/O, not-found, unsupported, or capacity error if the tensor cannot be read and
    /// converted within the byte budget.
    pub fn read_f32(&mut self, name: &str, budget_bytes: u64) -> Result<HostTensor> {
        match self {
            Self::Safetensors(file) => file.read_f32(name, budget_bytes),
        }
    }

    /// # Errors
    /// Returns an I/O, invalid-input, unsupported, or capacity error for invalid rows, tensor
    /// formats, or an insufficient byte budget.
    pub fn read_row_f32(&mut self, name: &str, row: usize, budget: u64) -> Result<Vec<f32>> {
        match self {
            Self::Safetensors(file) => file.read_row_f32(name, row, budget),
        }
    }

    /// # Errors
    /// Returns I/O, format, numeric, or callback errors. Earlier chunks may already be consumed.
    pub fn visit_float_chunks(
        &mut self,
        name: &str,
        chunk_bytes: usize,
        consume: impl FnMut(u64, &[u8]) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Safetensors(file) => file.visit_float_chunks(name, chunk_bytes, consume),
        }
    }
}

/// A package's shard files and the stored-name to shard map they imply.
pub struct TensorFiles {
    pub shards: BTreeMap<String, TensorFile>,
    pub tensor_map: BTreeMap<String, String>,
}

/// Open every shard an HF index names, or the single `model.safetensors` shard.
///
/// # Errors
/// Rejects an unreadable index, duplicate tensor names, a shard set that disagrees with the
/// index, and a declared `total_size` that matches neither payload convention.
pub fn load_tensor_files(root: &Path) -> Result<TensorFiles> {
    let index_path = root.join("model.safetensors.index.json");
    let index = if index_path.exists() {
        Some(SafetensorsIndex::parse(&read_bounded(
            &package_path(root, "model.safetensors.index.json")?,
            INDEX_MAX_BYTES,
        )?)?)
    } else {
        None
    };
    let names = index.as_ref().map_or_else(
        || vec!["model.safetensors".into()],
        SafetensorsIndex::shards,
    );
    let mut shards = BTreeMap::new();
    let mut tensor_map = BTreeMap::new();
    for name in names {
        let file = TensorFile::open(package_path(root, &name)?)?;
        for tensor in file.tensors().keys() {
            if tensor_map.insert(tensor.clone(), name.clone()).is_some() {
                return Err(Error::invalid(format!(
                    "tensor {tensor} duplicated across shards"
                )));
            }
        }
        shards.insert(name, file);
    }
    if let Some(index) = &index {
        if index.weight_map != tensor_map {
            return Err(Error::invalid(
                "shard contents disagree with HF weight index",
            ));
        }
        let bytes = shards
            .values()
            .flat_map(|s| s.tensors().values())
            .try_fold(0u64, |sum, h| sum.checked_add(h.byte_len()))
            .ok_or_else(|| Error::invalid("package size overflow"))?;
        // Some compressed-tensors exporters include safetensors headers in
        // total_size. Accept either exact convention; tensor maps, offsets and
        // complete payload coverage have already been independently validated.
        let file_bytes = shards
            .values()
            .try_fold(bytes, |sum, shard| sum.checked_add(shard.header_bytes()))
            .ok_or_else(|| Error::invalid("package file size overflow"))?;
        if let Some(declared) = index.declared_weight_bytes()?
            && bytes != declared
            && file_bytes != declared
        {
            return Err(Error::invalid(
                "HF index total_size disagrees with shard payload",
            ));
        }
    }
    Ok(TensorFiles { shards, tensor_map })
}

/// Bounded reader for a standalone tensor file at the package root.
pub struct TensorSet {
    file: TensorFile,
}

impl TensorSet {
    /// Open a required tensor file such as `readouts.safetensors`.
    /// # Errors
    /// Returns an invalid-input or I/O error when the file is missing or malformed.
    pub fn open(root: &Path, name: &str) -> Result<Self> {
        Ok(Self {
            file: TensorFile::open(package_path(root, name)?)?,
        })
    }

    /// Open a tensor file only when it exists.
    /// # Errors
    /// Returns an I/O or invalid-input error when a present file is malformed.
    pub fn open_optional(root: &Path, name: &str) -> Result<Option<Self>> {
        if !root.join(name).exists() {
            return Ok(None);
        }
        Self::open(root, name).map(Some)
    }

    /// Stored tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.file.tensors().keys().map(String::as_str)
    }

    /// # Errors
    /// Returns an I/O, not-found, unsupported, or capacity error for a missing tensor or an
    /// exceeded budget.
    pub fn read_f32(&mut self, name: &str, budget: u64) -> Result<HostTensor> {
        self.file.read_f32(name, budget)
    }
}

/// One resolved optional asset: absent, read, or the error that prevented the read.
type Asset = StdResult<Option<Vec<u8>>, Error>;

/// One bounded read of a package's metadata.
///
/// Optional assets stay unread until a consumer asks for them, so a weight-only open does not
/// load a tokenizer or model card. The first accessor call reads its asset once and every later
/// call reuses that result; an absent asset is `None`, while an inaccessible or oversized one is
/// the recorded error.
#[derive(Debug)]
pub struct PackageSource {
    root: PathBuf,
    config: Option<Vec<u8>>,
    generation_config: OnceLock<Asset>,
    readme: OnceLock<Asset>,
    preprocessor_config: OnceLock<Asset>,
    video_preprocessor_config: OnceLock<Asset>,
    tokenizer_json: OnceLock<Asset>,
    tokenizer_config: OnceLock<Asset>,
    chat_template_jinja: OnceLock<Asset>,
}

impl PackageSource {
    /// Canonicalize the package root and read its required configuration.
    ///
    /// Every other metadata asset is validated on first use, so opening a package for weights
    /// does not pay for text or multimodal assets it will never read.
    ///
    /// # Errors
    /// Returns an I/O, invalid-input, or capacity error for an inaccessible or oversized
    /// configuration, or a root that cannot be canonicalized.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| Error::invalid(e.to_string()))?;
        Ok(Self {
            config: read_optional(&root, "config.json", CONFIG_MAX_BYTES)?,
            generation_config: OnceLock::new(),
            readme: OnceLock::new(),
            preprocessor_config: OnceLock::new(),
            video_preprocessor_config: OnceLock::new(),
            tokenizer_json: OnceLock::new(),
            tokenizer_config: OnceLock::new(),
            chat_template_jinja: OnceLock::new(),
            root,
        })
    }

    /// Canonicalized package root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve an optional asset once, returning a borrow valid for the source's lifetime.
    fn asset<'a>(
        &'a self,
        cell: &'a OnceLock<Asset>,
        name: &str,
        max: u64,
    ) -> Result<Option<&'a [u8]>> {
        match cell.get_or_init(|| read_optional(&self.root, name, max)) {
            Ok(Some(bytes)) => Ok(Some(bytes)),
            Ok(None) => Ok(None),
            Err(error) => Err(error.clone()),
        }
    }

    /// Raw `config.json` bytes, when the package has one.
    #[must_use]
    pub fn config(&self) -> Option<&[u8]> {
        self.config.as_deref()
    }

    /// `config.json` bytes, which every executable package must provide.
    /// # Errors
    /// Returns an invalid-input error when the package has no configuration.
    pub fn require_config(&self) -> Result<&[u8]> {
        self.config
            .as_deref()
            .ok_or_else(|| Error::invalid("package has no config.json"))
    }

    /// Parsed optional `generation_config.json`; absent means an empty object.
    /// # Errors
    /// Returns an invalid-input error for malformed JSON.
    pub fn generation_config(&self) -> Result<serde_json::Value> {
        let bytes = self.asset(
            &self.generation_config,
            "generation_config.json",
            CONFIG_MAX_BYTES,
        )?;
        bytes.map_or_else(
            || Ok(serde_json::json!({})),
            |bytes| serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string())),
        )
    }

    /// Optional model card UTF-8 text.
    /// # Errors
    /// Returns an invalid-input error when the card is not UTF-8.
    pub fn readme(&self) -> Result<Option<&str>> {
        self.asset(&self.readme, "README.md", TEXT_ASSET_MAX_BYTES)?
            .map(|bytes| str::from_utf8(bytes).map_err(|e| Error::invalid(e.to_string())))
            .transpose()
    }

    /// Required `preprocessor_config.json` bytes.
    /// # Errors
    /// Returns an invalid-input error when the package has no preprocessor configuration.
    pub fn preprocessor_config(&self) -> Result<&[u8]> {
        self.asset(
            &self.preprocessor_config,
            "preprocessor_config.json",
            PREPROCESSOR_MAX_BYTES,
        )?
        .ok_or_else(|| Error::invalid("package has no preprocessor_config.json"))
    }

    /// Required `video_preprocessor_config.json` bytes.
    /// # Errors
    /// Returns an invalid-input error when the package has no video preprocessor configuration.
    pub fn video_preprocessor_config(&self) -> Result<&[u8]> {
        self.asset(
            &self.video_preprocessor_config,
            "video_preprocessor_config.json",
            PREPROCESSOR_MAX_BYTES,
        )?
        .ok_or_else(|| Error::invalid("package has no video_preprocessor_config.json"))
    }

    /// Raw `tokenizer.json` bytes, when the package has a tokenizer.
    /// # Errors
    /// Returns an I/O, invalid-input, or capacity error for an unreadable or oversized tokenizer.
    pub fn tokenizer_json(&self) -> Result<Option<&[u8]>> {
        self.asset(
            &self.tokenizer_json,
            "tokenizer.json",
            TOKENIZER_JSON_MAX_BYTES,
        )
    }

    /// Raw `tokenizer_config.json` bytes, when the package has one.
    /// # Errors
    /// Returns an I/O, invalid-input, or capacity error for an unreadable or oversized asset.
    pub fn tokenizer_config(&self) -> Result<Option<&[u8]>> {
        self.asset(
            &self.tokenizer_config,
            "tokenizer_config.json",
            TEXT_ASSET_MAX_BYTES,
        )
    }

    /// Raw `chat_template.jinja` bytes, when the package has one.
    /// # Errors
    /// Returns an I/O, invalid-input, or capacity error for an unreadable or oversized asset.
    pub fn chat_template_jinja(&self) -> Result<Option<&[u8]>> {
        self.asset(
            &self.chat_template_jinja,
            "chat_template.jinja",
            TEXT_ASSET_MAX_BYTES,
        )
    }

    /// Whether this package carries the text assets a server needs.
    ///
    /// This only stats the tokenizer file, so deciding whether to serve text does not read it.
    #[must_use]
    pub fn has_text_assets(&self) -> bool {
        self.root.join("tokenizer.json").exists()
    }

    /// Resolve the chat template with the package's existing precedence: a
    /// `chat_template.jinja` file overrides a template embedded in `tokenizer_config.json`.
    /// # Errors
    /// Returns an unsupported error for named templates without a default, and an invalid-input
    /// error for a malformed template field.
    pub fn chat_template(&self) -> Result<Option<String>> {
        let template = match self.tokenizer_config()? {
            Some(bytes) => embedded_template(bytes)?,
            None => None,
        };
        if let Some(bytes) = self.chat_template_jinja()? {
            return Ok(Some(
                str::from_utf8(bytes)
                    .map_err(|e| Error::invalid(e.to_string()))?
                    .to_owned(),
            ));
        }
        Ok(template)
    }

    /// Shard files and their stored-name map.
    /// # Errors
    /// Returns an I/O or invalid-input error for missing, malformed, duplicate, or inconsistent
    /// shards.
    pub fn tensor_files(&self) -> Result<TensorFiles> {
        load_tensor_files(&self.root)
    }
}

/// The `chat_template` field of `tokenizer_config.json`, if present.
fn embedded_template(bytes: &[u8]) -> Result<Option<String>> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| Error::invalid(e.to_string()))?;
    let Some(template) = value.get("chat_template") else {
        return Ok(None);
    };
    if let Some(text) = template.as_str() {
        return Ok(Some(text.to_owned()));
    }
    if let Some(list) = template.as_array() {
        return list
            .iter()
            .find(|entry| entry.get("name").and_then(|v| v.as_str()) == Some("default"))
            .and_then(|entry| entry.get("template"))
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .map(Some)
            .ok_or_else(|| Error::unsupported("named chat templates require a default template"));
    }
    if template.is_null() {
        return Ok(None);
    }
    Err(Error::invalid("invalid chat template asset"))
}

/// A package identified by its provider before any weight is bound.
pub struct PackageProbe {
    source: PackageSource,
    provider: Arc<dyn ModelProvider>,
    imported: ImportedModel,
}

impl PackageProbe {
    /// Probe with the default provider registry.
    /// # Errors
    /// Returns an I/O, invalid-input, or unsupported error for a missing configuration or an
    /// unregistered architecture.
    pub fn open(root: impl AsRef<Path>, id: ModelId) -> Result<Self> {
        Self::open_with(default_registry(), root, id)
    }

    /// Probe with an explicit provider registry.
    /// # Errors
    /// Returns an I/O, invalid-input, or unsupported error for a missing configuration or an
    /// unregistered architecture.
    pub fn open_with(
        registry: &ModelRegistry,
        root: impl AsRef<Path>,
        id: ModelId,
    ) -> Result<Self> {
        let source = PackageSource::open(root)?;
        let config = source.require_config()?;
        let provider = registry.resolve(config)?;
        let imported = provider.import(id, config)?;
        Ok(Self {
            source,
            provider,
            imported,
        })
    }

    /// The metadata source this probe opened.
    #[must_use]
    pub const fn source(&self) -> &PackageSource {
        &self.source
    }

    /// The provider that owns the detected architecture.
    #[must_use]
    pub fn provider(&self) -> &dyn ModelProvider {
        &*self.provider
    }

    /// The imported model and its declared capability plans.
    #[must_use]
    pub const fn imported(&self) -> &ImportedModel {
        &self.imported
    }

    /// Whether the detected package also carries text assets.
    #[must_use]
    pub fn has_text_assets(&self) -> bool {
        self.source.has_text_assets()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/source.rs"]
mod tests;
