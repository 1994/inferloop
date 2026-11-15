//! Validated native storage bindings for mixed FP8/NVFP4 packages.
use crate::constants::CONFIG_MAX_BYTES;
#[cfg(feature = "gguf")]
use crate::storage::gguf_package::GgufPackage;
use crate::storage::source::TensorFiles;
use crate::{
    ImportedModel, ModelRegistry, PackageSource, TensorDtype, TensorFile, default_registry,
};
use infer_core::{Error, ErrorCode, ModelId, Result};
use infer_ir::{DataflowGraph, TensorStorage};
use infer_spi::ModelProvider;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, sync::Arc};

mod activations;
mod draft;
pub use draft::DFlash2Package;

/// Values sharing one E4M3 block scale in an NVFP4 weight.
const NVFP4_GROUP_SIZE: usize = 16;
/// Bits per weight in channel-wise FP8 quantization.
const FP8_WEIGHT_BITS: usize = 8;
/// Bits per weight in NVFP4 quantization.
const NVFP4_WEIGHT_BITS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightEncoding {
    Float,
    /// Per-output-channel FP8 scales (`weight_scale`).
    Fp8Channel,
    /// Block-scaled FP8 (`weight_scale_inv`, one scale per 128x128 block).
    Fp8Block,
    Nvfp4,
    /// GGML `Q4_K` grouped integers.
    GroupedInt4,
    /// GGML `Q6_K` grouped integers.
    GroupedInt6,
}

/// Grouped-integer block family carried by a weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupedIntKind {
    /// `Q4_K`.
    Q4K,
    /// `Q6_K`.
    Q6K,
}

/// Contiguous view into an immutable validated stored tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightView {
    pub stored_shape: Vec<usize>,
    pub element_offset: u64,
    #[serde(default)]
    pub rows: Vec<usize>,
    /// `[first, count]` columns kept from every selected row; `None` keeps whole rows.
    #[serde(default)]
    pub columns: Option<[usize; 2]>,
}
impl WeightView {
    /// Whether the view selects rows or columns, so its bytes need an owned gather.
    #[must_use]
    pub const fn gathered(&self) -> bool {
        !self.rows.is_empty() || self.columns.is_some()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSource {
    pub name: String,
    pub shard: String,
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub bytes: u64,
    #[serde(default)]
    pub view: Option<WeightView>,
}

/// A validated weight in the encoding its checkpoint stores.
///
/// Each variant owns exactly the sources its encoding needs: a per-channel FP8 weight carries
/// one scale vector, a block weight also carries its block geometry, and NVFP4 carries both
/// global scales. There is no state in which an encoding is missing a source it requires.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[expect(
    clippy::large_enum_variant,
    reason = "one entry per package weight in a load-time map; boxing NVFP4's four sources would add an allocation per weight without changing steady-state access"
)]
pub enum DeviceWeight {
    /// Unquantized stored floating-point values.
    Float { data: WeightSource },
    /// One FP8 scale per output channel.
    Fp8Channel {
        data: WeightSource,
        scale: WeightSource,
        /// Activations are dynamically quantized per token before the GEMM.
        token_input: bool,
    },
    /// Block-scaled FP8: `block` is the `[rows, columns]` of one scale group.
    Fp8Block {
        data: WeightSource,
        scale: WeightSource,
        block: [usize; 2],
        /// Activations are dynamically quantized per token block before the GEMM.
        token_input: bool,
    },
    /// Packed E2M1 code pairs with one E4M3 scale per 16-column group.
    Nvfp4 {
        data: WeightSource,
        scale: WeightSource,
        global: WeightSource,
        input_global: WeightSource,
    },
    /// Raw grouped-integer super-blocks.
    GroupedInt {
        data: WeightSource,
        kind: GroupedIntKind,
    },
}

impl DeviceWeight {
    /// The encoding tag this weight carries.
    #[must_use]
    pub const fn encoding(&self) -> WeightEncoding {
        match self {
            Self::Float { .. } => WeightEncoding::Float,
            Self::Fp8Channel { .. } => WeightEncoding::Fp8Channel,
            Self::Fp8Block { .. } => WeightEncoding::Fp8Block,
            Self::Nvfp4 { .. } => WeightEncoding::Nvfp4,
            Self::GroupedInt { kind, .. } => match kind {
                GroupedIntKind::Q4K => WeightEncoding::GroupedInt4,
                GroupedIntKind::Q6K => WeightEncoding::GroupedInt6,
            },
        }
    }

    /// The stored weight payload.
    #[must_use]
    pub const fn data(&self) -> &WeightSource {
        match self {
            Self::Float { data }
            | Self::Fp8Channel { data, .. }
            | Self::Fp8Block { data, .. }
            | Self::Nvfp4 { data, .. }
            | Self::GroupedInt { data, .. } => data,
        }
    }

    /// Mutable access to the stored payload descriptor, for a load-time layout change.
    pub const fn data_mut(&mut self) -> &mut WeightSource {
        match self {
            Self::Float { data }
            | Self::Fp8Channel { data, .. }
            | Self::Fp8Block { data, .. }
            | Self::Nvfp4 { data, .. }
            | Self::GroupedInt { data, .. } => data,
        }
    }

    /// The scale source, absent for an unquantized weight.
    #[must_use]
    pub const fn scale(&self) -> Option<&WeightSource> {
        match self {
            Self::Float { .. } => None,
            Self::Fp8Channel { scale, .. }
            | Self::Fp8Block { scale, .. }
            | Self::Nvfp4 { scale, .. } => Some(scale),
            Self::GroupedInt { .. } => None,
        }
    }

    /// The NVFP4 weight-global scale, absent for other encodings.
    #[must_use]
    pub const fn global_scale(&self) -> Option<&WeightSource> {
        match self {
            Self::Nvfp4 { global, .. } => Some(global),
            _ => None,
        }
    }

    /// The NVFP4 input-global scale, absent for other encodings.
    #[must_use]
    pub const fn input_global_scale(&self) -> Option<&WeightSource> {
        match self {
            Self::Nvfp4 { input_global, .. } => Some(input_global),
            _ => None,
        }
    }

    /// The FP8 block geometry, absent for other encodings.
    #[must_use]
    pub const fn block(&self) -> Option<[usize; 2]> {
        match self {
            Self::Fp8Block { block, .. } => Some(*block),
            _ => None,
        }
    }

    /// Whether activations are quantized before this weight's GEMM.
    #[must_use]
    pub const fn token_input(&self) -> bool {
        match self {
            Self::Fp8Channel { token_input, .. } | Self::Fp8Block { token_input, .. } => {
                *token_input
            }
            Self::Float { .. } | Self::Nvfp4 { .. } | Self::GroupedInt { .. } => false,
        }
    }

    /// The logical `[rows, columns]` shape, with NVFP4's packed columns expanded.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        let mut shape = self.data().shape.clone();
        if matches!(self, Self::Nvfp4 { .. })
            && let Some(columns) = shape.last_mut()
        {
            // Binding rejects a packed shape whose expanded width overflows the validated graph
            // shape; saturation keeps a mutated descriptor from panicking.
            *columns = columns.saturating_mul(2);
        }
        shape
    }
}

pub struct QuantizedPackage {
    /// Provider that owns this package's architecture and weight-name mapping.
    pub provider: Arc<dyn ModelProvider>,
    pub imported: ImportedModel,
    pub graph: DataflowGraph,
    pub weights: BTreeMap<String, DeviceWeight>,
    pub mtp: BTreeMap<String, DeviceWeight>,
    /// Attached `DFlash2` draft, bound against this target by [`Self::attach_dflash2`].
    pub dflash2: Option<DFlash2Package>,
    /// KV storage declared by the checkpoint, separate from weight encoding.
    pub kv_cache_dtype: Option<TensorDtype>,
    shards: BTreeMap<String, TensorFile>,
    tensor_map: BTreeMap<String, String>,
    activation_rules: Vec<activations::Rule>,
    #[cfg(feature = "gguf")]
    gguf: Option<GgufPackage>,
    #[cfg(feature = "gguf")]
    gguf_dense: BTreeMap<String, Vec<u8>>,
}

impl QuantizedPackage {
    /// Open and validate every text-backbone weight without reading its payload.
    /// # Errors
    /// Rejects unsupported compression, missing scales, mismatched shapes and invalid shards.
    pub fn open(root: impl AsRef<Path>, id: ModelId) -> Result<Self> {
        Self::open_with(default_registry(), root, id)
    }

    /// Open with an explicit provider registry, so a caller can add its own model families.
    /// # Errors
    /// Rejects unsupported compression, mismatched shapes, invalid shards and unregistered
    /// architectures.
    pub fn open_with(
        registry: &ModelRegistry,
        root: impl AsRef<Path>,
        id: ModelId,
    ) -> Result<Self> {
        #[cfg(feature = "gguf")]
        if root
            .as_ref()
            .extension()
            .is_some_and(|extension| extension == "gguf")
        {
            return Self::open_gguf(registry, root.as_ref(), id);
        }
        let source = PackageSource::open(root.as_ref())?;
        let config = source.require_config_bounded(CONFIG_MAX_BYTES)?;
        let mut value: serde_json::Value =
            serde_json::from_slice(config).map_err(|e| Error::invalid(e.to_string()))?;
        if let Some(quant) = value.get("quantization_config") {
            if quant["quant_method"] == "fp8" {
                let normalized = normalize_fp8_scheme(quant)?;
                value["quantization_config"] = normalized;
            }
            let quant = &value["quantization_config"];
            if quant["quant_method"] != "compressed-tensors" {
                return Err(Error::unsupported(
                    "only compressed-tensors or fp8 quantized packages are supported",
                ));
            }
            validate_groups(quant)?;
        }
        let provider = registry.resolve(config)?;
        let imported = provider.import(id, config)?;
        let graph = provider.graph(&imported.model)?;
        let TensorFiles { shards, tensor_map } = source.tensor_files()?;
        let mut package = Self {
            provider,
            imported,
            graph,
            weights: BTreeMap::new(),
            mtp: BTreeMap::new(),
            dflash2: None,
            kv_cache_dtype: declared_kv_dtype(&value["quantization_config"]["kv_cache_scheme"])?,
            shards,
            tensor_map,
            activation_rules: activations::Rule::parse(&value["quantization_config"])?,
            #[cfg(feature = "gguf")]
            gguf: None,
            #[cfg(feature = "gguf")]
            gguf_dense: BTreeMap::new(),
        };
        package.bind_backbone()?;
        package.bind_mtp()?;
        Ok(package)
    }

    /// Open a native GGUF text package.
    /// # Errors
    /// Rejects invalid metadata, a provider that cannot import the config, or an unsupported
    /// weight encoding.
    #[cfg(feature = "gguf")]
    pub fn open_gguf(registry: &ModelRegistry, path: &Path, id: ModelId) -> Result<Self> {
        let package = GgufPackage::open(path)?;
        let config = package.config();
        let provider = registry.resolve(config)?;
        let imported = provider.import(id, config)?;
        let graph = provider.graph(&imported.model)?;
        let mut dense = BTreeMap::new();
        for slot in package.slots().values() {
            if matches!(
                slot.dtype,
                infer_format::GgufType::F32
                    | infer_format::GgufType::F16
                    | infer_format::GgufType::Bf16
            ) {
                dense.insert(slot.slot.clone(), package.blocks(&slot.slot, u64::MAX)?);
            }
        }
        let mut package = Self {
            provider,
            imported,
            graph,
            weights: BTreeMap::new(),
            mtp: BTreeMap::new(),
            dflash2: None,
            kv_cache_dtype: None,
            shards: BTreeMap::new(),
            tensor_map: BTreeMap::new(),
            activation_rules: Vec::new(),
            gguf: Some(package),
            gguf_dense: dense,
        };
        package.bind_gguf()?;
        Ok(package)
    }

    #[cfg(feature = "gguf")]
    fn bind_gguf(&mut self) -> Result<()> {
        let package = self
            .gguf
            .as_ref()
            .ok_or_else(|| Error::invariant("GGUF package"))?;
        let anchor = self.provider.anchor_slot();
        let prefix = self
            .provider
            .weight_prefixes()
            .iter()
            .copied()
            .find(|prefix| package.slots().contains_key(&format!("{prefix}{anchor}")))
            .ok_or_else(|| Error::invalid(format!("missing GGUF anchor {anchor}")))?;
        let mut weights = BTreeMap::new();
        for spec in &self.graph.tensors {
            let TensorStorage::Weight { slot } = &spec.storage else {
                continue;
            };
            let source_name = self.provider.weight_source(slot, prefix);
            let source = package
                .slots()
                .get(&source_name)
                .ok_or_else(|| Error::invalid(format!("missing GGUF slot {source_name}")))?;
            let elements = source
                .shape
                .iter()
                .try_fold(1u64, |count, dimension| {
                    count.checked_mul(*dimension as u64)
                })
                .ok_or_else(|| Error::invalid("GGUF weight shape overflow"))?;
            let data = WeightSource {
                name: source_name,
                shard: "gguf".to_owned(),
                dtype: TensorDtype::U8,
                shape: source.shape.clone(),
                bytes: source
                    .dtype
                    .byte_len(elements)
                    .map_err(|_| Error::unsupported("GGUF weight byte length"))?,
                view: None,
            };
            let weight = match source.dtype {
                infer_format::GgufType::Q4K => DeviceWeight::GroupedInt {
                    data,
                    kind: GroupedIntKind::Q4K,
                },
                infer_format::GgufType::Q6K => DeviceWeight::GroupedInt {
                    data,
                    kind: GroupedIntKind::Q6K,
                },
                _ => DeviceWeight::Float { data },
            };
            weights.insert(slot.clone(), weight);
        }
        self.weights = weights;
        Ok(())
    }

    /// Read one stored tensor under an explicit staging budget.
    /// # Errors
    /// Returns errors for a foreign source descriptor, I/O failure, or exceeded budget.
    pub fn read(&mut self, source: &WeightSource, budget: u64) -> Result<Vec<u8>> {
        #[cfg(feature = "gguf")]
        if source.shard == "gguf"
            && let Some(package) = &self.gguf
        {
            return package.blocks(&source.name, budget);
        }
        let Some(view) = source.view.as_ref().filter(|view| view.gathered()) else {
            return Ok(self.read_view(source, budget)?.to_vec());
        };
        let file = self
            .shards
            .get(&source.shard)
            .ok_or_else(|| Error::invalid("unknown weight shard"))?;
        let header = file
            .tensors()
            .get(&source.name)
            .ok_or_else(|| Error::invalid("unknown weight tensor"))?;
        let [stored_rows, columns] = header.shape.as_slice() else {
            return Err(Error::invalid("gathered weight rank"));
        };
        let [first, width] = view.columns.unwrap_or([0, *columns]);
        let rows: Vec<usize> = if view.rows.is_empty() {
            (0..*stored_rows).collect()
        } else {
            view.rows.clone()
        };
        let bytes = rows
            .len()
            .checked_mul(width)
            .and_then(|n| (n as u64).checked_mul(source.dtype.bytes()))
            .ok_or_else(|| Error::invalid("gathered weight size overflow"))?;
        if source.bytes != bytes
            || source.shape != [rows.len(), width]
            || view.stored_shape != header.shape
            || view.element_offset != 0
            || source.dtype != header.dtype
            || first.checked_add(width).is_none_or(|end| end > *columns)
            || rows.iter().any(|row| *row >= *stored_rows)
        {
            return Err(Error::invalid("gathered weight descriptor changed"));
        }
        if bytes > budget {
            return Err(Error::new(
                ErrorCode::Capacity,
                "gathered weight exceeds staging budget",
            ));
        }
        let element =
            usize::try_from(source.dtype.bytes()).map_err(|e| Error::invalid(e.to_string()))?;
        let row_bytes = columns
            .checked_mul(element)
            .ok_or_else(|| Error::invalid("weight row overflow"))?;
        let (skip, kept) = (first * element, width * element);
        let stored = file.bytes(&source.name, u64::MAX)?;
        let mut output =
            Vec::with_capacity(usize::try_from(bytes).map_err(|e| Error::invalid(e.to_string()))?);
        for row in &rows {
            let start = row * row_bytes + skip;
            output.extend_from_slice(&stored[start..start + kept]);
        }
        Ok(output)
    }

    /// Borrow one stored tensor directly from its mapped shard.
    /// # Errors
    /// Rejects foreign source descriptors, invalid bounds or an exceeded budget.
    pub fn read_view(&self, source: &WeightSource, budget: u64) -> Result<&[u8]> {
        #[cfg(feature = "gguf")]
        if source.shard == "gguf" {
            let bytes = self
                .gguf_dense
                .get(&source.name)
                .ok_or_else(|| Error::unsupported("GGUF dense tensor is not cached"))?;
            if bytes.len() as u64 > budget {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "GGUF tensor exceeds staging budget",
                ));
            }
            return Ok(bytes);
        }
        let file = self
            .shards
            .get(&source.shard)
            .ok_or_else(|| Error::invalid("unknown weight shard"))?;
        let header = file
            .tensors()
            .get(&source.name)
            .ok_or_else(|| Error::invalid("unknown weight tensor"))?;
        if header.dtype != source.dtype {
            return Err(Error::invalid("weight source dtype changed"));
        }
        if let Some(view) = &source.view {
            if view.gathered() {
                return Err(Error::unsupported("gathered weight requires owned read"));
            }
            let elements = source
                .shape
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
                .ok_or_else(|| Error::invalid("weight view shape overflow"))?;
            let bytes = elements
                .checked_mul(source.dtype.bytes())
                .ok_or_else(|| Error::invalid("weight view size overflow"))?;
            let start = view
                .element_offset
                .checked_mul(source.dtype.bytes())
                .ok_or_else(|| Error::invalid("weight view offset overflow"))?;
            let end = start
                .checked_add(bytes)
                .ok_or_else(|| Error::invalid("weight view range overflow"))?;
            if header.shape != view.stored_shape || source.bytes != bytes || end > header.byte_len()
            {
                return Err(Error::invalid("weight view descriptor changed"));
            }
            if bytes > budget {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "weight view exceeds staging budget",
                ));
            }
            let start = usize::try_from(start).map_err(|e| Error::invalid(e.to_string()))?;
            let end = usize::try_from(end).map_err(|e| Error::invalid(e.to_string()))?;
            return file
                .bytes(&source.name, u64::MAX)?
                .get(start..end)
                .ok_or_else(|| Error::invalid("weight view bounds"));
        }
        if header.shape != source.shape || header.byte_len() != source.bytes {
            return Err(Error::invalid("weight source descriptor changed"));
        }
        file.bytes(&source.name, budget)
    }

    /// Read one embedding row without staging the entire embedding table.
    /// # Errors
    /// Rejects foreign descriptors, invalid rows and excessive staging.
    pub fn read_float_row(
        &mut self,
        source: &WeightSource,
        row: usize,
        budget: u64,
    ) -> Result<Vec<f32>> {
        if source.view.is_some() {
            return Err(Error::unsupported("embedding row from weight view"));
        }
        let file = self
            .shards
            .get_mut(&source.shard)
            .ok_or_else(|| Error::invalid("unknown embedding shard"))?;
        let header = file
            .tensors()
            .get(&source.name)
            .ok_or_else(|| Error::invalid("unknown embedding tensor"))?;
        if header.shape != source.shape
            || header.dtype != source.dtype
            || header.byte_len() != source.bytes
        {
            return Err(Error::invalid("embedding descriptor changed"));
        }
        file.read_row_f32(&source.name, row, budget)
    }

    /// Inspect a tensor descriptor without reading its payload.
    /// # Errors
    /// Rejects names not present in the validated package index.
    pub fn source(&self, name: &str) -> Result<WeightSource> {
        let shard = self
            .tensor_map
            .get(name)
            .ok_or_else(|| Error::invalid(format!("missing weight {name}")))?;
        let header = &self.shards[shard].tensors()[name];
        Ok(WeightSource {
            name: name.into(),
            shard: shard.clone(),
            dtype: header.dtype,
            shape: header.shape.clone(),
            bytes: header.byte_len(),
            view: None,
        })
    }

    fn bind_backbone(&mut self) -> Result<()> {
        let anchor = self.provider.anchor_slot();
        let prefix = self
            .provider
            .weight_prefixes()
            .iter()
            .copied()
            .find(|prefix| self.tensor_map.contains_key(&format!("{prefix}{anchor}")))
            .ok_or_else(|| Error::invalid(format!("missing {anchor} tensor")))?;
        let mut weights = BTreeMap::new();
        for spec in &self.graph.tensors {
            if let TensorStorage::Weight { slot } = &spec.storage {
                let name = self.provider.weight_source(slot, prefix);
                weights.insert(
                    slot.clone(),
                    self.bind_expert_or_matrix(&name, &spec.shape)?,
                );
            }
        }
        self.weights = weights;
        Ok(())
    }

    fn bind_expert_or_matrix(&self, name: &str, shape: &[usize]) -> Result<DeviceWeight> {
        if self.tensor_map.contains_key(name)
            || self
                .tensor_map
                .contains_key(&name.replace(".weight", ".weight_packed"))
        {
            return self.bind(name, shape);
        }
        let Some((prefix, suffix)) = name.rsplit_once(".experts.") else {
            if let Some(weight) = self.bind_linear_view(name, shape)? {
                return Ok(weight);
            }
            return self.bind(name, shape);
        };
        let Some((expert, projection)) = suffix.split_once('.') else {
            return self.bind(name, shape);
        };
        let expert = expert
            .parse::<usize>()
            .map_err(|e| Error::invalid(e.to_string()))?;
        let [rows, columns] = shape else {
            return Err(Error::invalid("expert matrix shape"));
        };
        let (bank, part) = match projection {
            "gate_proj.weight" => ("gate_up_proj", 0),
            "up_proj.weight" => ("gate_up_proj", 1),
            "down_proj.weight" => ("down_proj", 0),
            _ => return self.bind(name, shape),
        };
        let base = [
            format!("{prefix}.experts.{bank}"),
            format!("{prefix}.experts.{bank}.weight"),
        ]
        .into_iter()
        .find(|candidate| self.tensor_map.contains_key(candidate))
        .ok_or_else(|| Error::invalid(format!("unbound expert weight {name}")))?;
        let source = self.source(&base)?;
        let width = if bank == "gate_up_proj" {
            rows.checked_mul(2)
                .ok_or_else(|| Error::invalid("fused expert shape overflow"))?
        } else {
            *rows
        };
        let data = Self::expert_view(source, expert, part * rows, *rows, width, *columns)?;
        if data.dtype != TensorDtype::F8E4m3 {
            if !data.dtype.is_host_float() {
                return Err(Error::unsupported("packed expert weight dtype"));
            }
            return Ok(DeviceWeight::Float { data });
        }
        let scale_base = base.strip_suffix(".weight").unwrap_or(&base);
        let token_input = activations::Rule::for_module(&self.activation_rules, scale_base)?;
        if let Some([block_rows, block_columns]) =
            activations::Rule::weight_block_for_module(&self.activation_rules, scale_base)?
        {
            if !rows.is_multiple_of(block_rows) || !columns.is_multiple_of(block_columns) {
                return Err(Error::invalid("expert FP8 block shape"));
            }
            let scale = self.source(&format!("{scale_base}.weight_scale_inv"))?;
            let scale = Self::expert_view(
                scale,
                expert,
                part * rows / block_rows,
                rows / block_rows,
                width / block_rows,
                columns / block_columns,
            )?;
            return Ok(DeviceWeight::Fp8Block {
                data,
                scale,
                block: [block_rows, block_columns],
                token_input,
            });
        }
        let scale = self.source(&format!("{scale_base}.weight_scale"))?;
        let scale = Self::expert_view(scale, expert, part * rows, *rows, width, 1)?;
        Ok(DeviceWeight::Fp8Channel {
            data,
            scale,
            token_input,
        })
    }

    fn bind_linear_view(&self, name: &str, shape: &[usize]) -> Result<Option<DeviceWeight>> {
        let Some((prefix, slot)) = name.rsplit_once(".linear_attn.") else {
            return Ok(None);
        };
        let Some(layer) = prefix
            .rsplit('.')
            .next()
            .and_then(|s| s.parse::<usize>().ok())
        else {
            return Ok(None);
        };
        let Some(infer_ir::Mixer::LinearAttention {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            ..
        }) = self.imported.model.mixers.get(layer)
        else {
            return Ok(None);
        };
        let ratio = value_heads / key_heads;
        let group = 2 * key_dim + 2 * ratio * value_dim;
        let (bank, indices): (&str, Vec<usize>) = match slot {
            "in_proj_qkv.weight" => (
                "in_proj_qkvz.weight",
                (0..*key_heads)
                    .flat_map(|h| (0..*key_dim).map(move |d| h * group + d))
                    .chain(
                        (0..*key_heads)
                            .flat_map(|h| (0..*key_dim).map(move |d| h * group + key_dim + d)),
                    )
                    .chain((0..*key_heads).flat_map(|h| {
                        (0..ratio * value_dim).map(move |d| h * group + 2 * key_dim + d)
                    }))
                    .collect(),
            ),
            "in_proj_z.weight" => (
                "in_proj_qkvz.weight",
                (0..*key_heads)
                    .flat_map(|h| {
                        (0..ratio * value_dim)
                            .map(move |d| h * group + 2 * key_dim + ratio * value_dim + d)
                    })
                    .collect(),
            ),
            "in_proj_b.weight" => (
                "in_proj_ba.weight",
                (0..*key_heads)
                    .flat_map(|h| (0..ratio).map(move |d| h * 2 * ratio + d))
                    .collect(),
            ),
            "in_proj_a.weight" => (
                "in_proj_ba.weight",
                (0..*key_heads)
                    .flat_map(|h| (0..ratio).map(move |d| h * 2 * ratio + ratio + d))
                    .collect(),
            ),
            _ => return Ok(None),
        };
        let fused = format!("{prefix}.linear_attn.{bank}");
        if !self.tensor_map.contains_key(&fused) {
            return Ok(None);
        }
        let mut data = self.source(&fused)?;
        let [stored_rows, columns] = data.shape.as_slice() else {
            return Err(Error::invalid("fused linear projection rank"));
        };
        if shape != [indices.len(), *columns]
            || indices.iter().any(|row| *row >= *stored_rows)
            || !data.dtype.is_host_float()
        {
            return Err(Error::unsupported(
                "fused linear projection shape or quantization",
            ));
        }
        let bytes = (indices.len() as u64)
            .checked_mul(*columns as u64)
            .and_then(|n| n.checked_mul(data.dtype.bytes()))
            .ok_or_else(|| Error::invalid("fused projection size overflow"))?;
        data.view = Some(WeightView {
            stored_shape: data.shape.clone(),
            element_offset: 0,
            rows: indices,
            columns: None,
        });
        data.shape = shape.to_vec();
        data.bytes = bytes;
        Ok(Some(DeviceWeight::Float { data }))
    }

    fn expert_view(
        mut source: WeightSource,
        expert: usize,
        start_row: usize,
        rows: usize,
        stored_rows: usize,
        columns: usize,
    ) -> Result<WeightSource> {
        let [experts, actual_rows, actual_columns] = source.shape.as_slice() else {
            return Err(Error::invalid("packed experts must be rank three"));
        };
        if expert >= *experts
            || *actual_rows != stored_rows
            || *actual_columns != columns
            || start_row
                .checked_add(rows)
                .is_none_or(|end| end > stored_rows)
        {
            return Err(Error::invalid("packed expert shape"));
        }
        let offset = expert
            .checked_mul(stored_rows)
            .and_then(|n| n.checked_add(start_row))
            .and_then(|n| n.checked_mul(columns))
            .ok_or_else(|| Error::invalid("expert view offset overflow"))?;
        let bytes = rows
            .checked_mul(columns)
            .and_then(|n| (n as u64).checked_mul(source.dtype.bytes()))
            .ok_or_else(|| Error::invalid("expert view size overflow"))?;
        source.view = Some(WeightView {
            stored_shape: source.shape.clone(),
            element_offset: offset as u64,
            rows: Vec::new(),
            columns: None,
        });
        source.shape = vec![rows, columns];
        source.bytes = bytes;
        Ok(source)
    }

    fn bind(&self, name: &str, shape: &[usize]) -> Result<DeviceWeight> {
        let base = name.strip_suffix(".weight").unwrap_or(name);
        if self
            .tensor_map
            .contains_key(&format!("{base}.weight_packed"))
        {
            return self.bind_nvfp4(base, shape);
        }
        let data = self.source(name)?;
        // Block scales exist only where the checkpoint stores FP8 bytes: a group target like
        // `Linear` also names embeddings and norms, which stay in their own dtype.
        if data.dtype != TensorDtype::F8E4m3 {
            if !data.dtype.is_host_float() || data.shape != shape {
                return Err(Error::invalid(format!(
                    "{name}: floating weight dtype/shape mismatch"
                )));
            }
            return Ok(DeviceWeight::Float { data });
        }
        let token_input = activations::Rule::for_module(&self.activation_rules, base)?;
        if let Some(block) =
            activations::Rule::weight_block_for_module(&self.activation_rules, base)?
        {
            return self.bind_fp8_block(base, data, shape, block, token_input);
        }
        let [rows, _columns] = shape else {
            return Err(Error::invalid("quantized weights must be matrices"));
        };
        let scale = self.source(&format!("{base}.weight_scale"))?;
        if data.shape != shape || scale.shape != [*rows, 1] || !scale.dtype.is_host_float() {
            return Err(Error::invalid(format!(
                "{base}: invalid per-channel FP8 binding"
            )));
        }
        Ok(DeviceWeight::Fp8Channel {
            data,
            scale,
            token_input,
        })
    }

    fn bind_fp8_block(
        &self,
        base: &str,
        data: WeightSource,
        shape: &[usize],
        block: [usize; 2],
        token_input: bool,
    ) -> Result<DeviceWeight> {
        let scale = self.source(&format!("{base}.weight_scale_inv"))?;
        validate_fp8_block(base, shape, block, &data, &scale)?;
        Ok(DeviceWeight::Fp8Block {
            data,
            scale,
            block,
            token_input,
        })
    }

    fn bind_nvfp4(&self, base: &str, shape: &[usize]) -> Result<DeviceWeight> {
        if shape.len() != 2 {
            return Err(Error::invalid("quantized weights must be matrices"));
        }
        let data = self.source(&format!("{base}.weight_packed"))?;
        let scale = self.source(&format!("{base}.weight_scale"))?;
        validate_nvfp4(base, shape, &data, &scale)?;
        let global = self.scalar_source(base, "weight_global_scale")?;
        let input_global = self.scalar_source(base, "input_global_scale")?;
        Ok(DeviceWeight::Nvfp4 {
            data,
            scale,
            global,
            input_global,
        })
    }

    fn scalar_source(&self, base: &str, suffix: &str) -> Result<WeightSource> {
        let source = self.source(&format!("{base}.{suffix}"))?;
        if source.dtype != TensorDtype::F32 || source.shape != [1] {
            return Err(Error::invalid(format!("{base}: invalid global scale")));
        }
        Ok(source)
    }

    fn bind_mtp(&mut self) -> Result<()> {
        let Some(prefix) = self
            .imported
            .speculation
            .as_ref()
            .map(|plan| plan.prefix.clone())
            .filter(|prefix| !prefix.is_empty())
        else {
            return Ok(());
        };
        let names: Vec<_> = self
            .tensor_map
            .keys()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect();
        // MTP is optional at load time; enabling it requires the complete head in the executor.
        for name in names {
            if name.ends_with(".weight") {
                let shape = self.source(&name)?.shape;
                self.mtp.insert(name.clone(), self.bind(&name, &shape)?);
            }
        }
        Ok(())
    }
}

/// Check one block-scaled FP8 binding before its sources become a device weight.
fn validate_fp8_block(
    base: &str,
    shape: &[usize],
    block: [usize; 2],
    data: &WeightSource,
    scale: &WeightSource,
) -> Result<()> {
    let [rows, columns] = shape else {
        return Err(Error::invalid("quantized weights must be matrices"));
    };
    let [block_rows, block_columns] = block;
    if data.shape != shape
        || block_rows == 0
        || block_columns == 0
        || !rows.is_multiple_of(block_rows)
        || !columns.is_multiple_of(block_columns)
        || scale.shape != [rows / block_rows, columns / block_columns]
        || !scale.dtype.is_host_float()
    {
        return Err(Error::invalid(format!(
            "{base}: invalid block-scaled FP8 binding"
        )));
    }
    Ok(())
}

/// Check one NVFP4 binding against its packed payload and block-16 scales.
fn validate_nvfp4(
    base: &str,
    shape: &[usize],
    data: &WeightSource,
    scale: &WeightSource,
) -> Result<()> {
    let [rows, columns] = shape else {
        return Err(Error::invalid("quantized weights must be matrices"));
    };
    if !columns.is_multiple_of(NVFP4_GROUP_SIZE)
        || data.dtype != TensorDtype::U8
        || data.shape != [*rows, columns / 2]
        || scale.dtype != TensorDtype::F8E4m3
        || scale.shape != [*rows, columns / NVFP4_GROUP_SIZE]
    {
        return Err(Error::invalid(format!(
            "{base}: invalid NVFP4 block-16 binding"
        )));
    }
    Ok(())
}

/// Translate vLLM's native `fp8` scheme into the compressed-tensors shape the loader reads.
///
/// `activation_scheme: "dynamic"` with per-`weight_block_size` block scales is the `DeepSeek`
/// convention: one scale per 128x128 weight block and one per 128-element activation block.
/// The `ignored_layers` list needs no translation because an ignored module stores bf16 weights
/// and the encoding is derived from the stored dtype.
fn normalize_fp8_scheme(quant: &serde_json::Value) -> Result<serde_json::Value> {
    if quant["fmt"] != "e4m3" || quant["activation_scheme"] != "dynamic" {
        return Err(Error::unsupported("unsupported FP8 checkpoint scheme"));
    }
    let block = quant["weight_block_size"]
        .as_array()
        .and_then(|sizes| {
            sizes
                .iter()
                .map(|size| size.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![FP8_WEIGHT_BITS, FP8_WEIGHT_BITS]);
    let [block_rows, block_columns] = block.as_slice() else {
        return Err(Error::invalid(
            "FP8 weight block size must be two-dimensional",
        ));
    };
    let strategy = if *block_rows == 1 || *block_columns == 1 {
        "channel"
    } else {
        "block"
    };
    let mut weights = serde_json::json!({
        "type": "float",
        "symmetric": true,
        "dynamic": false,
        "num_bits": FP8_WEIGHT_BITS,
        "strategy": strategy,
    });
    let mut activations = serde_json::json!({
        "type": "float",
        "num_bits": FP8_WEIGHT_BITS,
        "strategy": "token",
        "dynamic": true,
        "symmetric": true,
    });
    if strategy == "block" {
        weights["block_size"] = serde_json::json!([block_rows, block_columns]);
        activations["block_size"] = serde_json::json!([1, block_columns]);
        activations["strategy"] = serde_json::json!("block");
    }
    Ok(serde_json::json!({
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "weights": weights,
                "input_activations": activations,
                "targets": ["Linear"],
            }
        },
    }))
}

fn validate_groups(quant: &serde_json::Value) -> Result<()> {
    let groups = quant["config_groups"]
        .as_object()
        .ok_or_else(|| Error::invalid("missing quantization groups"))?;
    if groups.is_empty() {
        return Err(Error::invalid("empty quantization groups"));
    }
    for group in groups.values() {
        let weights = &group["weights"];
        let valid = weights["type"] == "float"
            && weights["symmetric"] == true
            && weights["dynamic"] == false
            && ((group["format"] == "float-quantized"
                && weights["num_bits"] == FP8_WEIGHT_BITS
                && matches!(weights["strategy"].as_str(), Some("channel" | "block")))
                || (group["format"] == "nvfp4-pack-quantized"
                    && weights["num_bits"] == NVFP4_WEIGHT_BITS
                    && weights["strategy"] == "tensor_group"
                    && weights["group_size"] == NVFP4_GROUP_SIZE));
        if !valid {
            return Err(Error::new(
                ErrorCode::Unsupported,
                "unsupported mixed-precision quantization group",
            ));
        }
    }
    Ok(())
}

fn declared_kv_dtype(scheme: &serde_json::Value) -> Result<Option<TensorDtype>> {
    if scheme.is_null() {
        return Ok(None);
    }
    if scheme["num_bits"] == FP8_WEIGHT_BITS
        && scheme["type"] == "float"
        && scheme["strategy"] == "tensor"
        && scheme["dynamic"] == false
        && scheme["symmetric"] == true
    {
        Ok(Some(TensorDtype::F8E4m3))
    } else {
        Err(Error::unsupported(
            "unsupported checkpoint KV quantization scheme",
        ))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage_quantized.rs"]
mod kv_tests;
