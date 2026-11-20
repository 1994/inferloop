//! Bind a `DFlash2` draft package against the target package it speculates for.
use super::{DeviceWeight, QuantizedPackage, WeightSource, WeightView};
use crate::providers::dflash2::DFlash2Config;
use crate::storage::source::TensorFiles;
use crate::{PackageSource, TensorDtype};
use infer_core::{Error, Result};
use infer_model_recipes::features::{attach_tap_projection, tap_projection_slot};
use std::collections::BTreeMap;
use std::path::Path;

/// Shard-key prefix that keeps draft shards apart from the target's own.
const SHARD_PREFIX: &str = "draft/";
/// Weight block a vLLM `fp8` draft declares when its configuration omits `weight_block_size`.
const DEFAULT_FP8_BLOCK: [usize; 2] = [128, 128];
/// Tensor-name suffixes that carry scales or calibration, never weights of their own.
const SCALE_SUFFIXES: [&str; 5] = [
    ".weight_scale",
    ".weight_scale_inv",
    ".input_scale",
    ".k_scale",
    ".v_scale",
];

/// A `DFlash2` draft bound to its target: configuration plus every draft weight, by the
/// checkpoint's own tensor name.
pub struct DFlash2Package {
    pub config: DFlash2Config,
    pub weights: BTreeMap<String, DeviceWeight>,
}

impl QuantizedPackage {
    /// Attach the `DFlash2` draft stored at `root`.
    ///
    /// The draft's shards join this package under their own keys, so one [`Self::read`] serves
    /// target and draft weights. The target graph gains the fused tap feature
    /// ([`attach_tap_projection`]): the draft's `fc` is split into one `[hidden, hidden]` column
    /// block per tap and bound to the target's graph as [`tap_projection_slot`] weights.
    ///
    /// # Errors
    /// Rejects a configuration that is not a compatible `DFlash2` draft for this target, an
    /// inventory that disagrees with the draft geometry, unsupported quantization, or a target
    /// graph whose tapped layers do not end in a residual.
    pub fn attach_dflash2(&mut self, root: impl AsRef<Path>) -> Result<()> {
        if self.dflash2.is_some() {
            return Err(Error::invalid("a DFlash2 draft is already attached"));
        }
        let root = root.as_ref();
        if root
            .extension()
            .is_some_and(|extension| extension == "gguf")
        {
            #[cfg(feature = "gguf")]
            return self.attach_dflash2_gguf(root);
            #[cfg(not(feature = "gguf"))]
            return Err(Error::unsupported(
                "GGUF DFlash2 drafts require the gguf feature",
            ));
        }
        let source = PackageSource::open(root)?;
        let raw = source.require_config()?;
        let config = DFlash2Config::parse(raw)?;
        let value: serde_json::Value =
            serde_json::from_slice(raw).map_err(|e| Error::invalid(e.to_string()))?;
        self.check_draft(&config)?;
        let block = fp8_block(&value["quantization_config"])?;
        let TensorFiles { shards, tensor_map } = source.tensor_files()?;
        let mut sources = BTreeMap::new();
        for (name, shard) in &tensor_map {
            let header = &shards[shard].tensors()[name];
            sources.insert(
                name.clone(),
                WeightSource {
                    name: name.clone(),
                    shard: format!("{SHARD_PREFIX}{shard}"),
                    dtype: header.dtype,
                    shape: header.shape.clone(),
                    bytes: header.byte_len(),
                    view: None,
                },
            );
        }
        for (shard, file) in shards {
            self.shards.insert(format!("{SHARD_PREFIX}{shard}"), file);
        }
        let mut weights = BTreeMap::new();
        for (name, source) in &sources {
            if SCALE_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
                continue;
            }
            weights.insert(name.clone(), bind(name, source, &sources, block)?);
        }
        let fc = weights
            .get("fc.weight")
            .ok_or_else(|| Error::invalid("draft package is missing fc.weight"))?;
        let taps = column_blocks(fc, config.hidden_size, config.geometry().target_taps.len())?;
        self.bind_draft(config, weights, taps)
    }

    /// Check that `config` is a `DFlash2` draft this target can carry.
    fn check_draft(&self, config: &DFlash2Config) -> Result<()> {
        config.check_block_semantics()?;
        config.check_special_tokens()?;
        let geometry = config.geometry();
        geometry.validate()?;
        geometry.check_target(&self.imported.model)?;
        if config.vocab_size != self.imported.model.vocab_size {
            return Err(Error::invalid(format!(
                "draft vocabulary {} does not match the target's {}",
                config.vocab_size, self.imported.model.vocab_size
            )));
        }
        Ok(())
    }

    /// Check the draft inventory, bind one `fc` column block per tap into the target graph and
    /// keep the draft.
    fn bind_draft(
        &mut self,
        config: DFlash2Config,
        weights: BTreeMap<String, DeviceWeight>,
        taps: Vec<DeviceWeight>,
    ) -> Result<()> {
        let inventory = weights
            .iter()
            .map(|(name, weight)| (name.clone(), weight.shape()))
            .collect();
        config.check_weight_inventory(&inventory)?;
        let geometry = config.geometry();
        for (index, slice) in taps.into_iter().enumerate() {
            self.weights.insert(tap_projection_slot(index), slice);
        }
        attach_tap_projection(&mut self.graph, &geometry.target_taps, config.hidden_size)?;
        self.dflash2 = Some(DFlash2Package { config, weights });
        Ok(())
    }

    /// Attach a `DFlash2` draft stored as a `dflash` GGUF file.
    ///
    /// The file does not record the target's depth, so the target being attached supplies
    /// `num_target_layers`. Its float tensors stay in host memory as the GGUF target's do, and
    /// each `fc` tap block is cut out once here, since GGUF bytes have no gathered views.
    #[cfg(feature = "gguf")]
    fn attach_dflash2_gguf(&mut self, path: &Path) -> Result<()> {
        let draft = crate::GgufPackage::open(path)?;
        if draft.architecture() != GGUF_ARCHITECTURE {
            return Err(Error::invalid(format!(
                "{} is a {} GGUF, not a DFlash2 draft",
                path.display(),
                draft.architecture()
            )));
        }
        let mut value: serde_json::Value = serde_json::from_slice(draft.config())
            .map_err(|error| Error::invalid(error.to_string()))?;
        value["num_target_layers"] = self.imported.model.mixers.len().into();
        let config = DFlash2Config::parse(
            &serde_json::to_vec(&value).map_err(|error| Error::invalid(error.to_string()))?,
        )?;
        self.check_draft(&config)?;
        let mut weights = BTreeMap::new();
        for slot in draft.slots().values() {
            let data = dense_source(&format!("{SHARD_PREFIX}{}", slot.slot), slot)?;
            self.gguf_dense
                .insert(data.name.clone(), draft.float_bytes(&slot.slot, u64::MAX)?);
            weights.insert(slot.slot.clone(), DeviceWeight::Float { data });
        }
        let hidden = config.hidden_size;
        let fc = weights
            .get("fc.weight")
            .ok_or_else(|| Error::invalid("draft package is missing fc.weight"))?
            .data()
            .clone();
        let parts = config.geometry().target_taps.len();
        let blocks = self.cut_columns(&fc, hidden, parts)?;
        self.bind_draft(config, weights, blocks)
    }

    /// Cut `[rows, parts * hidden]` dense GGUF bytes into `parts` cached `[rows, hidden]` blocks.
    #[cfg(feature = "gguf")]
    fn cut_columns(
        &mut self,
        fc: &WeightSource,
        hidden: usize,
        parts: usize,
    ) -> Result<Vec<DeviceWeight>> {
        let [rows, columns] = fc.shape.as_slice() else {
            return Err(Error::invalid("draft fc rank"));
        };
        if Some(*columns) != parts.checked_mul(hidden) {
            return Err(Error::invalid("draft fc shape"));
        }
        let element =
            usize::try_from(fc.dtype.bytes()).map_err(|e| Error::invalid(e.to_string()))?;
        let stored = self
            .gguf_dense
            .get(&fc.name)
            .ok_or_else(|| Error::invariant("draft fc is not cached"))?;
        let (row_bytes, kept) = (columns * element, hidden * element);
        let mut cut = Vec::with_capacity(parts);
        for part in 0..parts {
            let mut bytes = Vec::with_capacity(rows * kept);
            for row in stored.chunks_exact(row_bytes) {
                bytes.extend_from_slice(&row[part * kept..(part + 1) * kept]);
            }
            let data = WeightSource {
                name: format!("{}#{part}", fc.name),
                shape: vec![*rows, hidden],
                bytes: bytes.len() as u64,
                ..fc.clone()
            };
            cut.push((data, bytes));
        }
        Ok(cut
            .into_iter()
            .map(|(data, bytes)| {
                self.gguf_dense.insert(data.name.clone(), bytes);
                DeviceWeight::Float { data }
            })
            .collect())
    }
}

/// The `general.architecture` of a `DFlash2` GGUF draft.
#[cfg(feature = "gguf")]
const GGUF_ARCHITECTURE: &str = "dflash";

/// A cached dense GGUF float tensor named `name`; quantized draft weights are rejected, as the
/// draft's CUDA projections take float or FP8 weights.
#[cfg(feature = "gguf")]
fn dense_source(name: &str, slot: &crate::GgufSlot) -> Result<WeightSource> {
    let dtype = match slot.float_type() {
        infer_format::GgufType::F32 => TensorDtype::F32,
        infer_format::GgufType::F16 => TensorDtype::F16,
        infer_format::GgufType::Bf16 => TensorDtype::BF16,
        other => {
            return Err(Error::unsupported(format!(
                "DFlash2 GGUF draft weight {} is {}; the draft takes float weights",
                slot.slot,
                other.name()
            )));
        }
    };
    let elements = slot
        .shape
        .iter()
        .try_fold(1u64, |count, dimension| {
            count.checked_mul(*dimension as u64)
        })
        .ok_or_else(|| Error::invalid("GGUF draft shape overflow"))?;
    Ok(WeightSource {
        name: name.to_owned(),
        // Dense GGUF bytes are served from the cache the `gguf` shard reads.
        shard: "gguf".to_owned(),
        dtype,
        shape: slot.shape.clone(),
        bytes: elements
            .checked_mul(dtype.bytes())
            .ok_or_else(|| Error::invalid("GGUF draft size overflow"))?,
        view: None,
    })
}

/// Block geometry a block-scaled FP8 draft declares, if any.
fn fp8_block(quant: &serde_json::Value) -> Result<Option<[usize; 2]>> {
    if quant.is_null() {
        return Ok(None);
    }
    match quant["quant_method"].as_str() {
        Some("fp8") => {
            let Some(sizes) = quant["weight_block_size"].as_array() else {
                return Ok(Some(DEFAULT_FP8_BLOCK));
            };
            let sizes: Vec<_> = sizes
                .iter()
                .filter_map(|size| size.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect();
            match sizes.as_slice() {
                [rows, columns] if *rows > 0 && *columns > 0 => Ok(Some([*rows, *columns])),
                _ => Err(Error::invalid(
                    "draft FP8 weight block must be two-dimensional",
                )),
            }
        }
        Some("compressed-tensors") => Ok(None),
        _ => Err(Error::unsupported(
            "DFlash2 drafts support BF16, compressed-tensors FP8 or block FP8 weights",
        )),
    }
}

fn bind(
    name: &str,
    data: &WeightSource,
    sources: &BTreeMap<String, WeightSource>,
    block: Option<[usize; 2]>,
) -> Result<DeviceWeight> {
    if data.dtype != TensorDtype::F8E4m3 {
        if !data.dtype.is_host_float() {
            return Err(Error::unsupported(format!("{name}: draft weight dtype")));
        }
        return Ok(DeviceWeight::Float { data: data.clone() });
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let [rows, columns] = data.shape.as_slice() else {
        return Err(Error::invalid(format!("{name}: FP8 draft weight rank")));
    };
    if let Some(scale) = sources.get(&format!("{base}.weight_scale")) {
        if scale.shape != [*rows, 1] || !scale.dtype.is_host_float() {
            return Err(Error::invalid(format!("{base}: per-channel FP8 scale")));
        }
        return Ok(DeviceWeight::Fp8Channel {
            data: data.clone(),
            scale: scale.clone(),
            // The draft's projections are dynamically token-quantized in the reference.
            token_input: true,
        });
    }
    if let Some(scale) = sources.get(&format!("{base}.weight_scale_inv")) {
        let block = block.unwrap_or(DEFAULT_FP8_BLOCK);
        let [block_rows, block_columns] = block;
        if !rows.is_multiple_of(block_rows)
            || !columns.is_multiple_of(block_columns)
            || scale.shape != [rows / block_rows, columns / block_columns]
            || !scale.dtype.is_host_float()
        {
            return Err(Error::invalid(format!("{base}: block FP8 scale")));
        }
        return Ok(DeviceWeight::Fp8Block {
            data: data.clone(),
            scale: scale.clone(),
            block,
            token_input: true,
        });
    }
    Err(Error::invalid(format!(
        "{base}: FP8 draft weight has no scale"
    )))
}

/// Split `[hidden, parts * hidden]` into `parts` column blocks of `[hidden, hidden]`.
fn column_blocks(weight: &DeviceWeight, hidden: usize, parts: usize) -> Result<Vec<DeviceWeight>> {
    if weight.shape() != [hidden, parts * hidden] {
        return Err(Error::invalid("draft fc shape"));
    }
    let view = |source: &WeightSource, first: usize, count: usize| -> Result<WeightSource> {
        let [rows, _] = source.shape.as_slice() else {
            return Err(Error::invalid("draft fc rank"));
        };
        let bytes = (*rows as u64)
            .checked_mul(count as u64)
            .and_then(|n| n.checked_mul(source.dtype.bytes()))
            .ok_or_else(|| Error::invalid("draft fc slice overflow"))?;
        Ok(WeightSource {
            shape: vec![*rows, count],
            bytes,
            view: Some(WeightView {
                stored_shape: source.shape.clone(),
                element_offset: 0,
                rows: Vec::new(),
                columns: Some([first, count]),
            }),
            ..source.clone()
        })
    };
    (0..parts)
        .map(|part| match weight {
            DeviceWeight::Float { data } => Ok(DeviceWeight::Float {
                data: view(data, part * hidden, hidden)?,
            }),
            DeviceWeight::Fp8Channel {
                data,
                scale,
                token_input,
            } => Ok(DeviceWeight::Fp8Channel {
                data: view(data, part * hidden, hidden)?,
                // A per-channel scale has one column, so a column slice leaves it unchanged.
                scale: scale.clone(),
                token_input: *token_input,
            }),
            DeviceWeight::Fp8Block {
                data,
                scale,
                block,
                token_input,
            } => {
                let [_, block_columns] = *block;
                if !hidden.is_multiple_of(block_columns) {
                    return Err(Error::invalid("draft fc FP8 block columns"));
                }
                let blocks = hidden / block_columns;
                Ok(DeviceWeight::Fp8Block {
                    data: view(data, part * hidden, hidden)?,
                    scale: view(scale, part * blocks, blocks)?,
                    block: *block,
                    token_input: *token_input,
                })
            }
            DeviceWeight::Nvfp4 { .. } => Err(Error::unsupported("draft fc encoding")),
            DeviceWeight::GroupedInt { .. } => Err(Error::unsupported("grouped-int draft weight")),
        })
        .collect()
}
