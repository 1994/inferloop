//! Bind a `DFlash2` draft package against the target package it speculates for.
use super::{DeviceWeight, QuantizedPackage, WeightEncoding, WeightSource, WeightView};
use crate::constants::CONFIG_MAX_BYTES;
use crate::providers::dflash2::DFlash2Config;
use crate::storage::package::{load_shards, read_bounded};
use crate::{TensorDtype, package_path};
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
        let raw = read_bounded(&package_path(root, "config.json")?, CONFIG_MAX_BYTES)?;
        let config = DFlash2Config::parse(&raw)?;
        let value: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| Error::invalid(e.to_string()))?;
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
        let block = fp8_block(&value["quantization_config"])?;
        let (shards, tensor_map) = load_shards(root)?;
        let mut sources = BTreeMap::new();
        for (name, shard) in &tensor_map {
            let header = &shards[shard].tensors[name];
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
        let inventory = weights
            .iter()
            .map(|(name, weight)| (name.clone(), weight.shape.clone()))
            .collect();
        config.check_weight_inventory(&inventory)?;
        let hidden = config.hidden_size;
        let fc = weights
            .get("fc.weight")
            .ok_or_else(|| Error::invalid("draft package is missing fc.weight"))?;
        for (index, slice) in column_blocks(fc, hidden, geometry.target_taps.len())?
            .into_iter()
            .enumerate()
        {
            self.weights.insert(tap_projection_slot(index), slice);
        }
        attach_tap_projection(&mut self.graph, &geometry.target_taps, hidden)?;
        self.dflash2 = Some(DFlash2Package { config, weights });
        Ok(())
    }
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
    let mut weight = DeviceWeight {
        shape: data.shape.clone(),
        encoding: WeightEncoding::Float,
        data: data.clone(),
        scale: None,
        global_scale: None,
        input_global_scale: None,
        weight_block: None,
        fp8_token_input: false,
    };
    if data.dtype != TensorDtype::F8E4m3 {
        if !data.dtype.is_host_float() {
            return Err(Error::unsupported(format!("{name}: draft weight dtype")));
        }
        return Ok(weight);
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let [rows, columns] = data.shape.as_slice() else {
        return Err(Error::invalid(format!("{name}: FP8 draft weight rank")));
    };
    if let Some(scale) = sources.get(&format!("{base}.weight_scale")) {
        if scale.shape != [*rows, 1] || !scale.dtype.is_host_float() {
            return Err(Error::invalid(format!("{base}: per-channel FP8 scale")));
        }
        weight.encoding = WeightEncoding::Fp8Channel;
        weight.scale = Some(scale.clone());
    } else if let Some(scale) = sources.get(&format!("{base}.weight_scale_inv")) {
        let [block_rows, block_columns] = block.unwrap_or(DEFAULT_FP8_BLOCK);
        if !rows.is_multiple_of(block_rows)
            || !columns.is_multiple_of(block_columns)
            || scale.shape != [rows / block_rows, columns / block_columns]
            || !scale.dtype.is_host_float()
        {
            return Err(Error::invalid(format!("{base}: block FP8 scale")));
        }
        weight.encoding = WeightEncoding::Fp8Block;
        weight.weight_block = Some([block_rows, block_columns]);
        weight.scale = Some(scale.clone());
    } else {
        return Err(Error::invalid(format!(
            "{base}: FP8 draft weight has no scale"
        )));
    }
    // The draft's projections are dynamically token-quantized in the reference.
    weight.fp8_token_input = true;
    Ok(weight)
}

/// Split `[hidden, parts * hidden]` into `parts` column blocks of `[hidden, hidden]`.
fn column_blocks(weight: &DeviceWeight, hidden: usize, parts: usize) -> Result<Vec<DeviceWeight>> {
    if weight.shape != [hidden, parts * hidden] {
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
        .map(|part| {
            let scale = match (weight.encoding, &weight.scale) {
                (WeightEncoding::Float, _) => None,
                (WeightEncoding::Fp8Channel, Some(scale)) => Some(scale.clone()),
                (WeightEncoding::Fp8Block, Some(scale)) => {
                    let [_, block_columns] = weight
                        .weight_block
                        .ok_or_else(|| Error::invalid("draft fc FP8 block"))?;
                    if !hidden.is_multiple_of(block_columns) {
                        return Err(Error::invalid("draft fc FP8 block columns"));
                    }
                    let blocks = hidden / block_columns;
                    Some(view(scale, part * blocks, blocks)?)
                }
                _ => return Err(Error::unsupported("draft fc encoding")),
            };
            Ok(DeviceWeight {
                shape: vec![hidden, hidden],
                data: view(&weight.data, part * hidden, hidden)?,
                scale,
                ..weight.clone()
            })
        })
        .collect()
}
