//! The `qwen3vl_merger` mmproj projector: the Qwen3-VL vision tower, its merger and deepstack
//! mergers.
use crate::storage::gguf::{
    ExpectedTensors, GgufProjector, TensorSpec, VisionTensor, bool_array, count, is_float,
};
use infer_core::{Error, Result};
use infer_format::{Metadata, MetadataValue};
use infer_spi::{HiddenActivation, ModalityEncoder};
use serde::Serialize;

/// RGB channels of the vision patch embedding.
pub const IN_CHANNELS: usize = 3;
/// Frames one `Conv3D` patch spans; llama.cpp stores one `Conv2D` slice per frame.
pub const TEMPORAL_PATCH: usize = 2;
/// Package prefix the vision inventory expects.
const VISION_PREFIX: &str = "model.visual.";
/// Axial vision `RoPE` theta used by the Qwen3-VL family.
const VISION_ROPE_THETA: f64 = 10_000.0;
/// Q, K and V projections packed into one vision tensor.
const QKV_WIDTH: usize = 3;
/// Deepstack merger count the text configuration expects.
const DEEPSTACK_COUNT: usize = 3;
/// Qwen image processor minimum pixels.
const MIN_PIXELS: u64 = 65_536;
/// Qwen image processor maximum pixels.
const MAX_PIXELS: u64 = 16_777_216;
/// Reciprocal of the 8-bit pixel range.
const RESCALE: f64 = 1.0 / 255.0;

/// The `qwen3vl_merger` projector.
pub struct Qwen3VlMerger;

/// The `qwen3vl_merger` adapter.
pub const QWEN3VL_MERGER: Qwen3VlMerger = Qwen3VlMerger;

/// The image-processor configuration an mmproj implies.
#[derive(Serialize)]
struct Preprocessor {
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    size: Edges,
    image_mean: [f64; IN_CHANNELS],
    image_std: [f64; IN_CHANNELS],
    rescale: f64,
}

#[derive(Serialize)]
struct Edges {
    shortest_edge: u64,
    longest_edge: u64,
}

/// Vision geometry shared by the block and deepstack inventory.
struct Geometry {
    hidden: usize,
    ffn: usize,
    merged: usize,
    out: usize,
}

impl GgufProjector for Qwen3VlMerger {
    fn name(&self) -> &'static str {
        "qwen3vl_merger"
    }

    fn validate(&self, metadata: &Metadata, tensors: &[TensorSpec]) -> Result<()> {
        let hidden = count(metadata, "clip.vision.embedding_length")?;
        let merge = count(metadata, "clip.vision.spatial_merge_size")?;
        let patch = count(metadata, "clip.vision.patch_size")?;
        let geometry = Geometry {
            hidden,
            ffn: count(metadata, "clip.vision.feed_forward_length")?,
            merged: hidden
                .checked_mul(merge)
                .and_then(|value| value.checked_mul(merge))
                .ok_or_else(|| Error::invalid("Qwen3-VL merged width overflow"))?,
            out: count(metadata, "clip.vision.projection_dim")?,
        };
        let mut expected = ExpectedTensors::new();
        for name in ["v.patch_embd.weight", "v.patch_embd.weight.1"] {
            expected.insert(name, vec![hidden, IN_CHANNELS, patch, patch])?;
        }
        expected.insert("v.patch_embd.bias", vec![hidden])?;
        expected.insert(
            "v.position_embd.weight",
            vec![position_grid(metadata)?, hidden],
        )?;
        for field in ["weight", "bias"] {
            expected.insert(format!("v.post_ln.{field}"), vec![hidden])?;
        }
        let Geometry { merged, out, .. } = geometry;
        expected.insert("mm.0.weight", vec![merged, merged])?;
        expected.insert("mm.0.bias", vec![merged])?;
        expected.insert("mm.2.weight", vec![out, merged])?;
        expected.insert("mm.2.bias", vec![out])?;
        for layer in 0..count(metadata, "clip.vision.block_count")? {
            insert_block(&mut expected, layer, &geometry)?;
        }
        let deepstack = deepstack_layers(metadata)?;
        if deepstack.len() != DEEPSTACK_COUNT {
            return Err(Error::invalid("Qwen3-VL deepstack layer count"));
        }
        for layer in deepstack {
            insert_deepstack(&mut expected, layer, &geometry)?;
        }
        expected.check(tensors, is_float)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the positive LayerNorm epsilon is well inside the f32 range"
    )]
    fn encoder(&self, mmproj: &Metadata) -> Result<ModalityEncoder> {
        Ok(ModalityEncoder {
            prefix: VISION_PREFIX.into(),
            depth: count(mmproj, "clip.vision.block_count")?,
            hidden_size: count(mmproj, "clip.vision.embedding_length")?,
            intermediate_size: count(mmproj, "clip.vision.feed_forward_length")?,
            heads: count(mmproj, "clip.vision.attention.head_count")?,
            in_channels: IN_CHANNELS,
            patch_size: count(mmproj, "clip.vision.patch_size")?,
            temporal_patch_size: TEMPORAL_PATCH,
            spatial_merge_size: count(mmproj, "clip.vision.spatial_merge_size")?,
            out_hidden_size: count(mmproj, "clip.vision.projection_dim")?,
            position_embeddings: position_grid(mmproj)?,
            rope_theta: VISION_ROPE_THETA,
            norm_epsilon: mmproj.require_f64("clip.vision.attention.layer_norm_epsilon")? as f32,
            hidden_activation: HiddenActivation::GeluTanh,
            deepstack_layers: deepstack_layers(mmproj)?,
        })
    }

    fn tensor(&self, name: &str, encoder: &ModalityEncoder) -> Option<VisionTensor> {
        vision_tensor(name, &encoder.deepstack_layers)
    }

    fn preprocessor(&self, metadata: &Metadata, encoder: &ModalityEncoder) -> Result<Vec<u8>> {
        let config = Preprocessor {
            patch_size: encoder.patch_size,
            temporal_patch_size: encoder.temporal_patch_size,
            merge_size: encoder.spatial_merge_size,
            size: Edges {
                shortest_edge: MIN_PIXELS,
                longest_edge: MAX_PIXELS,
            },
            image_mean: normalization(metadata, "clip.vision.image_mean")?,
            image_std: normalization(metadata, "clip.vision.image_std")?,
            rescale: RESCALE,
        };
        serde_json::to_vec(&config).map_err(|error| Error::invalid(error.to_string()))
    }
}

/// Learned position embeddings: one per patch of the square training image.
/// # Errors
/// Rejects an image size that is not a whole number of patches.
pub fn position_grid(mmproj: &Metadata) -> Result<usize> {
    let image = count(mmproj, "clip.vision.image_size")?;
    let patch = count(mmproj, "clip.vision.patch_size")?;
    if patch == 0 || !image.is_multiple_of(patch) {
        return Err(Error::invalid(
            "Qwen3-VL image size must divide by patch size",
        ));
    }
    let side = image / patch;
    side.checked_mul(side)
        .ok_or_else(|| Error::invalid("Qwen3-VL position grid overflow"))
}

/// The vision layers whose output feeds a deepstack merger.
/// # Errors
/// Rejects a missing or malformed layer mask.
pub fn deepstack_layers(mmproj: &Metadata) -> Result<Vec<usize>> {
    Ok(bool_array(mmproj, "clip.vision.is_deepstack_layers")?
        .into_iter()
        .enumerate()
        .filter_map(|(index, enabled)| enabled.then_some(index))
        .collect())
}

fn insert_block(expected: &mut ExpectedTensors, layer: usize, geometry: &Geometry) -> Result<()> {
    let Geometry { hidden, ffn, .. } = *geometry;
    let qkv = hidden
        .checked_mul(QKV_WIDTH)
        .ok_or_else(|| Error::invalid("Qwen3-VL vision qkv width overflow"))?;
    for (field, shape) in [
        ("attn_qkv.weight", vec![qkv, hidden]),
        ("attn_qkv.bias", vec![qkv]),
        ("attn_out.weight", vec![hidden, hidden]),
        ("attn_out.bias", vec![hidden]),
        ("ffn_up.weight", vec![ffn, hidden]),
        ("ffn_up.bias", vec![ffn]),
        ("ffn_down.weight", vec![hidden, ffn]),
        ("ffn_down.bias", vec![hidden]),
        ("ln1.weight", vec![hidden]),
        ("ln1.bias", vec![hidden]),
        ("ln2.weight", vec![hidden]),
        ("ln2.bias", vec![hidden]),
    ] {
        expected.insert(format!("v.blk.{layer}.{field}"), shape)?;
    }
    Ok(())
}

fn insert_deepstack(
    expected: &mut ExpectedTensors,
    layer: usize,
    geometry: &Geometry,
) -> Result<()> {
    let Geometry { merged, out, .. } = *geometry;
    for (field, shape) in [
        ("norm.weight", vec![merged]),
        ("norm.bias", vec![merged]),
        ("fc1.weight", vec![merged, merged]),
        ("fc1.bias", vec![merged]),
        ("fc2.weight", vec![out, merged]),
        ("fc2.bias", vec![out]),
    ] {
        expected.insert(format!("v.deepstack.{layer}.{field}"), shape)?;
    }
    Ok(())
}

fn normalization(metadata: &Metadata, key: &str) -> Result<[f64; IN_CHANNELS]> {
    let values = metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or_else(|| Error::invalid(format!("missing {key}")))?;
    if values.len() != IN_CHANNELS {
        return Err(Error::invalid(format!("{key} width")));
    }
    let mut out = [0.0f64; IN_CHANNELS];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value
            .as_f64()
            .ok_or_else(|| Error::invalid(format!("{key} value")))?;
    }
    Ok(out)
}

/// Map one mmproj tensor to the HF vision inventory, or a `Conv3D` patch slice.
fn vision_tensor(name: &str, deepstack_layers: &[usize]) -> Option<VisionTensor> {
    let slot = |name: String| Some(VisionTensor::Slot(format!("{VISION_PREFIX}{name}")));
    match name {
        "v.patch_embd.weight" => return Some(VisionTensor::PatchSlice(0)),
        "v.patch_embd.weight.1" => return Some(VisionTensor::PatchSlice(1)),
        "v.patch_embd.bias" => return slot("patch_embed.proj.bias".to_owned()),
        "v.position_embd.weight" => return slot("pos_embed.weight".to_owned()),
        _ => {}
    }
    if let Some(field) = name.strip_prefix("v.post_ln.") {
        return slot(format!("merger.norm.{field}"));
    }
    if let Some(rest) = name.strip_prefix("v.deepstack.") {
        let (layer, field) = rest.split_once('.')?;
        let layer: usize = layer.parse().ok()?;
        let index = deepstack_layers.iter().position(|known| *known == layer)?;
        let field = match field {
            "norm.weight" => "norm.weight",
            "norm.bias" => "norm.bias",
            "fc1.weight" => "linear_fc1.weight",
            "fc1.bias" => "linear_fc1.bias",
            "fc2.weight" => "linear_fc2.weight",
            "fc2.bias" => "linear_fc2.bias",
            _ => return None,
        };
        return slot(format!("deepstack_merger_list.{index}.{field}"));
    }
    if let Some(rest) = name.strip_prefix("mm.") {
        let (index, field) = rest.split_once('.')?;
        let target = match index {
            "0" => "merger.linear_fc1",
            "2" => "merger.linear_fc2",
            _ => return None,
        };
        return slot(format!("{target}.{field}"));
    }
    let (layer, field) = name.strip_prefix("v.blk.")?.split_once('.')?;
    let field = match field {
        "attn_qkv.weight" => "attn.qkv.weight",
        "attn_qkv.bias" => "attn.qkv.bias",
        "attn_out.weight" => "attn.proj.weight",
        "attn_out.bias" => "attn.proj.bias",
        "ffn_up.weight" => "mlp.linear_fc1.weight",
        "ffn_up.bias" => "mlp.linear_fc1.bias",
        "ffn_down.weight" => "mlp.linear_fc2.weight",
        "ffn_down.bias" => "mlp.linear_fc2.bias",
        "ln1.weight" => "norm1.weight",
        "ln1.bias" => "norm1.bias",
        "ln2.weight" => "norm2.weight",
        "ln2.bias" => "norm2.bias",
        _ => return None,
    };
    slot(format!("blocks.{layer}.{field}"))
}
