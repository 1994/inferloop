//! Shape and storage checks for a Qwen3-VL GGUF before a package is opened for weights.
use infer_core::{Error, Result};
use infer_gguf::{GgufType, Metadata};
use std::collections::BTreeMap;

/// Q, K and V projections packed into one vision tensor.
const QKV_WIDTH: usize = 3;
/// Two temporal slices replace the HF `Conv3D` patch weight.
const PATCH_SLICES: usize = 2;
/// Deepstack merger count expected by the text configuration.
const DEEPSTACK_COUNT: u64 = 3;

/// Vision geometry shared by the block and deepstack inventory.
struct VisionGeometry {
    layers: usize,
    hidden: usize,
    ffn: usize,
    qkv: usize,
    merged: usize,
    out: usize,
}

/// One stored tensor, as the validator needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Stored GGUF name.
    pub name: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Storage type.
    pub ty: GgufType,
}

/// Validate every text tensor against the `qwen3vl` geometry.
/// # Errors
/// Rejects a missing or extra tensor, a shape mismatch, or an unsupported storage type.
pub fn validate_text(metadata: &Metadata, tensors: &[TensorSpec]) -> Result<()> {
    let hidden = usize::try_from(metadata.require_u64("qwen3vl.embedding_length")?)
        .map_err(|_| Error::invalid("Qwen3-VL hidden size exceeds usize"))?;
    let layers = usize::try_from(metadata.require_u64("qwen3vl.block_count")?)
        .map_err(|_| Error::invalid("Qwen3-VL layer count exceeds usize"))?;
    let heads = usize::try_from(metadata.require_u64("qwen3vl.attention.head_count")?)
        .map_err(|_| Error::invalid("Qwen3-VL head count exceeds usize"))?;
    let kv_heads = usize::try_from(metadata.require_u64("qwen3vl.attention.head_count_kv")?)
        .map_err(|_| Error::invalid("Qwen3-VL KV head count exceeds usize"))?;
    let head_dim = usize::try_from(metadata.require_u64("qwen3vl.attention.key_length")?)
        .map_err(|_| Error::invalid("Qwen3-VL head width exceeds usize"))?;
    let ffn = usize::try_from(metadata.require_u64("qwen3vl.feed_forward_length")?)
        .map_err(|_| Error::invalid("Qwen3-VL FFN width exceeds usize"))?;
    let vocab = metadata.require_strings("tokenizer.ggml.tokens")?.len();
    let q_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::invalid("Qwen3-VL q width overflow"))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::invalid("Qwen3-VL kv width overflow"))?;
    let mut expected = BTreeMap::new();
    insert(&mut expected, "token_embd.weight", vec![vocab, hidden])?;
    insert(&mut expected, "output_norm.weight", vec![hidden])?;
    if tensors.iter().any(|tensor| tensor.name == "output.weight") {
        insert(&mut expected, "output.weight", vec![vocab, hidden])?;
    }
    for layer in 0..layers {
        let prefix = format!("blk.{layer}");
        insert(
            &mut expected,
            &format!("{prefix}.attn_norm.weight"),
            vec![hidden],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.attn_q.weight"),
            vec![q_width, hidden],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.attn_k.weight"),
            vec![kv_width, hidden],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.attn_v.weight"),
            vec![kv_width, hidden],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.attn_output.weight"),
            vec![hidden, q_width],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.attn_q_norm.weight"),
            vec![head_dim],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.attn_k_norm.weight"),
            vec![head_dim],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.ffn_norm.weight"),
            vec![hidden],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.ffn_gate.weight"),
            vec![ffn, hidden],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.ffn_up.weight"),
            vec![ffn, hidden],
        )?;
        insert(
            &mut expected,
            &format!("{prefix}.ffn_down.weight"),
            vec![hidden, ffn],
        )?;
    }
    check(tensors, &mut expected, |ty| {
        matches!(
            ty,
            GgufType::F32
                | GgufType::F16
                | GgufType::Bf16
                | GgufType::Q4_0
                | GgufType::Q4K
                | GgufType::Q5K
                | GgufType::Q6K
                | GgufType::Q8_0
        )
    })
}

/// Validate every mmproj tensor against the `clip` geometry and the two patch slices.
/// # Errors
/// Rejects a missing or extra tensor, a shape mismatch, or a non-float storage type.
pub fn validate_vision(metadata: &Metadata, tensors: &[TensorSpec]) -> Result<()> {
    let hidden = usize::try_from(metadata.require_u64("clip.vision.embedding_length")?)
        .map_err(|_| Error::invalid("Qwen3-VL vision hidden size exceeds usize"))?;
    let layers = usize::try_from(metadata.require_u64("clip.vision.block_count")?)
        .map_err(|_| Error::invalid("Qwen3-VL vision depth exceeds usize"))?;
    let ffn = usize::try_from(metadata.require_u64("clip.vision.feed_forward_length")?)
        .map_err(|_| Error::invalid("Qwen3-VL vision FFN width exceeds usize"))?;
    let patch = usize::try_from(metadata.require_u64("clip.vision.patch_size")?)
        .map_err(|_| Error::invalid("Qwen3-VL patch size exceeds usize"))?;
    let image = usize::try_from(metadata.require_u64("clip.vision.image_size")?)
        .map_err(|_| Error::invalid("Qwen3-VL image size exceeds usize"))?;
    let merge = usize::try_from(metadata.require_u64("clip.vision.spatial_merge_size")?)
        .map_err(|_| Error::invalid("Qwen3-VL merge size exceeds usize"))?;
    let out = usize::try_from(metadata.require_u64("clip.vision.projection_dim")?)
        .map_err(|_| Error::invalid("Qwen3-VL projection width exceeds usize"))?;
    if patch == 0 || !image.is_multiple_of(patch) {
        return Err(Error::invalid(
            "Qwen3-VL image size must divide by patch size",
        ));
    }
    let positions = (image / patch)
        .checked_mul(image / patch)
        .ok_or_else(|| Error::invalid("Qwen3-VL position grid overflow"))?;
    let qkv = hidden
        .checked_mul(QKV_WIDTH)
        .ok_or_else(|| Error::invalid("Qwen3-VL vision qkv width overflow"))?;
    let merged = hidden
        .checked_mul(merge)
        .and_then(|value| value.checked_mul(merge))
        .ok_or_else(|| Error::invalid("Qwen3-VL merged width overflow"))?;
    let geometry = VisionGeometry {
        layers,
        hidden,
        ffn,
        qkv,
        merged,
        out,
    };
    let mut expected = BTreeMap::new();
    for slice in 0..PATCH_SLICES {
        let name = if slice == 0 {
            "v.patch_embd.weight".to_owned()
        } else {
            format!("v.patch_embd.weight.{slice}")
        };
        insert(&mut expected, &name, vec![hidden, QKV_WIDTH, patch, patch])?;
    }
    insert(&mut expected, "v.patch_embd.bias", vec![hidden])?;
    insert(
        &mut expected,
        "v.position_embd.weight",
        vec![positions, hidden],
    )?;
    insert(&mut expected, "v.post_ln.weight", vec![hidden])?;
    insert(&mut expected, "v.post_ln.bias", vec![hidden])?;
    insert(&mut expected, "mm.0.weight", vec![merged, merged])?;
    insert(&mut expected, "mm.0.bias", vec![merged])?;
    insert(&mut expected, "mm.2.weight", vec![out, merged])?;
    insert(&mut expected, "mm.2.bias", vec![out])?;
    insert_vision_blocks(&mut expected, &geometry)?;
    insert_deepstack(&mut expected, metadata, &geometry)?;
    check(tensors, &mut expected, |ty| {
        matches!(ty, GgufType::F32 | GgufType::F16 | GgufType::Bf16)
    })?;
    Ok(())
}

fn insert_vision_blocks(
    expected: &mut BTreeMap<String, Vec<usize>>,
    geometry: &VisionGeometry,
) -> Result<()> {
    for layer in 0..geometry.layers {
        let prefix = format!("v.blk.{layer}");
        insert(
            expected,
            &format!("{prefix}.attn_qkv.weight"),
            vec![geometry.qkv, geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.attn_qkv.bias"),
            vec![geometry.qkv],
        )?;
        insert(
            expected,
            &format!("{prefix}.attn_out.weight"),
            vec![geometry.hidden, geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.attn_out.bias"),
            vec![geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.ffn_up.weight"),
            vec![geometry.ffn, geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.ffn_up.bias"),
            vec![geometry.ffn],
        )?;
        insert(
            expected,
            &format!("{prefix}.ffn_down.weight"),
            vec![geometry.hidden, geometry.ffn],
        )?;
        insert(
            expected,
            &format!("{prefix}.ffn_down.bias"),
            vec![geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.ln1.weight"),
            vec![geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.ln1.bias"),
            vec![geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.ln2.weight"),
            vec![geometry.hidden],
        )?;
        insert(
            expected,
            &format!("{prefix}.ln2.bias"),
            vec![geometry.hidden],
        )?;
    }
    Ok(())
}

fn insert_deepstack(
    expected: &mut BTreeMap<String, Vec<usize>>,
    metadata: &Metadata,
    geometry: &VisionGeometry,
) -> Result<()> {
    let deepstack = bool_array(metadata, "clip.vision.is_deepstack_layers")?
        .into_iter()
        .filter(|enabled| *enabled)
        .count();
    if u64::try_from(deepstack).unwrap_or(u64::MAX) != DEEPSTACK_COUNT {
        return Err(Error::invalid("Qwen3-VL deepstack layer count"));
    }
    for layer in 0..geometry.layers {
        if !bool_array(metadata, "clip.vision.is_deepstack_layers")?[layer] {
            continue;
        }
        let prefix = format!("v.deepstack.{layer}");
        insert(
            expected,
            &format!("{prefix}.norm.weight"),
            vec![geometry.merged],
        )?;
        insert(
            expected,
            &format!("{prefix}.norm.bias"),
            vec![geometry.merged],
        )?;
        insert(
            expected,
            &format!("{prefix}.fc1.weight"),
            vec![geometry.merged, geometry.merged],
        )?;
        insert(
            expected,
            &format!("{prefix}.fc1.bias"),
            vec![geometry.merged],
        )?;
        insert(
            expected,
            &format!("{prefix}.fc2.weight"),
            vec![geometry.out, geometry.merged],
        )?;
        insert(expected, &format!("{prefix}.fc2.bias"), vec![geometry.out])?;
    }
    Ok(())
}

fn insert(
    expected: &mut BTreeMap<String, Vec<usize>>,
    name: &str,
    shape: Vec<usize>,
) -> Result<()> {
    if expected.insert(name.to_owned(), shape).is_some() {
        return Err(Error::invalid(format!("duplicate expected tensor {name}")));
    }
    Ok(())
}

fn check(
    tensors: &[TensorSpec],
    expected: &mut BTreeMap<String, Vec<usize>>,
    supported: impl Fn(GgufType) -> bool,
) -> Result<()> {
    for tensor in tensors {
        let shape = expected
            .remove(&tensor.name)
            .ok_or_else(|| Error::invalid(format!("unexpected tensor {}", tensor.name)))?;
        if tensor.shape != shape {
            return Err(Error::invalid(format!(
                "tensor {} shape {:?}, expected {shape:?}",
                tensor.name, tensor.shape
            )));
        }
        if !supported(tensor.ty) {
            return Err(Error::invalid(format!(
                "tensor {} has unsupported type {}",
                tensor.name,
                tensor.ty.name()
            )));
        }
    }
    if let Some(name) = expected.keys().next() {
        return Err(Error::invalid(format!("missing tensor {name}")));
    }
    Ok(())
}

fn bool_array(metadata: &Metadata, key: &str) -> Result<Vec<bool>> {
    metadata
        .get(key)
        .and_then(infer_gguf::MetadataValue::as_array)
        .ok_or_else(|| Error::invalid(format!("GGUF metadata {key} is not an array")))?
        .iter()
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| Error::invalid(format!("GGUF metadata {key} is not a bool array")))
        })
        .collect()
}
