use super::{VisionTensor, config, text_tensor, vision_tensor};
use infer_gguf::{Metadata, MetadataValue};

fn metadata(entries: Vec<(&str, MetadataValue)>) -> Metadata {
    Metadata::from_entries(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value)),
    )
}

fn text_metadata() -> Metadata {
    metadata(vec![
        (
            "general.architecture",
            MetadataValue::String("qwen3vl".into()),
        ),
        (
            "tokenizer.ggml.tokens",
            MetadataValue::Array(vec![
                MetadataValue::String("a".into()),
                MetadataValue::String("<|image_pad|>".into()),
                MetadataValue::String("<|video_pad|>".into()),
                MetadataValue::String("b".into()),
            ]),
        ),
        ("qwen3vl.embedding_length", MetadataValue::U64(2048)),
        ("qwen3vl.block_count", MetadataValue::U64(28)),
        ("qwen3vl.attention.head_count", MetadataValue::U64(16)),
        ("qwen3vl.attention.head_count_kv", MetadataValue::U64(8)),
        ("qwen3vl.feed_forward_length", MetadataValue::U64(6144)),
        ("qwen3vl.context_length", MetadataValue::U64(262_144)),
        ("qwen3vl.attention.key_length", MetadataValue::U64(128)),
        (
            "qwen3vl.attention.layer_norm_rms_epsilon",
            MetadataValue::F32(1e-6),
        ),
        ("qwen3vl.rope.freq_base", MetadataValue::F32(5_000_000.0)),
        (
            "qwen3vl.rope.dimension_sections",
            MetadataValue::Array(vec![
                MetadataValue::U64(24),
                MetadataValue::U64(20),
                MetadataValue::U64(20),
                MetadataValue::U64(0),
            ]),
        ),
    ])
}

fn mmproj_metadata() -> Metadata {
    metadata(vec![
        ("general.architecture", MetadataValue::String("clip".into())),
        ("clip.vision.image_size", MetadataValue::U64(768)),
        ("clip.vision.patch_size", MetadataValue::U64(16)),
        ("clip.vision.block_count", MetadataValue::U64(24)),
        ("clip.vision.embedding_length", MetadataValue::U64(1024)),
        ("clip.vision.feed_forward_length", MetadataValue::U64(4096)),
        ("clip.vision.attention.head_count", MetadataValue::U64(16)),
        ("clip.vision.spatial_merge_size", MetadataValue::U64(2)),
        ("clip.vision.projection_dim", MetadataValue::U64(2048)),
        (
            "clip.vision.is_deepstack_layers",
            MetadataValue::Array(
                (0..24)
                    .map(|index| MetadataValue::Bool(matches!(index, 5 | 11 | 17)))
                    .collect(),
            ),
        ),
    ])
}

#[test]
fn synthesizes_the_hf_configuration() {
    let bytes = config(&text_metadata(), &mmproj_metadata()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["model_type"], "qwen3_vl");
    assert_eq!(value["image_token_id"], 1);
    assert_eq!(value["video_token_id"], 2);
    assert_eq!(value["text_config"]["vocab_size"], 4);
    assert_eq!(value["text_config"]["hidden_size"], 2048);
    assert_eq!(value["text_config"]["rope_scaling"]["mrope_section"][2], 20);
    assert_eq!(
        value["vision_config"]["deepstack_visual_indexes"],
        serde_json::json!([5, 11, 17])
    );
    assert_eq!(value["vision_config"]["num_position_embeddings"], 2304);
    assert_eq!(value["vision_config"]["hidden_act"], "gelu_pytorch_tanh");
}

#[test]
fn maps_text_tensors_to_the_backbone() {
    assert_eq!(
        text_tensor("token_embd.weight").as_deref(),
        Some("model.embed_tokens.weight")
    );
    assert_eq!(
        text_tensor("blk.12.attn_q.weight").as_deref(),
        Some("model.layers.12.self_attn.q_proj.weight")
    );
    assert_eq!(
        text_tensor("blk.3.ffn_down.weight").as_deref(),
        Some("model.layers.3.mlp.down_proj.weight")
    );
    assert!(text_tensor("blk.0.attn_q_norm.bias").is_none());
}

#[test]
fn maps_patch_slices_and_vision_blocks() {
    assert_eq!(
        vision_tensor("v.patch_embd.weight", &[]),
        Some(VisionTensor::PatchSlice(0))
    );
    assert_eq!(
        vision_tensor("v.patch_embd.weight.1", &[]),
        Some(VisionTensor::PatchSlice(1))
    );
    assert_eq!(
        vision_tensor("v.blk.7.attn_qkv.weight", &[]),
        Some(VisionTensor::Slot(
            "model.visual.blocks.7.attn.qkv.weight".into()
        ))
    );
    assert_eq!(
        vision_tensor("v.blk.23.ffn_down.bias", &[]),
        Some(VisionTensor::Slot(
            "model.visual.blocks.23.mlp.linear_fc2.bias".into()
        ))
    );
    assert_eq!(
        vision_tensor("v.position_embd.weight", &[]),
        Some(VisionTensor::Slot("model.visual.pos_embed.weight".into()))
    );
}

#[test]
fn maps_deepstack_mergers_by_their_text_layer() {
    let layers = [5, 11, 17];
    assert_eq!(
        vision_tensor("v.deepstack.11.fc1.weight", &layers),
        Some(VisionTensor::Slot(
            "model.visual.deepstack_merger_list.1.linear_fc1.weight".into()
        ))
    );
    assert_eq!(
        vision_tensor("v.deepstack.17.norm.bias", &layers),
        Some(VisionTensor::Slot(
            "model.visual.deepstack_merger_list.2.norm.bias".into()
        ))
    );
    assert!(vision_tensor("v.deepstack.4.fc1.weight", &layers).is_none());
}
