//! The `qwen3vl` text adapter and the `qwen3vl_merger` mmproj projector.
use super::support::{architecture, mapped as map, metadata, projector, spec, with};
use infer_format::{GgufType, Metadata, MetadataValue};
use infer_models::gguf::{TensorSpec, VisionTensor};
use infer_models::vision::slots;

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
        (
            "clip.projector_type",
            MetadataValue::String("qwen3vl_merger".into()),
        ),
        ("clip.vision.image_size", MetadataValue::U64(768)),
        ("clip.vision.patch_size", MetadataValue::U64(16)),
        ("clip.vision.block_count", MetadataValue::U64(24)),
        ("clip.vision.embedding_length", MetadataValue::U64(1024)),
        ("clip.vision.feed_forward_length", MetadataValue::U64(4096)),
        ("clip.vision.attention.head_count", MetadataValue::U64(16)),
        (
            "clip.vision.attention.layer_norm_epsilon",
            MetadataValue::F32(1e-6),
        ),
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

fn text_tensor(name: &str) -> Option<String> {
    map(
        architecture("qwen3vl"),
        &text_metadata(),
        &spec(name, &[1], GgufType::F32),
    )
    .map(|mapped| mapped.slot)
}

fn vision_tensor(name: &str) -> Option<VisionTensor> {
    let projector = projector("qwen3vl_merger");
    projector.tensor(name, &projector.encoder(&mmproj_metadata()).unwrap())
}

#[test]
fn synthesizes_the_hf_configuration() {
    let bytes = architecture("qwen3vl")
        .config(&text_metadata(), &[], Some(&mmproj_metadata()))
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["model_type"], "qwen3_vl");
    assert_eq!(value["image_token_id"], 1);
    assert_eq!(value["video_token_id"], 2);
    assert_eq!(value["language_model_only"], false);
    assert_eq!(value["text_config"]["vocab_size"], 4);
    assert_eq!(value["text_config"]["hidden_size"], 2048);
    assert_eq!(value["text_config"]["rope_scaling"]["mrope_section"][2], 20);
    assert_eq!(
        value["vision_config"]["deepstack_visual_indexes"],
        serde_json::json!([5, 11, 17])
    );
    assert_eq!(value["vision_config"]["num_position_embeddings"], 2304);
    assert_eq!(value["vision_config"]["hidden_act"], "gelu_pytorch_tanh");
    let text = architecture("qwen3vl")
        .config(&text_metadata(), &[], None)
        .unwrap();
    let text: serde_json::Value = serde_json::from_slice(&text).unwrap();
    assert_eq!(text["language_model_only"], true);
    assert!(text.get("vision_config").is_none());
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
    assert!(text_tensor("blk.28.attn_q.weight").is_none());
}

#[test]
fn maps_patch_slices_vision_blocks_and_deepstack_mergers() {
    assert_eq!(
        vision_tensor("v.patch_embd.weight"),
        Some(VisionTensor::PatchSlice(0))
    );
    assert_eq!(
        vision_tensor("v.patch_embd.weight.1"),
        Some(VisionTensor::PatchSlice(1))
    );
    for (stored, slot) in [
        (
            "v.blk.7.attn_qkv.weight",
            "model.visual.blocks.7.attn.qkv.weight",
        ),
        (
            "v.blk.23.ffn_down.bias",
            "model.visual.blocks.23.mlp.linear_fc2.bias",
        ),
        ("v.position_embd.weight", "model.visual.pos_embed.weight"),
        ("v.patch_embd.bias", "model.visual.patch_embed.proj.bias"),
        ("v.post_ln.weight", "model.visual.merger.norm.weight"),
        ("mm.2.bias", "model.visual.merger.linear_fc2.bias"),
        (
            "v.deepstack.11.fc1.weight",
            "model.visual.deepstack_merger_list.1.linear_fc1.weight",
        ),
        (
            "v.deepstack.17.norm.bias",
            "model.visual.deepstack_merger_list.2.norm.bias",
        ),
    ] {
        assert_eq!(
            vision_tensor(stored),
            Some(VisionTensor::Slot(slot.into())),
            "{stored}"
        );
    }
    assert!(vision_tensor("v.deepstack.4.fc1.weight").is_none());
    assert!(vision_tensor("mm.1.weight").is_none());
}

#[test]
fn declares_the_vision_encoder_and_preprocessor() {
    let projector = projector("qwen3vl_merger");
    let encoder = projector.encoder(&mmproj_metadata()).unwrap();
    assert_eq!(encoder.depth, 24);
    assert_eq!(encoder.hidden_size, 1024);
    assert_eq!(encoder.heads, 16);
    assert_eq!(encoder.temporal_patch_size, 2);
    assert_eq!(encoder.position_embeddings, 2304);
    assert_eq!(encoder.norm_epsilon, 1e-6);
    assert_eq!(encoder.deepstack_layers, [5, 11, 17]);
    assert_eq!(
        encoder.hidden_activation,
        infer_spi::HiddenActivation::GeluTanh
    );
    assert_eq!(slots(&encoder).unwrap().len(), 12 * 24 + 9);
    let mut with_norm = mmproj_metadata();
    for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
        with_norm = with(
            &with_norm,
            key,
            Some(MetadataValue::Array(vec![MetadataValue::F32(0.5); 3])),
        );
    }
    let config: serde_json::Value =
        serde_json::from_slice(&projector.preprocessor(&with_norm, &encoder).unwrap()).unwrap();
    assert_eq!(config["patch_size"], 16);
    assert_eq!(config["merge_size"], 2);
    assert_eq!(config["image_mean"], serde_json::json!([0.5, 0.5, 0.5]));
    assert!(
        projector
            .preprocessor(&mmproj_metadata(), &encoder)
            .is_err()
    );
}

fn text_specs() -> Vec<TensorSpec> {
    let mut specs = vec![
        TensorSpec {
            name: "token_embd.weight".into(),
            shape: vec![4, 8],
            ty: GgufType::Q4K,
        },
        TensorSpec {
            name: "output_norm.weight".into(),
            shape: vec![8],
            ty: GgufType::F32,
        },
    ];
    for (suffix, shape) in [
        ("attn_norm.weight", vec![8]),
        ("attn_q.weight", vec![8, 8]),
        ("attn_k.weight", vec![4, 8]),
        ("attn_v.weight", vec![4, 8]),
        ("attn_output.weight", vec![8, 8]),
        ("attn_q_norm.weight", vec![4]),
        ("attn_k_norm.weight", vec![4]),
        ("ffn_norm.weight", vec![8]),
        ("ffn_gate.weight", vec![16, 8]),
        ("ffn_up.weight", vec![16, 8]),
        ("ffn_down.weight", vec![8, 16]),
    ] {
        specs.push(TensorSpec {
            name: format!("blk.0.{suffix}"),
            shape,
            ty: GgufType::Q4K,
        });
    }
    specs
}

fn tiny_text_metadata() -> Metadata {
    metadata(vec![
        ("qwen3vl.embedding_length", MetadataValue::U64(8)),
        ("qwen3vl.block_count", MetadataValue::U64(1)),
        ("qwen3vl.attention.head_count", MetadataValue::U64(2)),
        ("qwen3vl.attention.head_count_kv", MetadataValue::U64(1)),
        ("qwen3vl.attention.key_length", MetadataValue::U64(4)),
        ("qwen3vl.feed_forward_length", MetadataValue::U64(16)),
        (
            "tokenizer.ggml.tokens",
            MetadataValue::Array(vec![
                MetadataValue::String("a".into()),
                MetadataValue::String("b".into()),
                MetadataValue::String("c".into()),
                MetadataValue::String("d".into()),
            ]),
        ),
    ])
}

#[test]
fn validates_text_shapes_and_rejects_mismatches() {
    assert!(
        architecture("qwen3vl")
            .validate(&tiny_text_metadata(), &text_specs())
            .is_ok()
    );
    let mut broken = text_specs();
    broken[1].shape = vec![7];
    assert!(
        architecture("qwen3vl")
            .validate(&tiny_text_metadata(), &broken)
            .is_err()
    );
    let mut missing = text_specs();
    missing.pop();
    assert!(
        architecture("qwen3vl")
            .validate(&tiny_text_metadata(), &missing)
            .is_err()
    );
}
