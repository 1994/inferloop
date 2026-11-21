//! The `qwen35` adapter: Qwen3.5's hybrid backbone and its `nextn` MTP head.
use super::support::{architecture, mapped as map, spec as tensor, with};
use infer_format::{GgufType, Metadata, MetadataValue};
use infer_models::gguf::{MappedTensor, TensorSpec, ValueTransform};

fn metadata() -> Metadata {
    let u = MetadataValue::U64;
    let tokens = ["a", "<|image_pad|>", "<|video_pad|>"]
        .map(|token| MetadataValue::String(token.into()))
        .to_vec();
    Metadata::from_entries(
        [
            (
                "general.architecture",
                MetadataValue::String("qwen35".into()),
            ),
            ("qwen35.block_count", u(5)),
            ("qwen35.nextn_predict_layers", u(1)),
            ("qwen35.context_length", u(4096)),
            ("qwen35.embedding_length", u(64)),
            ("qwen35.feed_forward_length", u(128)),
            ("qwen35.attention.head_count", u(4)),
            ("qwen35.attention.head_count_kv", u(2)),
            ("qwen35.attention.key_length", u(32)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                MetadataValue::F32(1e-6),
            ),
            ("qwen35.rope.freq_base", MetadataValue::F32(10_000.0)),
            ("qwen35.rope.dimension_count", u(8)),
            (
                "qwen35.rope.dimension_sections",
                MetadataValue::Array(vec![u(2), u(1), u(1), u(0)]),
            ),
            ("qwen35.full_attention_interval", u(4)),
            ("qwen35.ssm.conv_kernel", u(4)),
            ("qwen35.ssm.state_size", u(16)),
            ("qwen35.ssm.group_count", u(2)),
            ("qwen35.ssm.time_step_rank", u(2)),
            ("qwen35.ssm.inner_size", u(32)),
            ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
        ]
        .map(|(key, value)| (key.to_owned(), value)),
    )
}

fn spec(name: &str, shape: &[usize]) -> TensorSpec {
    tensor(name, shape, GgufType::F32)
}

fn mapped(name: &str, shape: &[usize]) -> Option<MappedTensor> {
    map(architecture("qwen35"), &metadata(), &spec(name, shape))
}

#[test]
fn backbone_and_head_tensors_take_their_hf_names_and_inverse_transforms() {
    let norm = mapped("blk.0.attn_norm.weight", &[64]).unwrap();
    assert_eq!(
        norm.slot,
        "model.language_model.layers.0.input_layernorm.weight"
    );
    assert_eq!(norm.transform, ValueTransform::PlusOne);
    let gated = mapped("blk.0.ssm_norm.weight", &[16]).unwrap();
    assert_eq!(
        gated.slot,
        "model.language_model.layers.0.linear_attn.norm.weight"
    );
    assert_eq!(gated.transform, ValueTransform::Identity);
    let decay = mapped("blk.1.ssm_a", &[2]).unwrap();
    assert_eq!(
        decay.slot,
        "model.language_model.layers.1.linear_attn.A_log"
    );
    assert_eq!(decay.transform, ValueTransform::NegatedExp);
    let conv = mapped("blk.1.ssm_conv1d.weight", &[96, 4]).unwrap();
    assert_eq!(conv.shape, Some(vec![96, 1, 4]));
    assert_eq!(
        mapped("blk.3.attn_q.weight", &[64, 256]).unwrap().slot,
        "model.language_model.layers.3.self_attn.q_proj.weight"
    );
    // Block 4 is the trailing nextn block: the MTP head.
    let fc = mapped("blk.4.nextn.eh_proj.weight", &[64, 128]).unwrap();
    assert_eq!(
        (fc.slot.as_str(), fc.transform),
        ("mtp.fc.weight", ValueTransform::Identity)
    );
    for (field, slot) in [
        ("nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight"),
        ("nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"),
        ("nextn.shared_head_norm.weight", "mtp.norm.weight"),
    ] {
        let head = mapped(&format!("blk.4.{field}"), &[64]).unwrap();
        assert_eq!(
            (head.slot.as_str(), head.transform),
            (slot, ValueTransform::PlusOne)
        );
    }
    let layer = mapped("blk.4.attn_k.weight", &[32, 64]).unwrap();
    assert_eq!(layer.slot, "mtp.layers.0.self_attn.k_proj.weight");
    assert!(mapped("blk.0.unknown.weight", &[1]).is_none());
    assert!(mapped("blk.5.attn_q.weight", &[1]).is_none());
}

#[test]
fn the_synthesized_config_imports_as_the_hybrid_with_its_mtp_head() {
    let tensors = [spec("token_embd.weight", &[3, 64])];
    let config = architecture("qwen35")
        .config(&metadata(), &tensors, None)
        .unwrap();
    let value: serde_json::Value = serde_json::from_slice(&config).unwrap();
    let text = &value["text_config"];
    assert_eq!(text["num_hidden_layers"], 4);
    assert_eq!(text["mtp_num_hidden_layers"], 1);
    assert_eq!(text["layer_types"][3], "full_attention");
    assert_eq!(text["layer_types"][0], "linear_attention");
    assert_eq!(
        text["rope_parameters"]["mrope_section"],
        serde_json::json!([2, 1, 1])
    );
    assert_eq!(text["rope_parameters"]["partial_rotary_factor"], 0.25);
    assert_eq!(value["tie_word_embeddings"], true);
    let imported = infer_models::default_registry()
        .resolve(&config)
        .unwrap()
        .import(infer_core::ModelId::ONE, &config)
        .unwrap();
    assert_eq!(imported.model.mixers.len(), 4);
    assert_eq!(imported.draft_layers, 1);
    let untied = [
        spec("token_embd.weight", &[3, 64]),
        spec("output.weight", &[3, 64]),
    ];
    let value: serde_json::Value = serde_json::from_slice(
        &architecture("qwen35")
            .config(&metadata(), &untied, None)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(value["tie_word_embeddings"], false);
}

#[test]
fn tiled_value_heads_are_rejected_explicitly() {
    let tiled = with(
        &with(
            &metadata(),
            "qwen35.ssm.time_step_rank",
            Some(MetadataValue::U64(6)),
        ),
        "qwen35.ssm.inner_size",
        Some(MetadataValue::U64(96)),
    );
    let error = architecture("qwen35")
        .validate(&tiled, &[spec("token_embd.weight", &[3, 64])])
        .unwrap_err();
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
}
