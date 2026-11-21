//! The `dflash` adapter: a `DFlash2` draft as llama.cpp stores it.
use super::support::{architecture, mapped as map, spec};
use infer_format::{GgufType, Metadata, MetadataValue};
use infer_models::DFlash2Config;

fn metadata(causal: bool) -> Metadata {
    let u = MetadataValue::U64;
    let tokens = (0..32)
        .map(|i| MetadataValue::String(format!("t{i}")))
        .collect();
    Metadata::from_entries(
        [
            (
                "general.architecture",
                MetadataValue::String("dflash".into()),
            ),
            ("dflash.block_count", u(2)),
            ("dflash.context_length", u(4096)),
            ("dflash.embedding_length", u(64)),
            ("dflash.feed_forward_length", u(128)),
            ("dflash.attention.head_count", u(4)),
            ("dflash.attention.head_count_kv", u(2)),
            ("dflash.attention.key_length", u(16)),
            ("dflash.attention.causal", MetadataValue::Bool(causal)),
            ("dflash.attention.sliding_window", u(256)),
            (
                "dflash.attention.sliding_window_pattern",
                MetadataValue::Array(vec![MetadataValue::Bool(true); 2]),
            ),
            (
                "dflash.attention.layer_norm_rms_epsilon",
                MetadataValue::F32(1e-6),
            ),
            ("dflash.rope.freq_base", MetadataValue::F32(10_000.0)),
            ("dflash.block_size", u(8)),
            ("dflash.conv_kernel_size", u(2)),
            ("dflash.conv_group_size", u(16)),
            ("dflash.selector_rank", u(8)),
            ("dflash.selector_top_k", u(4)),
            (
                "dflash.target_layers",
                MetadataValue::Array(vec![u(2), u(5)]),
            ),
            ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
            ("tokenizer.ggml.mask_token_id", u(30)),
            ("tokenizer.ggml.padding_token_id", u(31)),
        ]
        .map(|(key, value)| (key.to_owned(), value)),
    )
}

fn mapped(name: &str) -> Option<String> {
    map(
        architecture("dflash"),
        &metadata(false),
        &spec(name, &[64], GgufType::Bf16),
    )
    .map(|mapped| mapped.slot)
}

#[test]
fn the_synthesized_config_parses_as_the_draft_with_hf_tap_numbers() {
    let config = architecture("dflash")
        .config(&metadata(false), &[], None)
        .unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&config).unwrap();
    assert!(value.get("num_target_layers").is_none());
    value["num_target_layers"] = 6.into();
    let parsed = DFlash2Config::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(parsed.dflash_config.target_layer_ids, [1, 4]);
    assert_eq!(parsed.dflash_config.mask_token_id, Some(30));
    assert_eq!((parsed.eos_token_id, parsed.pad_token_id), (None, Some(31)));
    assert_eq!(parsed.num_hidden_layers, 2);
    parsed.check_block_semantics().unwrap();
    parsed.check_special_tokens().unwrap();
}

#[test]
fn tensors_take_the_safetensors_draft_names() {
    for (stored, slot) in [
        (
            "selector_predecessor.weight",
            "candidate_selector.predecessor_codebook",
        ),
        ("enc.output_norm.weight", "hidden_norm.weight"),
        ("output_norm.weight", "norm.weight"),
        (
            "blk.1.attn_conv_base",
            "layers.1.attention_conv.base_kernel",
        ),
        (
            "blk.0.ffn_conv_proj.weight",
            "layers.0.mlp_conv.kernel_projection.weight",
        ),
        (
            "blk.1.attn_q_norm.weight",
            "layers.1.self_attn.q_norm.weight",
        ),
        ("blk.0.ffn_down.weight", "layers.0.mlp.down_proj.weight"),
    ] {
        assert_eq!(mapped(stored).as_deref(), Some(slot), "{stored}");
    }
    assert_eq!(mapped("blk.2.attn_q.weight"), None);
    assert_eq!(mapped("token_embd.weight"), None);
}

#[test]
fn a_causal_draft_is_rejected() {
    assert!(
        architecture("dflash")
            .validate(&metadata(false), &[])
            .is_ok()
    );
    let error = architecture("dflash")
        .validate(&metadata(true), &[])
        .unwrap_err();
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
}
