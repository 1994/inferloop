/// A released Qwen3-VL checkpoint must resolve and import like its 3.5 sibling.
///
/// The fixture mirrors `Qwen/Qwen3-VL-2B-Instruct`, the package the end-to-end parity run uses:
/// a plain `qwen3_vl` text backbone, tied embeddings and the same vision geometry family.
#[test]
fn qwen3_vl_configuration_imports_with_its_image_encoder() -> Result<()> {
    const CONFIG: &str = r#"{
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "image_token_id": 151655,
            "video_token_id": 151656,
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 2048,
                "vocab_size": 151936,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 6144,
                "max_position_embeddings": 262144,
                "head_dim": 128,
                "rms_norm_eps": 1e-06,
                "rope_theta": 5000000,
                "rope_scaling": {"mrope_section": [24,20,20], "mrope_interleaved": true, "rope_type": "default"},
                "hidden_act": "silu",
                "tie_word_embeddings": true
            },
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_heads": 16,
                "in_channels": 3,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 2048,
                "num_position_embeddings": 2304,
                "hidden_act": "gelu_pytorch_tanh",
                "deepstack_visual_indexes": [5, 11, 17]
            }
        }"#;
    let bytes = CONFIG.as_bytes();
    let provider = crate::default_registry().resolve(bytes)?;
    assert!(provider.architectures().contains(&"qwen3_vl"));
    let imported = provider.import(ModelId::ONE, bytes)?;
    assert_eq!(imported.model.hidden_size, 2048);
    assert_eq!(imported.model.vocab_size, 151_936);
    assert_eq!(imported.draft_layers, 0);
    assert_eq!(imported.model.position.multimodal_sections, [24, 20, 20]);
    assert!(imported.model.position.interleaved);
    assert!((imported.model.position.rope_theta - 5_000_000.0).abs() < f64::EPSILON);
    let image = require_encoder(&imported, &Modality::Image)?;
    assert_eq!(image.prefix, "model.visual.");
    assert_eq!(image.depth, 24);
    assert_eq!(image.hidden_size, 1024);
    assert_eq!(image.intermediate_size, 4096);
    assert_eq!(image.out_hidden_size, 2048);
    assert_eq!(
        slots(image)?.len(),
        12 * image.depth + 9,
        "the vision inventory must follow the declared depth"
    );
    assert_eq!(placeholders(&imported, &Modality::Image)?, [151_655]);
    // Each deepstack level is added to the image tokens after one leading decoder layer, in order.
    assert_eq!(imported.model.media_features, 3);
    let graph = provider.graph(&imported.model)?;
    let features: Vec<_> = graph
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            infer_ir::TensorOp::MediaFeature { index } => Some((index, node.layer)),
            _ => None,
        })
        .collect();
    assert_eq!(features, [(0, Some(0)), (1, Some(1)), (2, Some(2))]);
    assert_eq!(graph.media_features(), 3);
    Ok(())
}
use super::*;
use crate::prompt::placeholders;
use crate::prompt::require_encoder;
use crate::vision::slots;
#[test]
fn imports_pinned_qwen38_hybrid_architecture() {
    let imported = QwenProvider
        .import_manifest(
            ModelId::new(1).unwrap(),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../examples/qwen3.8-27b/config.json"
            )),
        )
        .unwrap();
    assert_eq!(imported.model.mixers.len(), 64);
    assert_eq!(
        imported
            .model
            .mixers
            .iter()
            .filter(|m| matches!(m, Mixer::Attention { .. }))
            .count(),
        16
    );
    assert_eq!(
        imported
            .model
            .state
            .iter()
            .filter(|s| s.kind == StateKind::LinearAttention)
            .count(),
        48
    );
    assert_eq!(imported.model.hidden_size, 5120);
    assert_eq!(imported.model.vocab_size, 248_320);
    assert!(imported.model.modalities.contains(&Modality::Image));
    assert_eq!(imported.draft_layers, 1);
}
#[test]
fn official_bf16_package_does_not_fit_32_gib() {
    let imported = QwenProvider
        .import(
            ModelId::new(1).unwrap(),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../examples/qwen3.8-27b/config.json"
            )),
        )
        .unwrap();
    let index = crate::SafetensorsIndex::parse(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../examples/qwen3.8-27b/model.safetensors.index.json"
    )))
    .unwrap();
    let budget = crate::memory_estimate(
        &imported.model,
        index.weight_bytes().unwrap(),
        4096,
        1,
        1 << 30,
        32 << 30,
    )
    .unwrap();
    assert_eq!(index.weight_bytes().unwrap(), 55_562_855_904);
    assert!(!budget.fits);
    assert!(budget.state_bytes > 0);
}
#[test]
fn rejects_shard_path_traversal() {
    let bytes = br#"{"metadata":{"total_size":4},"weight_map":{"weight":"../weight.safetensors"}}"#;
    assert!(crate::SafetensorsIndex::parse(bytes).is_err());
}

#[test]
fn moe_hybrid_import_preserves_routing_and_dense_layer_policy() -> Result<()> {
    let mut config: serde_json::Value = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../examples/qwen-next-moe-tiny/config.json"
    )))
    .map_err(|e| Error::invalid(e.to_string()))?;
    config["decoder_sparse_step"] = 2.into();
    let bytes = serde_json::to_vec(&config).map_err(|e| Error::invalid(e.to_string()))?;
    let imported = QwenProvider.import_manifest(ModelId::ONE, &bytes)?;
    let FeedForward::Moe {
        experts,
        active,
        shared_intermediate,
        dense_layers,
        ..
    } = &imported.model.feed_forward
    else {
        return Err(Error::invariant("expected MoE"));
    };
    assert_eq!((*experts, *active), (4, 2));
    assert_eq!(*shared_intermediate, Some(16));
    assert_eq!(dense_layers, &[0]);
    assert!(matches!(
        imported.model.mixers[0],
        Mixer::LinearAttention { .. }
    ));
    assert!(matches!(
        imported.model.mixers[1],
        Mixer::Attention {
            output_gate: true,
            ..
        }
    ));
    config["num_experts_per_tok"] = 5.into();
    assert!(
        QwenProvider
            .import_manifest(ModelId::ONE, &serde_json::to_vec(&config).unwrap())
            .is_err()
    );
    config["num_experts_per_tok"] = 2.into();
    config["decoder_sparse_step"] = 0.into();
    assert!(
        QwenProvider
            .import_manifest(ModelId::ONE, &serde_json::to_vec(&config).unwrap())
            .is_err()
    );
    Ok(())
}
