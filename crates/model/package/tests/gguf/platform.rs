//! What the platform owns for every architecture.
use super::support::{metadata, spec};
use infer_core::ErrorCode;
use infer_format::{GgufType, MetadataValue};
use infer_ir::{ExecutableEncoding, GroupedIntFormat};
use infer_models::default_registry;
use infer_models::gguf::{
    Block, ExpectedTensors, LayerPlan, ValueTransform, common_block_field, encoding_for,
    trim_trailing_zeros,
};

#[test]
fn trailing_nextn_blocks_become_the_draft_head() {
    let file = metadata(vec![
        ("demo.block_count", MetadataValue::U64(25)),
        ("demo.nextn_predict_layers", MetadataValue::U64(1)),
    ]);
    let layers = LayerPlan::of(&file, "demo").unwrap();
    assert_eq!(
        layers,
        LayerPlan {
            backbone: 24,
            draft: 1
        }
    );
    assert_eq!(
        layers.block("blk.3.attn_q.weight"),
        Some(Block {
            index: 3,
            draft: None,
            field: "attn_q.weight"
        })
    );
    assert_eq!(
        layers.block("blk.24.nextn.enorm.weight").unwrap().draft,
        Some(0)
    );
    assert_eq!(layers.block("blk.25.attn_q.weight"), None);
    assert_eq!(layers.block("token_embd.weight"), None);
}

#[test]
fn a_file_without_nextn_has_no_draft_and_one_with_only_nextn_is_rejected() {
    let plain = metadata(vec![("demo.block_count", MetadataValue::U64(4))]);
    assert_eq!(LayerPlan::of(&plain, "demo").unwrap().draft, 0);
    let only = metadata(vec![
        ("demo.block_count", MetadataValue::U64(1)),
        ("demo.nextn_predict_layers", MetadataValue::U64(1)),
    ]);
    assert!(LayerPlan::of(&only, "demo").is_err());
}

#[test]
fn transforms_recover_the_converter_inputs() {
    let original = [-0.75_f32, 0.0, 0.125];
    let mut stored: Vec<f32> = original.iter().map(|w| 1.0 + w).collect();
    ValueTransform::PlusOne.undo(&mut stored);
    assert_eq!(stored, original);
    let a_log = [0.5_f32, -1.25];
    let mut stored: Vec<f32> = a_log.iter().map(|x| -x.exp()).collect();
    ValueTransform::NegatedExp.undo(&mut stored);
    for (recovered, expected) in stored.iter().zip(a_log) {
        assert!(
            (recovered - expected).abs() < 1e-6,
            "{recovered} vs {expected}"
        );
    }
}

#[test]
fn shared_block_names() {
    assert_eq!(
        common_block_field("attn_q.weight"),
        Some("self_attn.q_proj.weight")
    );
    assert_eq!(
        common_block_field("post_attention_norm.weight"),
        Some("post_attention_layernorm.weight")
    );
    assert_eq!(common_block_field("ssm_a"), None);
    assert_eq!(trim_trailing_zeros(vec![11, 11, 10, 0]), [11, 11, 10]);
}

#[test]
fn the_registry_names_the_adapters_it_has() {
    let adapters = default_registry().gguf();
    for name in ["qwen3vl", "qwen35", "dflash"] {
        assert_eq!(adapters.architecture(name).unwrap().name(), name);
    }
    let error = adapters.architecture("demo").map(|_| ()).unwrap_err();
    assert_eq!(error.code, ErrorCode::Unsupported);
    assert!(error.message.contains("qwen3vl") && error.message.contains("dflash"));
    assert!(adapters.projector("qwen3vl_merger").is_ok());
    assert!(adapters.projector("llava").is_err());
}

#[test]
fn a_registry_takes_another_adapter_once() {
    let mut registry = infer_models::ModelRegistry::builtin();
    let qwen35 = default_registry().gguf().architecture("qwen35").unwrap();
    assert!(registry.gguf_mut().register_architecture(qwen35).is_err());
    let mut empty = infer_models::ModelRegistry::new();
    empty.gguf_mut().register_architecture(qwen35).unwrap();
    assert!(empty.gguf().architecture("qwen35").is_ok());
    assert!(empty.gguf().architecture("qwen3vl").is_err());
}

#[test]
fn expected_tensors_reject_missing_extra_and_misshapen() {
    let expect = || {
        let mut expected = ExpectedTensors::new();
        expected.insert("a", vec![2, 4]).unwrap();
        expected.insert("b", vec![4]).unwrap();
        expected
    };
    let exact = [
        spec("a", &[2, 4], GgufType::F32),
        spec("b", &[4], GgufType::F32),
    ];
    assert!(expect().check(&exact, |_| true).is_ok());
    assert!(expect().check(&exact[..1], |_| true).is_err());
    assert!(
        expect()
            .check(&[exact[0].clone(), spec("b", &[5], GgufType::F32)], |_| {
                true
            })
            .is_err()
    );
    let extra = [
        exact[0].clone(),
        exact[1].clone(),
        spec("c", &[1], GgufType::F32),
    ];
    assert!(expect().check(&extra, |_| true).is_err());
    assert!(expect().check(&exact, |_| false).is_err());
    assert!(expect().insert("a", vec![1]).is_err());
}

#[test]
fn maps_tensor_types_to_weight_encodings() {
    for (ty, format) in [
        (GgufType::Q4_0, GroupedIntFormat::Q4_0),
        (GgufType::Q4_1, GroupedIntFormat::Q4_1),
        (GgufType::Q5_0, GroupedIntFormat::Q5_0),
        (GgufType::Q5_1, GroupedIntFormat::Q5_1),
        (GgufType::Q8_0, GroupedIntFormat::Q8_0),
        (GgufType::Q4K, GroupedIntFormat::Q4K),
        (GgufType::Q5K, GroupedIntFormat::Q5K),
        (GgufType::Q6K, GroupedIntFormat::Q6K),
    ] {
        assert_eq!(
            encoding_for(ty).unwrap(),
            ExecutableEncoding::GroupedInt(format)
        );
    }
    for ty in [GgufType::F32, GgufType::F16, GgufType::Bf16] {
        assert_eq!(encoding_for(ty).unwrap(), ExecutableEncoding::Dense);
    }
    assert!(encoding_for(GgufType::Q2K).is_err());
    assert!(encoding_for(GgufType::Q8_1).is_err());
}
