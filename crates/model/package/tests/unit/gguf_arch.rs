use super::*;
use infer_format::{Metadata, MetadataValue};

fn metadata(entries: &[(&str, MetadataValue)]) -> Metadata {
    Metadata::from_entries(
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone())),
    )
}

#[test]
fn trailing_nextn_blocks_become_the_draft_head() {
    let file = metadata(&[
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
    let plain = metadata(&[("demo.block_count", MetadataValue::U64(4))]);
    assert_eq!(LayerPlan::of(&plain, "demo").unwrap().draft, 0);
    let only = metadata(&[
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
fn shared_block_names_and_unknown_architectures() {
    assert_eq!(
        common_block_field("attn_q.weight"),
        Some("self_attn.q_proj.weight")
    );
    assert_eq!(
        common_block_field("post_attention_norm.weight"),
        Some("post_attention_layernorm.weight")
    );
    assert_eq!(common_block_field("ssm_a"), None);
    let unknown = metadata(&[("general.architecture", MetadataValue::String("demo".into()))]);
    let error = architecture(&unknown)
        .map(GgufArchitecture::name)
        .unwrap_err();
    assert!(error.message.contains("qwen3vl") && error.message.contains("qwen35"));
    assert_eq!(trim_trailing_zeros(vec![11, 11, 10, 0]), [11, 11, 10]);
}
