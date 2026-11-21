//! `tokenizer.json` and `tokenizer_config.json` from GGUF metadata.
use super::support::with;
use infer_format::{Metadata, MetadataValue};
use infer_models::gguf::{tokenizer_config, tokenizer_json};

fn metadata() -> Metadata {
    Metadata::from_entries(vec![
        (
            "general.architecture".into(),
            MetadataValue::String("qwen3vl".into()),
        ),
        (
            "tokenizer.ggml.pre".into(),
            MetadataValue::String("qwen2".into()),
        ),
        ("qwen3vl.context_length".into(), MetadataValue::U64(128)),
        (
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(vec![
                MetadataValue::String("a".into()),
                MetadataValue::String("b".into()),
                MetadataValue::String("ab".into()),
                MetadataValue::String("<|end|>".into()),
            ]),
        ),
        (
            "tokenizer.ggml.merges".into(),
            MetadataValue::Array(vec![MetadataValue::String("a b".into())]),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            MetadataValue::Array(vec![
                MetadataValue::U64(1),
                MetadataValue::U64(1),
                MetadataValue::U64(1),
                MetadataValue::U64(3),
            ]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), MetadataValue::U64(3)),
        ("tokenizer.ggml.eos_token_id".into(), MetadataValue::U64(3)),
        (
            "tokenizer.ggml.padding_token_id".into(),
            MetadataValue::U64(3),
        ),
        (
            "tokenizer.chat_template".into(),
            MetadataValue::String("{{ messages }}".into()),
        ),
    ])
}

#[test]
fn builds_a_loadable_bpe_tokenizer() {
    let bytes = tokenizer_json(&metadata()).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes).unwrap();
    assert_eq!(tokenizer.encode("ab", false).unwrap().get_ids(), [2]);
    assert_eq!(tokenizer.token_to_id("<|end|>"), Some(3));
}

#[test]
fn writes_the_special_tokens_and_template() {
    let bytes = tokenizer_config(&metadata()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["bos_token"], "<|end|>");
    assert_eq!(value["chat_template"], "{{ messages }}");
    assert_eq!(value["model_max_length"], 128);
}

#[test]
fn the_split_follows_the_declared_pre_tokenizer() {
    let qwen35 = with(
        &metadata(),
        "tokenizer.ggml.pre",
        Some(MetadataValue::String("qwen35".into())),
    );
    let value: serde_json::Value =
        serde_json::from_slice(&tokenizer_json(&qwen35).unwrap()).unwrap();
    let pattern = value["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        .as_str()
        .unwrap();
    assert!(pattern.contains("[\\p{L}\\p{M}]+"), "{pattern}");
    let unknown = with(
        &metadata(),
        "tokenizer.ggml.pre",
        Some(MetadataValue::String("other".into())),
    );
    assert!(tokenizer_json(&unknown).is_err());
}

#[test]
fn a_family_without_bos_writes_none() {
    let without = with(&metadata(), "tokenizer.ggml.bos_token_id", None);
    let value: serde_json::Value =
        serde_json::from_slice(&tokenizer_config(&without).unwrap()).unwrap();
    assert!(value["bos_token"].is_null());
}
