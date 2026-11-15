use super::{tokenizer_config, tokenizer_json};
use infer_gguf::{Metadata, MetadataValue};

fn metadata() -> Metadata {
    Metadata::from_entries(vec![
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
