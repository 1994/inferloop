//! Opening GGUF files: adapter resolution and mmproj discovery.
use super::support::{container, metadata, scratch, strings, with};
use infer_core::ErrorCode;
use infer_format::MetadataValue;
use infer_models::GgufPackage;
use std::fs::{create_dir_all, write};
use std::os::unix::fs::symlink;

/// A complete `dflash` draft declaration with no tensors: the smallest file every platform stage
/// accepts, and an architecture that takes no mmproj.
fn draft() -> Vec<u8> {
    let u = MetadataValue::U64;
    let key = |name: &str| format!("dflash.{name}");
    let mut entries: Vec<(String, MetadataValue)> = [
        ("block_count", u(2)),
        ("context_length", u(4096)),
        ("embedding_length", u(64)),
        ("feed_forward_length", u(128)),
        ("attention.head_count", u(4)),
        ("attention.head_count_kv", u(2)),
        ("attention.key_length", u(16)),
        ("attention.sliding_window", u(256)),
        ("block_size", u(8)),
        ("conv_kernel_size", u(2)),
        ("conv_group_size", u(16)),
        ("selector_rank", u(8)),
        ("selector_top_k", u(4)),
    ]
    .into_iter()
    .map(|(name, value)| (key(name), value))
    .collect();
    entries.extend([
        (key("attention.causal"), MetadataValue::Bool(false)),
        (
            key("attention.sliding_window_pattern"),
            MetadataValue::Array(vec![MetadataValue::Bool(true); 2]),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            MetadataValue::F32(1e-6),
        ),
        (key("rope.freq_base"), MetadataValue::F32(10_000.0)),
        (key("target_layers"), MetadataValue::Array(vec![u(2), u(5)])),
    ]);
    let mut file = metadata(vec![
        (
            "general.architecture",
            MetadataValue::String("dflash".into()),
        ),
        ("tokenizer.ggml.model", MetadataValue::String("gpt2".into())),
        ("tokenizer.ggml.pre", MetadataValue::String("qwen2".into())),
        ("tokenizer.ggml.tokens", strings(&["a", "b", "ab", "<m>"])),
        ("tokenizer.ggml.merges", strings(&["a b"])),
        (
            "tokenizer.ggml.token_type",
            MetadataValue::Array(vec![u(1), u(1), u(1), u(3)]),
        ),
        ("tokenizer.ggml.mask_token_id", u(3)),
        ("tokenizer.ggml.eos_token_id", u(3)),
        ("tokenizer.ggml.padding_token_id", u(3)),
        (
            "tokenizer.chat_template",
            MetadataValue::String("{{ messages }}".into()),
        ),
    ]);
    for (name, value) in entries {
        file = with(&file, &name, Some(value));
    }
    container(&file)
}

#[test]
fn a_file_opens_through_its_adapter() {
    let root = scratch();
    let path = root.join("draft.gguf");
    write(&path, draft()).unwrap();
    let package = GgufPackage::open(&path).unwrap();
    assert_eq!(package.architecture(), "dflash");
    assert!(package.slots().is_empty());
    let config: serde_json::Value = serde_json::from_slice(package.config()).unwrap();
    assert_eq!(
        config["dflash_config"]["target_layer_ids"],
        serde_json::json!([1, 4])
    );
}

#[test]
fn an_architecture_without_an_adapter_is_unsupported() {
    let root = scratch();
    let path = root.join("other.gguf");
    let file = metadata(vec![(
        "general.architecture",
        MetadataValue::String("other".into()),
    )]);
    write(&path, container(&file)).unwrap();
    let error = GgufPackage::open(&path).map(|_| ()).unwrap_err();
    assert_eq!(error.code, ErrorCode::Unsupported);
}

#[test]
fn the_mmproj_is_found_beside_a_linked_model_not_its_target() {
    let root = scratch();
    let (store, deployment) = (root.join("store"), root.join("deployment"));
    create_dir_all(&store).unwrap();
    create_dir_all(&deployment).unwrap();
    write(store.join("model.gguf"), draft()).unwrap();
    let linked = deployment.join("model.gguf");
    symlink(store.join("model.gguf"), &linked).unwrap();
    // The draft takes no mmproj, so finding the sibling is what makes opening fail.
    write(deployment.join("mmproj-model-f16.gguf"), b"").unwrap();
    let error = GgufPackage::open_auto(&linked).map(|_| ()).unwrap_err();
    assert!(
        error.message.contains("takes no mmproj"),
        "{}",
        error.message
    );
    assert!(GgufPackage::open_auto(store.join("model.gguf")).is_ok());
}
