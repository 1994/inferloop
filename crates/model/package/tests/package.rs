use infer_core::*;
use infer_ir::*;
use infer_model_recipes::decoder::lower;
use infer_models::*;
use std::env::temp_dir;
use std::env::var;
use std::fs::create_dir_all;
use std::fs::read;
use std::fs::remove_dir_all;
use std::fs::remove_file;
use std::fs::write;
use std::process::id;
use std::{collections::BTreeMap, path::PathBuf, sync::atomic::AtomicU64, sync::atomic::Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn temp() -> PathBuf {
    let p = temp_dir().join(format!(
        "infer-tensor-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&p).unwrap();
    p
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn raw(header: &str, data: &[u8]) -> PathBuf {
    let root = temp();
    let p = root.join("model.safetensors");
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend(header.as_bytes());
    b.extend(data);
    write(&p, b).unwrap();
    p
}
#[test]
fn safetensors_float_conversion_and_row_read_are_exact() {
    let bf16 = raw(
        r#"{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#,
        &[0x80, 0x3f, 0, 0xc0, 0, 0x3f, 0, 0],
    );
    let mut file = SafetensorsFile::open(&bf16).unwrap();
    assert_eq!(
        file.read_f32("w", 1024).unwrap().data,
        [1.0, -2.0, 0.5, 0.0]
    );
    assert_eq!(file.read_row_f32("w", 1, 1024).unwrap(), [0.5, 0.0]);
    assert_eq!(file.read_f32("w", 1).unwrap_err().code, ErrorCode::Capacity);
    let f16 = raw(
        r#"{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
        &[0, 0x3c, 0, 0xc0, 1, 0, 0, 0x80],
    );
    let out = SafetensorsFile::open(&f16)
        .unwrap()
        .read_f32("w", 1024)
        .unwrap();
    assert_eq!(out.data, [1.0, -2.0, 2.0f32.powi(-24), -0.0]);
    remove_dir_all(bf16.parent().unwrap()).unwrap();
    remove_dir_all(f16.parent().unwrap()).unwrap();
}
#[test]
fn malformed_header_bounds_duplicates_and_nonfinite_payload_are_rejected() {
    for header in [
        r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#,
    ] {
        let p = raw(header, &[0; 8]);
        assert!(SafetensorsFile::open(&p).is_err());
        remove_dir_all(p.parent().unwrap()).unwrap();
    }
    let p = raw(
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &f32::NAN.to_le_bytes(),
    );
    assert!(
        SafetensorsFile::open(&p)
            .unwrap()
            .read_f32("w", 1024)
            .is_err()
    );
    remove_dir_all(p.parent().unwrap()).unwrap();
}
#[test]
fn host_package_binds_payload_and_respects_budget() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut package = ModelPackage::open(&root, ModelId::new(1).unwrap()).unwrap();
    assert!(
        package
            .manifest
            .bindings
            .iter()
            .any(|b| b.source.contains("A_log"))
    );
    assert!(
        package
            .manifest
            .bindings
            .iter()
            .any(|b| b.shape == vec![16, 8])
    );
    assert_eq!(
        package.load_host_weights(1).unwrap_err().code,
        ErrorCode::Capacity
    );
    let loaded = package.load_host_weights(1024 * 1024).unwrap();
    assert_eq!(loaded.len(), package.manifest.bindings.len());
    assert!(loaded.values().all(|t| t.validate().is_ok()));
}
#[test]
fn qwen38_required_bindings_exist_in_pinned_official_index() {
    let config = include_bytes!("../../../../examples/qwen3.8-27b/config.json");
    let model = QwenProvider
        .import_manifest(ModelId::new(1).unwrap(), config)
        .unwrap()
        .model;
    let index = SafetensorsIndex::parse(include_bytes!(
        "../../../../examples/qwen3.8-27b/model.safetensors.index.json"
    ))
    .unwrap();
    let graph = lower(&model).unwrap();
    let expected: BTreeMap<_, _> = graph
        .tensors
        .iter()
        .filter_map(|s| match &s.storage {
            TensorStorage::Weight { slot } => Some((
                if slot == "lm_head.weight" {
                    slot.clone()
                } else {
                    format!("model.language_model.{slot}")
                },
                s.shape.clone(),
            )),
            _ => None,
        })
        .collect();
    for name in expected.keys() {
        assert!(
            index.weight_map.contains_key(name),
            "missing official binding {name}"
        );
    }

    assert_eq!(
        expected.len(),
        index
            .weight_map
            .keys()
            .filter(|s| s.starts_with("model.language_model.") || s.as_str() == "lm_head.weight")
            .count(),
        "every backbone weight must be consumed"
    );
}
#[test]
fn official_qwen_text_parity_when_assets_are_supplied() {
    let Ok(root) = var("INFER_QWEN_TEXT_PACKAGE") else {
        return;
    };
    let assets = TextAssets::open(root, 262_144).unwrap();
    let golden: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../examples/qwen3.8-27b/text-golden.json"
    ))
    .unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone()).unwrap();
        let options = ChatOptions {
            enable_thinking: case["enable_thinking"].as_bool().unwrap(),
            ..Default::default()
        };
        let rendered = assets.render_chat(&messages, &options).unwrap();
        assert_eq!(rendered, case["rendered"].as_str().unwrap());
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        assert_eq!(assets.encode_chat(&messages, &options).unwrap(), tokens);
    }
}

#[test]
fn the_fixture_tokenizer_refuses_streaming_decode() {
    // Its decoder joins tokens with a separator, so piecewise decoding would not match.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let assets = TextAssets::open(&root, 4096).unwrap();
    assert!(!assets.supports_streaming_decode());
}

#[expect(
    clippy::unwrap_used,
    reason = "Test fixtures fail the test immediately on unexpected setup or runtime output"
)]
/// Write a copy of the fixture tokenizer with a byte-level decoder, which is what the served
/// packages use and the only decoder that decomposes per token.
fn byte_level_fixture() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&read(root.join("tokenizer.json")).unwrap()).unwrap();
    let byte_level = serde_json::json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true
    });
    tokenizer["decoder"] = byte_level.clone();
    tokenizer["pre_tokenizer"] = byte_level;
    let package = temp();
    write(
        package.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    package
}

#[test]
fn streaming_decode_matches_one_shot_decode_token_for_token() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let ids = TextAssets::open(&fixture, 4096).unwrap();
    let assets = TextAssets::open(byte_level_fixture(), 4096).unwrap();
    assert!(assets.supports_streaming_decode());
    for text in ["hello world system assistant token8 token13", "owner", "a"] {
        let tokens = ids.encode(text, false).unwrap();
        let whole = assets.decode(&tokens, true).unwrap();
        let mut decoder = assets.stream_decoder();
        let mut streamed = String::new();
        for token in &tokens {
            streamed.push_str(&decoder.push(*token).unwrap());
        }
        streamed.push_str(&decoder.finish().unwrap());
        assert_eq!(streamed, whole, "{text:?}");
    }
}

#[test]
fn an_incremental_decode_never_publishes_a_partial_character() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let ids = TextAssets::open(&fixture, 4096).unwrap();
    let assets = TextAssets::open(byte_level_fixture(), 4096).unwrap();
    let text = "hello world system assistant token8 token13 hello world system assistant";
    let tokens = ids.encode(text, false).unwrap();
    let whole = assets.decode(&tokens, true).unwrap();
    let mut decoder = assets.stream_decoder();
    let mut streamed = String::new();
    for token in &tokens {
        let piece = decoder.push(*token).unwrap();
        // Whatever is published must be a prefix of the finished text and already final.
        assert!(
            whole.starts_with(&format!("{streamed}{piece}")),
            "{piece:?}"
        );
        streamed.push_str(&piece);
    }
    streamed.push_str(&decoder.finish().unwrap());
    assert_eq!(streamed, whole);
}

#[test]
fn streaming_decode_matches_on_the_served_package() {
    let Ok(root) = var("INFER_QWEN_TEXT_PACKAGE") else {
        return;
    };
    let assets = TextAssets::open(&root, 262_144).unwrap();
    assert!(
        assets.supports_streaming_decode(),
        "the served packages must be stream-decodable"
    );
    let golden: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../examples/qwen3.8-27b/text-golden.json"
    ))
    .unwrap();
    for case in golden["cases"].as_array().unwrap() {
        for text in [
            case["rendered"].as_str().unwrap(),
            "hello world 你好世界 🌍 token8",
        ] {
            let tokens = assets.encode(text, false).unwrap();
            let whole = assets.decode(&tokens, true).unwrap();
            let mut decoder = assets.stream_decoder();
            let mut streamed = String::new();
            for token in &tokens {
                streamed.push_str(&decoder.push(*token).unwrap());
            }
            streamed.push_str(&decoder.finish().unwrap());
            assert_eq!(streamed, whole, "{text:?}");
        }
    }
}

#[test]
fn qwen3_hybrid_generation_resolves_the_mode_and_request_overrides() {
    let root = temp();
    write(
        root.join("README.md"),
        "---\nbase_model:\n- Qwen/Qwen3-0.6B-Base\n---\n# Qwen3-0.6B\n",
    )
    .unwrap();
    write(root.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
    write(
        root.join("generation_config.json"),
        r#"{"do_sample":true,"temperature":0.6,"top_p":0.95,"top_k":20}"#,
    )
    .unwrap();
    let defaults = GenerationDefaults::open(&root).unwrap();
    let thinking = defaults.resolve(&SamplingOverrides::default()).unwrap();
    assert!(thinking.enable_thinking);
    assert_eq!(thinking.sampling.temperature, 0.6);
    assert_eq!(thinking.sampling.top_p, 0.95);
    assert_eq!(thinking.sampling.top_k, Some(20));
    let direct = defaults
        .resolve(&SamplingOverrides {
            enable_thinking: Some(false),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(direct.sampling.temperature, 0.7);
    assert_eq!(direct.sampling.top_p, 0.8);
    assert_eq!(direct.sampling.top_k, Some(20));
    let custom = defaults
        .resolve(&SamplingOverrides {
            enable_thinking: Some(false),
            temperature: Some(0.0),
            top_p: Some(0.5),
            top_k: Some(0),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(custom.sampling.temperature, 0.0);
    assert_eq!(custom.sampling.top_p, 0.5);
    assert_eq!(custom.sampling.top_k, None);
    // If the optional generation configuration is absent, use the model family recommendation.
    remove_file(root.join("generation_config.json")).unwrap();
    let fallback = GenerationDefaults::open(&root)
        .unwrap()
        .resolve(&SamplingOverrides::default())
        .unwrap();
    assert!(fallback.enable_thinking);
    assert_eq!(fallback.sampling.temperature, 0.6);
    assert_eq!(fallback.sampling.top_p, 0.95);
    assert_eq!(fallback.sampling.top_k, Some(20));

    // A package that explicitly disables sampling stays greedy on mode selection.
    write(
        root.join("generation_config.json"),
        r#"{"do_sample":false,"temperature":0.6,"top_p":0.95,"top_k":20}"#,
    )
    .unwrap();
    let greedy = GenerationDefaults::open(&root)
        .unwrap()
        .resolve(&SamplingOverrides {
            enable_thinking: Some(false),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(greedy.sampling.temperature, 0.0);

    // The architecture alone does not identify a hybrid chat model.
    write(
        root.join("README.md"),
        "---\ntags:\n- Qwen/Qwen3-0.6B-Base\n---\n",
    )
    .unwrap();
    write(
        root.join("generation_config.json"),
        r#"{"do_sample":true,"temperature":0.55,"top_p":0.93}"#,
    )
    .unwrap();
    let other = GenerationDefaults::open(&root)
        .unwrap()
        .resolve(&SamplingOverrides {
            enable_thinking: Some(false),
            ..Default::default()
        })
        .unwrap();
    assert!(!other.enable_thinking);
    assert_eq!(other.sampling.temperature, 0.55);
    assert_eq!(other.sampling.top_p, 0.93);
    remove_dir_all(root).unwrap();
}

#[test]
fn qwen38_fallback_respects_package_values_and_explicit_greedy() {
    let root = temp();
    write(
        root.join("README.md"),
        "---\nlibrary_name: transformers\n---\n# Qwen3.8-27B\n",
    )
    .unwrap();
    write(root.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
    write(
        root.join("generation_config.json"),
        r#"{"do_sample":true,"temperature":1.3,"top_k":0}"#,
    )
    .unwrap();
    let defaults = GenerationDefaults::open(&root).unwrap();
    let thinking = defaults.resolve(&SamplingOverrides::default()).unwrap();
    assert!(thinking.enable_thinking);
    assert_eq!(thinking.sampling.temperature, 1.3);
    assert_eq!(thinking.sampling.top_k, None);
    assert_eq!(thinking.sampling.top_p, 0.95);
    assert_eq!(thinking.sources["temperature"], "generation_config.json");
    let instruct = defaults
        .resolve(&SamplingOverrides {
            enable_thinking: Some(false),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(instruct.sampling.temperature, 0.7);
    assert_eq!(instruct.sampling.top_p, 0.8);
    assert_eq!(instruct.sampling.top_k, None);
    write(
        root.join("generation_config.json"),
        r#"{"do_sample":false}"#,
    )
    .unwrap();
    let greedy = GenerationDefaults::open(&root)
        .unwrap()
        .resolve(&SamplingOverrides {
            enable_thinking: Some(false),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(greedy.sampling.temperature, 0.0);
    assert_eq!(
        greedy.sources["temperature"],
        "generation_config.json do_sample=false"
    );
    remove_dir_all(root).unwrap();
}
