use crate::constants::{CONFIG_MAX_BYTES, MIB_U64};
use crate::{PackageProbe, PackageSource, TensorSet, TextAssets, default_registry};
use infer_core::{ErrorCode, ModelId};
use std::env::temp_dir;
use std::fs::{create_dir_all, read, remove_dir_all, remove_file, write};
use std::path::PathBuf;
use std::process::id;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn temp() -> PathBuf {
    let path = temp_dir().join(format!(
        "infer-source-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&path).unwrap();
    path
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny")
}

#[test]
fn probe_detects_the_provider_and_text_assets() {
    let probe = PackageProbe::open(fixture(), ModelId::ONE).unwrap();
    assert!(probe.has_text_assets());
    assert_ne!(probe.imported().model.mixers.len(), 0);
    assert_ne!(probe.provider().metadata().name, "");
}

#[test]
fn missing_assets_keep_their_error_kind_and_oversized_metadata_is_rejected() {
    let root = temp();
    let source = PackageSource::open(&root).unwrap();
    assert!(!source.has_text_assets());
    assert_eq!(
        source.require_config().unwrap_err().code,
        ErrorCode::InvalidInput
    );
    assert!(matches!(
        TextAssets::from_source(default_registry(), &source, 64),
        Err(error) if error.code == ErrorCode::InvalidInput
    ));
    assert!(
        TensorSet::open_optional(&root, "readouts.safetensors")
            .unwrap()
            .is_none()
    );
    write(
        root.join("config.json"),
        vec![b' '; usize::try_from(CONFIG_MAX_BYTES).unwrap() + 1],
    )
    .unwrap();
    let source = PackageSource::open(&root).unwrap();
    assert_eq!(
        source
            .require_config_bounded(CONFIG_MAX_BYTES)
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    write(
        root.join("config.json"),
        vec![b' '; usize::try_from(16 * MIB_U64).unwrap() + 1],
    )
    .unwrap();
    assert_eq!(
        PackageSource::open(&root).unwrap_err().code,
        ErrorCode::Capacity
    );
    remove_dir_all(root).unwrap();
}

#[test]
fn probe_rejects_missing_config_and_unregistered_architectures() {
    let root = temp();
    assert!(matches!(
        PackageProbe::open(&root, ModelId::ONE),
        Err(error) if error.code == ErrorCode::InvalidInput
    ));
    write(
        root.join("config.json"),
        r#"{"model_type":"not-a-registered-model"}"#,
    )
    .unwrap();
    assert!(matches!(
        PackageProbe::open(&root, ModelId::ONE),
        Err(error) if error.code == ErrorCode::Unsupported
    ));
    remove_dir_all(root).unwrap();
}

#[test]
fn an_empty_text_budget_is_rejected_before_any_asset_read() {
    let root = temp();
    assert!(matches!(
        TextAssets::open(&root, 0),
        Err(error) if error.code == ErrorCode::InvalidInput
    ));
    remove_dir_all(root).unwrap();
}

#[test]
fn chat_template_precedence_and_rejections_match_package_rules() {
    let root = temp();
    write(
        root.join("tokenizer_config.json"),
        r#"{"chat_template":"embedded"}"#,
    )
    .unwrap();
    let source = PackageSource::open(&root).unwrap();
    assert_eq!(source.chat_template().unwrap().as_deref(), Some("embedded"));

    write(root.join("chat_template.jinja"), "jinja {{ messages }}").unwrap();
    let source = PackageSource::open(&root).unwrap();
    assert_eq!(
        source.chat_template().unwrap().as_deref(),
        Some("jinja {{ messages }}")
    );

    remove_file(root.join("chat_template.jinja")).unwrap();
    write(
        root.join("tokenizer_config.json"),
        r#"{"chat_template":[{"name":"named","template":"x"}]}"#,
    )
    .unwrap();
    let source = PackageSource::open(&root).unwrap();
    assert_eq!(
        source.chat_template().unwrap_err().code,
        ErrorCode::Unsupported
    );

    write(root.join("tokenizer_config.json"), r#"{"chat_template":3}"#).unwrap();
    let source = PackageSource::open(&root).unwrap();
    assert_eq!(
        source.chat_template().unwrap_err().code,
        ErrorCode::InvalidInput
    );

    write(
        root.join("tokenizer_config.json"),
        r#"{"chat_template":[{"name":"default","template":"chosen"}]}"#,
    )
    .unwrap();
    let source = PackageSource::open(&root).unwrap();
    assert_eq!(source.chat_template().unwrap().as_deref(), Some("chosen"));
    remove_dir_all(root).unwrap();
}

#[test]
fn text_assets_share_the_generation_defaults_of_their_source() {
    let root = temp();
    write(
        root.join("tokenizer.json"),
        read(fixture().join("tokenizer.json")).unwrap(),
    )
    .unwrap();
    write(
        root.join("config.json"),
        r#"{"text_config":{"eos_token_id":31}}"#,
    )
    .unwrap();
    write(
        root.join("generation_config.json"),
        r#"{"do_sample":false}"#,
    )
    .unwrap();
    let source = PackageSource::open(&root).unwrap();
    let assets = TextAssets::from_source(default_registry(), &source, 128).unwrap();
    let defaults = assets
        .generation
        .resolve(&crate::SamplingOverrides::default())
        .unwrap();
    assert_eq!(defaults.sampling.temperature, 0.0);
    assert_eq!(defaults.sampling.eos_tokens, [31]);
    assert_eq!(
        source.require_config().unwrap(),
        br#"{"text_config":{"eos_token_id":31}}"#.as_slice()
    );
    remove_dir_all(root).unwrap();
}
