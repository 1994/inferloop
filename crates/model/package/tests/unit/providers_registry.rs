use super::*;
use infer_core::{ModelId, ProviderId};
use infer_ir::{
    BackboneKind, CapabilityRequirements, DType, FeedForward, Head, Mixer, Modality, ModelIr,
    PositionSpec, PrecisionPlan,
};
use infer_spi::{FusionPlan, ImportedModel, ModalityPlan, PrecisionPolicy, SpeculationPlan};

/// Provider that claims one architecture and imports nothing.
struct Stub;

impl ModelProvider for Stub {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: ProviderId::ONE,
            name: "stub".into(),
            spi_version: infer_spi::SPI_VERSION,
        }
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["stub-arch"]
    }
    fn import(&self, _id: ModelId, _config: &[u8]) -> Result<ImportedModel> {
        Err(Error::unsupported("stub imports nothing"))
    }
}

/// An omni family declares every customization in one `import` result.
struct Omni;

fn omni_model(id: ModelId) -> ModelIr {
    ModelIr {
        id,
        backbone: BackboneKind::Decoder,
        vocab_size: 8,
        hidden_size: 4,
        max_sequence: 8,
        mixers: vec![Mixer::Attention {
            query_heads: 1,
            kv_heads: 1,
            head_dim: 4,
            sliding_window: None,
            output_gate: false,
            qk_norm: false,
        }],
        feed_forward: FeedForward::Dense { intermediate: 8 },
        position: PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-6,
        norm_weight_offset: 0.0,
        heads: vec![Head::LanguageModel],
        modalities: vec![
            Modality::Text,
            Modality::Image,
            Modality::Audio,
            Modality::Video,
        ],
        state: vec![],
        tied_embeddings: false,
    }
}

impl ModelProvider for Omni {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            id: ProviderId::new(2).unwrap_or(ProviderId::ONE),
            name: "omni-hf-config".into(),
            spi_version: infer_spi::SPI_VERSION,
        }
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["omni3", "Omni3ForConditionalGeneration"]
    }
    fn import(&self, id: ModelId, _config: &[u8]) -> Result<ImportedModel> {
        let mut imported = ImportedModel::new(omni_model(id), 1);
        imported.requirements = CapabilityRequirements {
            compute_dtypes: vec![DType::Bf16],
            ..Default::default()
        };
        imported.precision =
            PrecisionPolicy::Preferred(vec![PrecisionPlan::nvfp4_block(), PrecisionPlan::bf16()]);
        imported.speculation = Some(SpeculationPlan {
            prefix: "mtp.".into(),
            layers: 1,
            fusion: Some(FusionPlan {
                projection: "fc.weight".into(),
                norms: [
                    "pre_fc_norm_embedding.weight".into(),
                    "pre_fc_norm_hidden.weight".into(),
                ],
            }),
        });
        imported.modalities = vec![
            ModalityPlan {
                modality: Modality::Image,
                placeholder_tokens: vec![100],
                encoder: None,
            },
            ModalityPlan {
                modality: Modality::Audio,
                placeholder_tokens: vec![200],
                encoder: None,
            },
            ModalityPlan {
                modality: Modality::Video,
                placeholder_tokens: vec![300],
                encoder: None,
            },
        ];
        Ok(imported)
    }
}

#[test]
fn an_omni_family_converges_every_customization_into_one_import() -> Result<()> {
    let mut registry = ModelRegistry::new();
    registry.register(Arc::new(Omni))?;
    // Resolution uses the checkpoint's own architecture name; no core dispatch changed.
    let provider = registry.resolve(br#"{"architectures":["Omni3ForConditionalGeneration"]}"#)?;
    let imported = provider.import(ModelId::ONE, b"{}")?;

    assert_eq!(imported.modalities.len(), 3);
    assert_eq!(
        imported
            .speculation
            .as_ref()
            .map(|plan| plan.prefix.as_str()),
        Some("mtp.")
    );
    assert_eq!(imported.requirements.compute_dtypes, vec![DType::Bf16]);
    // Precision is a device decision: native FP4 where available, BF16 elsewhere.
    assert_eq!(
        imported
            .precision
            .resolve(|dtype| dtype == DType::Fp4E2M1)
            .storage,
        DType::Fp4E2M1
    );
    assert_eq!(
        imported
            .precision
            .resolve(|dtype| dtype == DType::Bf16)
            .storage,
        DType::Bf16
    );
    Ok(())
}

#[test]
fn builtin_registry_resolves_the_shipped_configuration() -> Result<()> {
    let provider = default_registry().resolve(include_bytes!(
        "../../../../../examples/qwen3.8-27b/config.json"
    ))?;
    assert!(provider.architectures().contains(&"qwen3_5"));
    Ok(())
}

#[test]
fn unknown_architecture_names_the_extension_point() {
    let registry = ModelRegistry::builtin();
    let Err(error) = registry.resolve(br#"{"model_type":"not-a-real-family"}"#) else {
        panic!("unknown architecture must not resolve");
    };
    assert!(error.to_string().contains("ModelProvider"), "{error}");
}

#[test]
fn duplicate_architecture_registration_is_rejected() -> Result<()> {
    let mut registry = ModelRegistry::new();
    registry.register(Arc::new(Stub))?;
    assert!(registry.register(Arc::new(Stub)).is_err());
    assert_eq!(registry.providers().len(), 1);
    Ok(())
}

#[test]
fn nested_text_backbone_hints_are_collected() -> Result<()> {
    let mut registry = ModelRegistry::new();
    registry.register(Arc::new(Stub))?;
    let resolved = registry.resolve(br#"{"text_config":{"model_type":"stub-arch"}}"#)?;
    assert_eq!(resolved.metadata().name, "stub");
    Ok(())
}

#[test]
fn unclaimed_architecture_is_unsupported_not_invalid() {
    let registry = ModelRegistry::new();
    let Err(error) = registry.resolve(br#"{"model_type":"stub-arch"}"#) else {
        panic!("empty registry resolves nothing");
    };
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
}

#[test]
fn text_assets_use_the_selected_provider_generation_policy() {
    use infer_spi::{GenerationPolicy, GenerationPreset};
    use std::env::temp_dir;
    use std::fs::{copy, create_dir_all, remove_dir_all, write};
    use std::path::Path;
    use std::process::id;

    struct SamplingProvider;
    impl ModelProvider for SamplingProvider {
        fn metadata(&self) -> ProviderMetadata {
            ProviderMetadata {
                id: ProviderId::ONE,
                name: "sampling-stub".into(),
                spi_version: infer_spi::SPI_VERSION,
            }
        }
        fn architectures(&self) -> &'static [&'static str] {
            &["sampling-stub"]
        }
        fn import(&self, _id: ModelId, _config: &[u8]) -> Result<ImportedModel> {
            Err(Error::unsupported("not needed for this test"))
        }
        fn generation_policy(&self, _card: Option<&str>) -> GenerationPolicy {
            GenerationPolicy {
                enable_thinking: true,
                non_thinking: Some(GenerationPreset {
                    temperature: Some(0.4),
                    top_p: Some(0.75),
                    top_k: Some(3),
                    source: "custom provider",
                    ..GenerationPreset::default()
                }),
                ..GenerationPolicy::default()
            }
        }
    }
    let root = temp_dir().join(format!("infer-provider-sampling-{}", id()));
    create_dir_all(&root).unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    copy(fixture.join("tokenizer.json"), root.join("tokenizer.json")).unwrap();
    write(
        root.join("config.json"),
        r#"{"model_type":"sampling-stub"}"#,
    )
    .unwrap();
    write(
        root.join("generation_config.json"),
        r#"{"do_sample":true,"temperature":0.9}"#,
    )
    .unwrap();
    let mut registry = ModelRegistry::new();
    registry.register(Arc::new(SamplingProvider)).unwrap();
    let assets = crate::TextAssets::open_with(&registry, &root, 128).unwrap();
    let default = assets
        .generation
        .resolve(&crate::SamplingOverrides::default())
        .unwrap();
    assert!(default.enable_thinking);
    assert_eq!(default.sampling.temperature, 0.9);
    let direct = assets
        .generation
        .resolve(&crate::SamplingOverrides {
            enable_thinking: Some(false),
            top_p: Some(0.5),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(direct.sampling.temperature, 0.4);
    assert_eq!(direct.sampling.top_p, 0.5);
    assert_eq!(direct.sampling.top_k, Some(3));
    assert_eq!(direct.sources["temperature"], "custom provider");
    assert_eq!(direct.sources["top_p"], "request override");
    let generic = crate::TextAssets::open(&root, 128).unwrap();
    assert!(
        !generic
            .generation
            .resolve(&crate::SamplingOverrides::default())
            .unwrap()
            .enable_thinking
    );
    remove_dir_all(root).unwrap();
}
