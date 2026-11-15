//! Models commands.
use super::{InspectModelOptions, InspectPackageOptions, TokenizeOptions, print, read_json};
use crate::constants::GIB_U64;
use crate::constants::MIB;
use crate::constants::MIB_U64;
use infer_core::{Error, ModelId, Result};
use infer_models::{ModelPackage, SafetensorsIndex, default_registry, memory_estimate};
use std::fs::read;

/// Conservative expansion factor from native weight storage to F32, applied to the staging
/// budget so an expanded weight chunk still fits its staging allocation.
const F32_EXPANSION_FACTOR: usize = 3;
/// Maximum weight bytes read into staging per load chunk.
const MAX_WEIGHT_CHUNK_BYTES: usize = 4 * MIB;

pub fn inspect_model(options: InspectModelOptions) -> Result<()> {
    let InspectModelOptions {
        config,
        index,
        context_tokens,
        sequences,
        device_memory_gib,
    } = options;
    let bytes = read(config).map_err(|e| Error::invalid(e.to_string()))?;
    let provider = default_registry().resolve(&bytes)?;
    let model = provider.import(ModelId::new(1)?, &bytes)?;
    let index: Option<SafetensorsIndex> = index
        .map(|p| {
            read(p)
                .map_err(|e| Error::invalid(e.to_string()))
                .and_then(|b| SafetensorsIndex::parse(&b))
        })
        .transpose()?;
    let bytes = device_memory_gib
        .checked_mul(GIB_U64)
        .ok_or_else(|| Error::invalid("device memory overflow"))?;
    let memory = index
        .as_ref()
        .map(|i| {
            memory_estimate(
                &model.model,
                i.declared_weight_bytes()?.unwrap_or(0),
                context_tokens,
                sequences,
                GIB_U64,
                bytes,
            )
        })
        .transpose()?;
    print(
        &serde_json::json!({"model":model,"memory":memory,"weight_shards":index.map(|i|i.shards()),"execution_supported":false,"next_requirement":"complete Qwen3.8 weight package and RTX 5090 CUDA/quantized execution validation"}),
    )?;

    Ok(())
}
pub fn inspect_package(options: InspectPackageOptions) -> Result<()> {
    let package = ModelPackage::open(options.package, ModelId::new(1)?)?;
    let budget = options
        .host_memory_mib
        .map(|n| {
            n.checked_mul(MIB_U64)
                .ok_or_else(|| Error::invalid("weight memory budget overflow"))
        })
        .transpose()?
        .unwrap_or(u64::MAX);
    let staging = options
        .staging_memory_mib
        .checked_mul(MIB)
        .ok_or_else(|| Error::invalid("weight staging budget overflow"))?;
    let plan = infer_models::WeightLoadPlan::build(
        &package,
        infer_models::LoadOptions {
            storage: if options.weights_f32 {
                infer_models::WeightStorage::F32
            } else {
                infer_models::WeightStorage::Native
            },
            resident_budget_bytes: budget,
            staging_budget_bytes: staging,
            chunk_bytes: (staging
                / if options.weights_f32 {
                    F32_EXPANSION_FACTOR
                } else {
                    1
                })
            .min(MAX_WEIGHT_CHUNK_BYTES),
        },
    )?;
    print(
        &serde_json::json!({"manifest":package.manifest,"program":package.graph,"weight_load":plan,"scope":"metadata preflight; weight residency excludes execution scratch, KV, and request state; vision/MTP excluded"}),
    )?;

    Ok(())
}
pub fn tokenize(options: TokenizeOptions) -> Result<()> {
    let TokenizeOptions {
        package,
        text,
        messages,
        enable_thinking,
    } = options;
    let probe = infer_models::PackageProbe::open(package, ModelId::new(1)?)?;
    let assets = infer_models::TextAssets::from_source(
        default_registry(),
        probe.source(),
        probe.imported().model.max_sequence,
    )?;
    let (rendered, tokens) = if let Some(text) = text {
        (text.clone(), assets.encode(&text, true)?)
    } else {
        let messages: Vec<infer_models::ChatMessage> = read_json(
            messages
                .as_deref()
                .ok_or_else(|| Error::invariant("validated argument"))?,
        )?;
        let rendered = assets.render_chat(
            &messages,
            &infer_models::ChatOptions {
                enable_thinking,
                ..Default::default()
            },
        )?;
        let tokens = assets.encode(&rendered, false)?;
        (rendered, tokens)
    };
    print(
        &serde_json::json!({"rendered":rendered,"tokens":tokens,"tokenizer_fingerprint":assets.fingerprint}),
    )?;

    Ok(())
}
