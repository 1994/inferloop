use super::*;
use infer_core::{ModelId, Result};
use infer_spi::{DraftHead, FusionPlan};
use std::path::Path;

/// Only metadata is needed: these tests never upload or read a draft payload.
fn package_with_head() -> Result<QuantizedPackage> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut package = QuantizedPackage::open(root, ModelId::ONE)?;
    package.imported.speculation = Some(DraftHead {
        prefix: "mtp.".into(),
        layers: 1,
        fusion: Some(FusionPlan {
            projection: "fc.weight".into(),
            norms: ["embedding_norm.weight".into(), "hidden_norm.weight".into()],
        }),
    });
    let (_, graph) = package.provider.draft_graph(&package.imported.model)?;
    let template = package.weights["embed_tokens.weight"].clone();
    for tensor in graph.tensors {
        let TensorStorage::Weight { slot } = tensor.storage else {
            continue;
        };
        if matches!(slot.as_str(), "embed_tokens.weight" | "lm_head.weight") {
            continue;
        }
        let mut weight = template.clone();
        weight.shape = tensor.shape;
        package.mtp.insert(format!("mtp.{slot}"), weight);
    }
    let hidden = package.imported.model.hidden_size;
    for (slot, shape) in [
        ("fc.weight", vec![hidden, hidden * 2]),
        ("embedding_norm.weight", vec![hidden]),
        ("hidden_norm.weight", vec![hidden]),
    ] {
        let mut weight = template.clone();
        weight.shape = shape;
        package.mtp.insert(format!("mtp.{slot}"), weight);
    }
    Ok(package)
}

#[test]
fn automatic_mtp_requires_complete_supported_provider_bindings() -> Result<()> {
    let mut package = package_with_head()?;
    assert_eq!(resolve(SpeculativeTokens::Depth(0), None, &package), 0);
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 2);
    assert_eq!(resolve(SpeculativeTokens::Auto, Some(2048), &package), 2);
    assert_eq!(resolve(SpeculativeTokens::Auto, Some(0), &package), 2);
    assert_eq!(resolve(SpeculativeTokens::Auto, Some(1), &package), 0);
    assert_eq!(resolve(SpeculativeTokens::Auto, Some(2), &package), 0);
    // Missing or mismatched fusion weights cannot make automatic mode fail a target load.
    let original = package.mtp.remove("mtp.fc.weight").unwrap();
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    let mut wrong = original.clone();
    wrong.shape[0] += 1;
    package.mtp.insert("mtp.fc.weight".into(), wrong);
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    package.mtp.insert("mtp.fc.weight".into(), original);
    package
        .mtp
        .remove("mtp.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    Ok(())
}

#[test]
fn unsupported_heads_disable_only_automatic_speculation() -> Result<()> {
    let mut package = package_with_head()?;
    package
        .imported
        .speculation
        .as_mut()
        .unwrap()
        .prefix
        .clear();
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    package = package_with_head()?;
    package.imported.speculation.as_mut().unwrap().fusion = None;
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    package = package_with_head()?;
    package.weights.remove("lm_head.weight");
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    package = package_with_head()?;
    package.imported.speculation.as_mut().unwrap().layers = 2;
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    package.imported.speculation = None;
    assert_eq!(resolve(SpeculativeTokens::Auto, None, &package), 0);
    assert_eq!(resolve(SpeculativeTokens::Depth(0), None, &package), 0);
    assert_eq!(resolve(SpeculativeTokens::Depth(2), None, &package), 2);
    // Explicit settings still reach the loader's existing errors instead of being hidden.
    assert_eq!(resolve(SpeculativeTokens::Depth(8), Some(1), &package), 8);
    Ok(())
}
