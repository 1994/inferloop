//! Full reduced-model parity against independent Transformers prefix goldens.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use std::env::args;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
use std::fs::read;
#[cfg(cuda_backend)]
use std::path::Path;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use infer_backend_cuda::{
        device::CudaDevice,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{Error, ModelId};
    let root = args()
        .nth(1)
        .ok_or_else(|| Error::invalid("model path required"))?;
    let decode_only = args().any(|arg| arg == "--decode-only");
    let verify_only = args().any(|arg| arg == "--verify-only");
    if decode_only && verify_only {
        return Err(Error::invalid("choose only one check mode").into());
    }
    let golden: Golden = serde_json::from_slice(&read(Path::new(&root).join("golden.json"))?)?;
    reference_graph(&root, &golden)?;
    enable_default()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        &root,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            prefill_width: if decode_only || verify_only { 1 } else { 32 },
            verification_width: if decode_only { 0 } else { 3 },
            ..LoadOptions::default()
        },
    )?;
    let mut scalar = loaded.sequence(64)?;
    let mut prompt = loaded.sequence(64)?;
    let mut verify = loaded.sequence(64)?;
    let mut max_error = 0.0f32;
    for prefix in &golden.prefixes {
        scalar.reset()?;
        let mut result = (Vec::new(), Vec::new());
        for (position, token) in prefix.tokens.iter().enumerate() {
            result = scalar.step(*token, position, position, None, true)?;
        }
        max_error = max_error.max(compare(
            &result.0,
            prefix
                .hidden
                .last()
                .ok_or_else(|| Error::invalid("golden hidden"))?,
        )?);
        max_error = max_error.max(compare(&result.1, &prefix.logits)?);
        if decode_only || verify_only {
            continue;
        }
        prompt.reset()?;
        let rows = prompt.prefill_batch(&prefix.tokens, 0, true)?;
        let result = rows
            .last()
            .ok_or_else(|| Error::invalid("prefill result"))?;
        max_error = max_error.max(compare(&result.1, &prefix.logits)?);
        max_error = max_error.max(compare(
            &result.0,
            prefix
                .hidden
                .last()
                .ok_or_else(|| Error::invalid("golden hidden"))?,
        )?);
    }
    let tokens = &golden
        .prefixes
        .last()
        .ok_or_else(|| Error::invalid("empty golden"))?
        .tokens;
    if !decode_only {
        verify.reset()?;
        for (chunk, inputs) in tokens.chunks(3).enumerate() {
            let results = verify.step_batch(inputs, chunk * 3)?;
            for (lane, result) in results.iter().take(inputs.len()).enumerate() {
                max_error = max_error.max(compare(
                    &result.1,
                    &golden.prefixes[chunk * 3 + lane].logits,
                )?);
            }
        }
    }
    // The bank and every weight must outlive their loader through captured ownership.
    drop(loaded);
    scalar.reset()?;
    let again = scalar.step(tokens[0], 0, 0, None, true)?;
    max_error = max_error.max(compare(&again.1, &golden.prefixes[0].logits)?);
    let mut paths = vec!["decode", "reset", "weight_lifetime"];
    if !decode_only && !verify_only {
        paths.push("prefill32");
    }
    if !decode_only {
        paths.push("verification3");
    }
    println!(
        "{}",
        serde_json::json!({"passed": true, "package": root, "prefixes": golden.prefixes.len(),
        "max_absolute_error": max_error, "paths": paths})
    );
    Ok(())
}

#[cfg(cuda_backend)]
#[derive(serde::Deserialize)]
struct Golden {
    prefixes: Vec<Prefix>,
}
#[cfg(cuda_backend)]
#[derive(serde::Deserialize)]
struct Prefix {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    hidden: Vec<Vec<f32>>,
}

#[cfg(cuda_backend)]
fn compare(actual: &[f32], expected: &[f32]) -> infer_core::Result<f32> {
    if actual.len() != expected.len() {
        return Err(infer_core::Error::invalid("parity output shape"));
    }
    let mut maximum = 0.0f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        if !actual.is_finite() || error > expected.abs().mul_add(0.000_5, 0.000_05) {
            return Err(infer_core::Error::invariant(format!(
                "MoE parity index {index}: {actual} vs {expected}, error {error}"
            )));
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

#[cfg(cuda_backend)]
fn reference_graph(root: &str, golden: &Golden) -> infer_core::Result<()> {
    use infer_ir::{StateKind, TensorOp, TensorStorage};
    use infer_state::physical::{PagedRows, PhysicalTensor};
    use std::collections::BTreeMap;
    let mut package = infer_models::QuantizedPackage::open(root, infer_core::ModelId::ONE)?;
    let mut weights = BTreeMap::new();
    for (slot, source) in package.weights.clone() {
        let bytes = package.read(&source.data, 1024 * 1024)?;
        let converted = infer_models::convert_float_bytes(&bytes, source.data.dtype)?;
        weights.insert(
            slot,
            converted
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>(),
        );
    }
    for prefix in &golden.prefixes {
        let mut values = BTreeMap::new();
        let mut states = BTreeMap::new();
        for spec in &package.graph.tensors {
            match &spec.storage {
                TensorStorage::Weight { slot } => {
                    values.insert(spec.id, weights[slot].clone());
                }
                TensorStorage::State { kind, .. } => {
                    let state = match kind {
                        StateKind::AttentionKv => PhysicalTensor::Kv {
                            keys: PagedRows::new(spec.shape[1], 4, 64)?,
                            values: PagedRows::new(spec.shape[1], 4, 64)?,
                        },
                        StateKind::Conv => PhysicalTensor::Conv {
                            channels: spec.shape[0],
                            kernel: spec.shape[1],
                            history: vec![0.0; spec.shape[0] * (spec.shape[1] - 1)],
                        },
                        StateKind::LinearAttention => PhysicalTensor::Delta {
                            heads: spec.shape[0],
                            key_dim: spec.shape[1],
                            value_dim: spec.shape[2],
                            recurrent: vec![0.0; spec.elements()?],
                        },
                        _ => return Err(infer_core::Error::unsupported("reference state")),
                    };
                    states.insert(spec.id, state);
                }
                TensorStorage::Activation => {}
            }
        }
        for (position, token) in prefix.tokens.iter().enumerate() {
            for node in &package.graph.nodes {
                let inputs: Vec<_> = node.inputs.iter().map(|id| values[id].as_slice()).collect();
                let inputs = if node.op == TensorOp::Embedding {
                    let start = usize::try_from(*token)
                        .map_err(|e| infer_core::Error::invalid(e.to_string()))?
                        * package.imported.model.hidden_size;
                    vec![&inputs[0][start..start + package.imported.model.hidden_size]]
                } else {
                    inputs
                };
                let output = infer_reference_ops::reference_operation(
                    &node.op,
                    &inputs,
                    node.states.first().and_then(|id| states.get_mut(id)),
                    *token,
                    position,
                )?;
                for (id, output) in node.outputs.iter().zip(output) {
                    values.insert(*id, output);
                }
            }
        }
        compare(
            &values[&package
                .graph
                .logits
                .ok_or_else(|| infer_core::Error::invalid("reference logits"))?],
            &prefix.logits,
        )?;
    }
    eprintln!("independent scalar graph matches Transformers");
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA MoE check requires Linux");
    exit(1);
}
