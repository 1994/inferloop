//! Same-checkpoint output and complete recurrent/KV-state gate for Rust SIMT attention.
use super::{LoadOptions, LoadedModel, quant_reuse_model_tests as reference};
use crate::resident::fp8_cache::{KvLayout, KvState, kv_geometry};
use crate::{
    device::CudaDevice,
    resident::{
        attention_oxide::{captures, without_oxide},
        fp8_cache::Fp8Caches,
    },
};
use infer_core::{Error, ModelId, Result};
use std::{env::var, fs::read};

#[test]
#[ignore = "requires CUDA/checkpoint/frozen511tokens/safe-run; complete model output and state equivalence"]
fn rust_attention_preserves_checkpoint_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = reference::prompt()?;
    let width = var("INFER_TEST_PREFILL_WIDTH")
        .unwrap_or_else(|_| "256".into())
        .parse::<usize>()
        .map_err(|e| Error::invalid(e.to_string()))?;
    assert!([64, 256, 512].contains(&width));
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: width,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    let before = captures();
    let mut control = without_oxide(|| loaded.sequence(1024))?;
    assert_eq!(captures(), before, "reference must use cuTile");
    let mut candidate = loaded.sequence(1024)?;
    let actual_captures = captures() - before;
    assert!(
        actual_captures >= 16,
        "Rust attention must actually be captured"
    );
    let mut next = tokens[0];
    for length in [511, 51] {
        for (chunk, input) in tokens[..length].chunks(width).enumerate() {
            let position = chunk * width;
            let a = candidate.prefill_batch(input, position, true)?;
            let b = control.prefill_batch(input, position, true)?;
            reference::outputs(&a, &b);
            reference::program_states(&loaded, &candidate, &control)?;
            next = reference::greedy(
                &a.last()
                    .ok_or_else(|| Error::invariant("prefill output"))?
                    .1,
            )?;
            println!(
                "{}",
                serde_json::json!({
                    "oxide_model_prefill":true, "width":width, "length":length,
                    "base":position, "count":input.len(), "full_hidden_logits_bits":true,
                    "full_F32_state_bits":true, "full_FP8_KV_bits":true,
                    "actual_captures":actual_captures,
                })
            );
        }
        for _ in 0..16 {
            let position = candidate.position();
            let a = candidate.step(next, position, position, None, true)?;
            let b = control.step(next, position, position, None, true)?;
            reference::outputs(&vec![a.clone()], &vec![b]);
            reference::program_states(&loaded, &candidate, &control)?;
            next = reference::greedy(&a.1)?;
        }
    }
    for (count, committed) in [(3, 1), (3, 2), (3, 3)] {
        let position = candidate.position();
        reference::outputs(
            &candidate.step_batch(&tokens[..count], position)?,
            &control.step_batch(&tokens[..count], position)?,
        );
        reference::program_states(&loaded, &candidate, &control)?;
        candidate.commit_batch(committed)?;
        control.commit_batch(committed)?;
        reference::program_states(&loaded, &candidate, &control)?;
        let position = candidate.position();
        reference::outputs(
            &vec![candidate.step(next, position, position, None, true)?],
            &vec![control.step(next, position, position, None, true)?],
        );
        reference::program_states(&loaded, &candidate, &control)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "oxide_complete_model_gate":true,"width":width,"actual_captures":actual_captures,
            "greedy_decode_steps":32,"verification_commit_lengths":[1,2,3],
            "full_hidden_logits_bits":true,"full_recurrent_and_KV_state_bits":true,
            "chunked_recurrent":false,
        })
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA/checkpoint/frozen1025tokens/safe-run; default-width full chunks and state bits"]
fn wide_attention_preserves_checkpoint_model_bits() -> Result<()> {
    checkpoint_bits(false, 2048)
}

#[test]
#[ignore = "requires CUDA/checkpoint/frozen1025tokens/safe-run; pooled physical KV stride and complete bits"]
fn pooled_prefill_preserves_checkpoint_model_bits() -> Result<()> {
    for width in [1024, 2048] {
        checkpoint_bits(true, width)?;
    }
    Ok(())
}

fn pooled_caches(loaded: &LoadedModel) -> Result<Fp8Caches> {
    let mut result = Fp8Caches::new();
    for spec in &loaded.graph.tensors {
        if loaded.weights.kv_scales.contains_key(&spec.id) {
            let (heads, dim) = kv_geometry(&loaded.graph, spec.id)?;
            let layout = KvLayout {
                heads,
                rows: 12288,
                dim,
            };
            result.insert(spec.id, KvState::zeros(&loaded.device, layout)?);
        }
    }
    Ok(result)
}

fn checkpoint_bits(pooled: bool, width: usize) -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let prompt = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(prompt).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 1025);
    let tokens = tokens
        .iter()
        .copied()
        .cycle()
        .take(2049)
        .collect::<Vec<_>>();
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            // Match serving: the loaded model stays wide while its worker may halve.
            prefill_width: 2048,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = 0;
    let build = || {
        if pooled {
            loaded
                .prompt_worker(4096, Some(pooled_caches(&loaded)?), width)
                .map(|(program, _)| program)
        } else {
            loaded.sequence(4096)
        }
    };
    let before = captures();
    let mut control = without_oxide(build)?;
    assert_eq!(captures(), before);
    let mut candidate = build()?;
    assert!(
        captures() >= before + 16,
        "wide exact attention must be captured"
    );
    for length in [2049, 1025, 511, 1] {
        control.reset()?;
        candidate.reset()?;
        let mut next = tokens[0];
        for (chunk, input) in tokens[..length].chunks(width).enumerate() {
            let a = candidate.prefill_batch(input, chunk * width, true)?;
            let b = control.prefill_batch(input, chunk * width, true)?;
            reference::outputs(&a, &b);
            reference::program_states(&loaded, &candidate, &control)?;
            next = reference::greedy(
                &a.last()
                    .ok_or_else(|| Error::invariant("prefill output"))?
                    .1,
            )?;
        }
        for _ in 0..4 {
            let position = candidate.position();
            let a = candidate.step(next, position, position, None, true)?;
            let b = control.step(next, position, position, None, true)?;
            reference::outputs(&vec![a.clone()], &vec![b]);
            reference::program_states(&loaded, &candidate, &control)?;
            next = reference::greedy(&a.1)?;
        }
        println!(
            "wide checkpoint bits passed: pooled={pooled}, width={width}, length={length}, captures={}",
            captures() - before
        );
    }
    Ok(())
}
