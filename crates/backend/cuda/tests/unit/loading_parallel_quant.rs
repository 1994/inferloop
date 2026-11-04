use super::{LoadOptions, LoadedModel};
use crate::resident::fp8_cache::KvState;
use crate::{
    device::CudaDevice,
    resident::{
        DeviceProgram,
        fp8_quant_reduction::{captures, with_parallel_conversion},
    },
};
use infer_core::{Error, ModelId, Result};
use std::env::var;
use std::fs::read;
fn equal(a: &[f32], b: &[f32], field: &str) {
    assert_eq!(a.len(), b.len(), "{field}");
    assert!(a.iter().chain(b).all(|x| x.is_finite()), "{field}");
    assert!(
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "{field} bits"
    );
}
fn states(loaded: &LoadedModel, a: &DeviceProgram, b: &DeviceProgram, label: &str) -> Result<()> {
    assert_eq!(a.states().len(), b.states().len());
    let mut float_elements = 0;
    let mut packed_bytes = 0;
    for (id, parts) in a.states() {
        assert_eq!(parts.len(), b.states()[id].len());
        for (x, y) in parts.iter().zip(&b.states()[id]) {
            let x = loaded.device().runtime().read(x.memory())?;
            let y = loaded.device().runtime().read(y.memory())?;
            equal(&x, &y, "full F32 state");
            float_elements += x.len();
        }
    }
    assert_eq!(a.fp8_states().len(), b.fp8_states().len());
    for (
        id,
        KvState {
            keys: ak,
            values: av,
            ..
        },
    ) in a.fp8_states()
    {
        let KvState {
            keys: bk,
            values: bv,
            ..
        } = &b.fp8_states()[id];
        for (x, y) in [(ak, bk), (av, bv)] {
            let x = loaded.device().runtime().read(x)?;
            let y = loaded.device().runtime().read(y)?;
            assert_eq!(x.len(), y.len());
            assert!(x.iter().chain(&y).all(|x| x.0 & 127 != 127));
            assert!(
                x.iter().zip(&y).all(|(x, y)| x.0 == y.0),
                "full FP8 state bits"
            );
            packed_bytes += x.len();
        }
    }
    println!(
        "{}",
        serde_json::json!({"parallel_quant_model_state":label,"F32_elements":float_elements,"FP8_bytes":packed_bytes,"different":0})
    );
    Ok(())
}
#[test]
#[ignore = "requires CUDA/checkpoint/frozen511tokens/safe-run; actual parallel quantization and complete state equivalence"]
fn full_model_parallel_quantization_census() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 511);
    let width = var("INFER_TEST_PREFILL_WIDTH")
        .unwrap_or_else(|_| "64".into())
        .parse::<usize>()
        .map_err(|e| Error::invalid(e.to_string()))?;
    assert!([64, 256].contains(&width));
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            mtp_depth: 0,
            prefill_width: width,
            ..LoadOptions::default()
        },
    )?;
    let before = captures();
    let mut old = with_parallel_conversion(false, || loaded.sequence(tokens.len() + 64))?;
    assert_eq!(captures(), before);
    let mut new = with_parallel_conversion(true, || loaded.sequence(tokens.len() + 64))?;
    let after = captures();
    assert!(after - before >= 100);
    println!(
        "{}",
        serde_json::json!({"actual_parallel_quant_captures":after-before})
    );
    let mut first = None;
    for (chunk, input) in tokens.chunks(width).enumerate() {
        let pos = chunk * width;
        let last = pos + input.len() == tokens.len();
        let a = old.prefill_batch(input, pos, last)?;
        let b = new.prefill_batch(input, pos, last)?;
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            equal(&x.0, &y.0, "prefill hidden");
            equal(&x.1, &y.1, "prefill logits");
        }
        states(&loaded, &old, &new, &format!("prefill{chunk}"))?;
        if last {
            first = Some(
                a.last()
                    .ok_or_else(|| Error::invariant("last prefill row"))?
                    .1
                    .clone(),
            );
        }
    }
    let mut next = greedy(&first.ok_or_else(|| Error::invariant("prefill logits"))?)?;
    for step in 0..16 {
        let pos = tokens.len() + step;
        let a = old.step(next, pos, pos, None, true)?;
        let b = new.step(next, pos, pos, None, true)?;
        equal(&a.0, &b.0, "decode hidden");
        equal(&a.1, &b.1, "decode logits");
        states(&loaded, &old, &new, &format!("decode{step}"))?;
        next = greedy(&a.1)?;
    }
    Ok(())
}
fn greedy(logits: &[f32]) -> Result<u32> {
    let (index, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .ok_or_else(|| Error::invariant("empty logits"))?;
    u32::try_from(index).map_err(|e| Error::invalid(e.to_string()))
}
