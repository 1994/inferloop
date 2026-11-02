//! Real checkpoint, one weight upload, distinct prompt captures and a prediction gate.
use super::{LoadOptions, LoadedModel};
use crate::tuning::table_path;
use crate::{
    device::{CudaDevice, device_error},
    resident::{BatchOutput, DeviceProgram},
};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, ModelId, Result};
use serde_json::{Value, json};
use std::env::var;
use std::env::var_os;
use std::fs::read;

const NARROW: usize = 64;
const WIDE: usize = 256;
const PROMPT_TOKENS: usize = 511;
const CAPACITY: usize = 1024;
const VERIFY: usize = 3;
const DECODE: usize = 8;
/// Largest KL divergence accepted between two prompt widths' next-token distributions. Widths
/// quantize different activation batches; the numeric contract judges predictions, not bits.
const MAXIMUM_LOGITS_KL: f64 = 0.25;

#[derive(Default)]
struct Agreement {
    fields: usize,
    elements: usize,
    differing: usize,
    non_finite: usize,
    maximum_absolute: f64,
    first_difference: Option<Value>,
    logits: usize,
    top_disagreements: usize,
    maximum_kl: f64,
}

/// `KL(wide || narrow)` of the distributions two logits rows define, and whether their top token
/// agrees.
fn prediction(wide: &[f32], narrow: &[f32]) -> (f64, bool) {
    let distribution = |logits: &[f32]| {
        let top = logits.iter().copied().fold(f32::MIN, f32::max);
        let weights: Vec<f64> = logits.iter().map(|v| f64::from(v - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.into_iter().map(|v| v / total).collect::<Vec<_>>()
    };
    let (p, q) = (distribution(wide), distribution(narrow));
    let divergence = p
        .iter()
        .zip(&q)
        .filter(|(p, q)| **p > 0.0 && **q > 0.0)
        .map(|(p, q)| p * (p / q).ln())
        .sum();
    let top = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
    };
    (divergence, top(wide) == top(narrow))
}

impl Agreement {
    fn observe(&mut self, stage: &str, field: &str, index: usize, left: f32, right: f32) {
        self.elements += 1;
        let finite = left.is_finite() && right.is_finite();
        self.non_finite += usize::from(!finite);
        if finite {
            self.maximum_absolute = self
                .maximum_absolute
                .max((f64::from(left) - f64::from(right)).abs());
        }
        if left.to_bits() != right.to_bits() || !finite {
            self.differing += 1;
            self.first_difference.get_or_insert_with(|| {
                json!({
                    "stage": stage, "field": field, "element": index,
                    "wide_bits": left.to_bits(), "narrow_bits": right.to_bits(),
                    "wide": finite.then_some(left), "narrow": finite.then_some(right),
                    "finite": finite
                })
            });
        }
    }

    fn compare(&mut self, stage: &str, field: &str, wide: &[f32], narrow: &[f32]) {
        assert_eq!(wide.len(), narrow.len(), "{stage}: {field} length");
        assert!(!wide.is_empty(), "{stage}: {field} must be populated");
        self.fields += 1;
        for (index, (&left, &right)) in wide.iter().zip(narrow).enumerate() {
            self.observe(stage, field, index, left, right);
        }
        // Past the prompt this fixture feeds tokens no model would predict, so only the rows
        // that end a real prompt carry a prediction worth holding the widths to.
        if field.starts_with("logits") && stage.starts_with("prompt_end") {
            let (divergence, same_top) = prediction(wide, narrow);
            self.logits += 1;
            self.top_disagreements += usize::from(!same_top);
            self.maximum_kl = self.maximum_kl.max(divergence);
        }
    }

    fn compare_fp8(&mut self, stage: &str, field: &str, wide: &[f8e4m3fn], narrow: &[f8e4m3fn]) {
        assert_eq!(wide.len(), narrow.len(), "{stage}: {field} length");
        assert!(!wide.is_empty(), "{stage}: {field} must be populated");
        self.fields += 1;
        for (index, (&left, &right)) in wide.iter().zip(narrow).enumerate() {
            // Keep raw-byte diagnostics; the finite conversion is injective, including signed zero.
            let left_value = fp8_value(left.0);
            let right_value = fp8_value(right.0);
            if left.0 != right.0 || !left_value.is_finite() || !right_value.is_finite() {
                self.first_difference.get_or_insert_with(|| {
                    json!({"stage":stage, "field":field, "element":index,
                        "wide_fp8_bits":left.0, "narrow_fp8_bits":right.0})
                });
            }
            self.observe(stage, field, index, left_value, right_value);
        }
    }

    fn report(&self, stage: &str) {
        eprintln!(
            "{}",
            json!({"stage":stage, "cumulative_fields":self.fields,
                "cumulative_elements":self.elements, "different_elements":self.differing,
                "non_finite_pairs":self.non_finite, "maximum_absolute":self.maximum_absolute,
                "logits_rows":self.logits, "top_disagreements":self.top_disagreements,
                "maximum_kl":self.maximum_kl,
                "first_difference":self.first_difference})
        );
    }
}

fn fp8_value(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;
    if exponent == 15 && fraction == 7 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        f32::from(fraction) / 512.0
    } else {
        (1.0 + f32::from(fraction) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Copy through one temporary tensor: the captured source remains owned by its program.
fn read_fp8(device: &CudaDevice, source: &Tensor<f8e4m3fn>) -> Result<Vec<f8e4m3fn>> {
    let mut copy = api::zeros::<f8e4m3fn>(&[source.size()])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    api::memcpy(&mut copy, source)
        .sync_on(&device.stream)
        .map_err(device_error)?;
    copy.to_host_vec()
        .sync_on(&device.stream)
        .map_err(device_error)
}

fn compare_states(
    loaded: &LoadedModel,
    wide: &DeviceProgram,
    narrow: &DeviceProgram,
    agreement: &mut Agreement,
    stage: &str,
) -> Result<()> {
    assert_eq!(
        wide.states().len(),
        narrow.states().len(),
        "{stage} state IDs"
    );
    assert!(
        !wide.states().is_empty(),
        "real checkpoint must have recurrent states"
    );
    for (id, actual) in wide.states() {
        let expected = narrow
            .states()
            .get(id)
            .ok_or_else(|| Error::invariant("state ID"))?;
        assert_eq!(actual.len(), expected.len(), "{stage}: {id:?} tensor count");
        let spec = loaded
            .graph()
            .tensors
            .iter()
            .find(|spec| spec.id == *id)
            .ok_or_else(|| Error::invariant("state spec"))?;
        for (part, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.shape(),
                expected.shape(),
                "{stage}: {id:?} state shape"
            );
            let field = format!("{id:?} {:?} part={part}", spec.storage);
            agreement.compare(
                stage,
                &field,
                &loaded.device().read_borrowed(actual)?,
                &loaded.device().read_borrowed(expected)?,
            );
        }
    }
    assert_eq!(
        wide.fp8_states().len(),
        narrow.fp8_states().len(),
        "{stage} FP8 KV IDs"
    );
    assert!(
        !wide.fp8_states().is_empty(),
        "fixture must exercise real FP8 KV"
    );
    for (id, (actual_keys, actual_values)) in wide.fp8_states() {
        let (expected_keys, expected_values) = narrow
            .fp8_states()
            .get(id)
            .ok_or_else(|| Error::invariant("FP8 KV ID"))?;
        for (kind, actual, expected) in [
            ("keys", actual_keys, expected_keys),
            ("values", actual_values, expected_values),
        ] {
            assert_eq!(
                actual.shape(),
                expected.shape(),
                "{stage}: {id:?} {kind} shape"
            );
            agreement.compare_fp8(
                stage,
                &format!("{id:?} FP8 {kind}"),
                &read_fp8(loaded.device(), actual)?,
                &read_fp8(loaded.device(), expected)?,
            );
        }
    }
    agreement.report(stage);
    Ok(())
}

fn compare_outputs(
    agreement: &mut Agreement,
    stage: &str,
    position: usize,
    wide: &BatchOutput,
    narrow: &BatchOutput,
    all_logits: bool,
) {
    assert_eq!(wide.len(), narrow.len(), "{stage} rows");
    assert!(!wide.is_empty(), "{stage} output rows");
    for (lane, (actual, expected)) in wide.iter().zip(narrow).enumerate() {
        agreement.compare(
            stage,
            &format!("hidden token={}", position + lane),
            &actual.0,
            &expected.0,
        );
        // Prompt readback supplies only the last row's logits; verification supplies every row.
        if all_logits || lane + 1 == wide.len() {
            agreement.compare(
                stage,
                &format!("logits token={}", position + lane),
                &actual.1,
                &expected.1,
            );
        }
    }
}

fn compare_prompt(
    loaded: &LoadedModel,
    tokens: &[u32],
    wide: &mut DeviceProgram,
    narrow: &mut DeviceProgram,
    agreement: &mut Agreement,
) -> Result<()> {
    for (chunk, tokens) in tokens.chunks(WIDE).enumerate() {
        let position = chunk * WIDE;
        let actual = wide.prefill_batch(tokens, position, true)?;
        let mut expected = Vec::with_capacity(tokens.len());
        for (part, tokens) in tokens.chunks(NARROW).enumerate() {
            expected.extend(narrow.prefill_batch(tokens, position + part * NARROW, true)?);
        }
        let stage = format!("prompt_end={}", position + tokens.len());
        compare_outputs(agreement, &stage, position, &actual, &expected, false);
        assert_eq!(wide.position(), narrow.position(), "{stage} position");
        compare_states(loaded, wide, narrow, agreement, &stage)?;
    }
    Ok(())
}

fn compare_continuation(
    loaded: &LoadedModel,
    tokens: &[u32],
    wide: &mut DeviceProgram,
    narrow: &mut DeviceProgram,
    agreement: &mut Agreement,
) -> Result<()> {
    let position = tokens.len();
    let verify: Vec<_> = tokens.iter().copied().cycle().take(VERIFY).collect();
    compare_outputs(
        agreement,
        "verification",
        position,
        &wide.step_batch(&verify, position)?,
        &narrow.step_batch(&verify, position)?,
        true,
    );
    compare_states(loaded, wide, narrow, agreement, "verification")?;
    wide.commit_batch(1)?;
    narrow.commit_batch(1)?;
    assert_eq!(wide.position(), position + 1);
    assert_eq!(narrow.position(), position + 1);
    compare_states(loaded, wide, narrow, agreement, "rollback_commit=1")?;
    for step in 0..DECODE {
        let position = position + 1 + step;
        let token = tokens[step % tokens.len()];
        let actual = wide.step(token, position, position, None, true)?;
        let expected = narrow.step(token, position, position, None, true)?;
        let stage = format!("decode_step={step}");
        agreement.compare(&stage, "hidden", &actual.0, &expected.0);
        agreement.compare(&stage, "logits", &actual.1, &expected.1);
        compare_states(loaded, wide, narrow, agreement, &stage)?;
    }
    Ok(())
}

fn exact_profile() -> Result<()> {
    for variable in [
        "INFER_CUDA_CHUNKED_RECURRENT",
        "INFER_CUDA_VENDOR_FP8",
        "INFER_CUDA_FP8_LT_AUTOTUNE",
        "INFER_CUBLAS_PROJECTIONS",
    ] {
        if var_os(variable).is_some() {
            return Err(Error::invalid(format!(
                "exact width test rejects {variable}"
            )));
        }
    }
    Ok(())
}

fn prompt() -> Result<(String, Vec<u32>)> {
    let tokens_path =
        var("INFER_TEST_TOKENS").map_err(|error| Error::invalid(error.to_string()))?;
    let tokens: Vec<u32> = serde_json::from_slice(
        &read(&tokens_path).map_err(|error| Error::invalid(error.to_string()))?,
    )
    .map_err(|error| Error::invalid(error.to_string()))?;
    if tokens.len() != PROMPT_TOKENS {
        return Err(Error::invalid(
            "exact width test requires the frozen 511-token prompt",
        ));
    }
    Ok((tokens_path, tokens))
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 511-token INFER_TEST_TOKENS"]
fn checkpoint_prefill_64_and_256_agree_on_predictions() -> Result<()> {
    exact_profile()?;
    let path = var("INFER_TEST_MODEL").map_err(|error| Error::invalid(error.to_string()))?;
    let (tokens_path, tokens) = prompt()?;
    let cache_path = table_path()
        .ok_or_else(|| Error::invalid("exact width test requires a frozen tuning table"))?;
    let cache = read(&cache_path).map_err(|error| Error::invalid(error.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: WIDE,
            verification_width: VERIFY,
            mtp_depth: 0,
            autotune: false,
            chunked_recurrent: false,
            ..LoadOptions::default()
        },
    )?;
    // Capture retains Arc-backed weight owners. Changing capture policy reuses all uploaded weights.
    loaded.weights.prefill_width = NARROW;
    loaded.weights.narrow_prefill_width = 0;
    let mut narrow = loaded.sequence(CAPACITY)?;
    loaded.weights.prefill_width = WIDE;
    let mut wide = loaded.sequence(CAPACITY)?;
    assert_eq!(narrow.prefill_width(), NARROW);
    assert_eq!(wide.prefill_width(), WIDE);
    assert_eq!(narrow.batch_width(), VERIFY);
    assert_eq!(wide.batch_width(), VERIFY);
    eprintln!(
        "{}",
        json!({"actual_widths":[narrow.prefill_width(),wide.prefill_width()],
        "capacity":CAPACITY, "verification_width":VERIFY, "mtp_depth":0,
        "chunked_recurrent":false, "autotune":false, "tokens":tokens_path,
        "cache_path":cache_path, "cache_bytes":cache.len(),
        "weight_uploads":1, "hardware":loaded.profile(),
        "prepared_configs_by_arm":{"narrow":narrow.prepared_gemm_configs(),
            "wide":wide.prepared_gemm_configs()},
        "prepared_configs_are_replay_trace":false})
    );
    let mut agreement = Agreement::default();
    compare_prompt(&loaded, &tokens, &mut wide, &mut narrow, &mut agreement)?;
    compare_continuation(&loaded, &tokens, &mut wide, &mut narrow, &mut agreement)?;
    let cache_after = read(&cache_path).map_err(|error| Error::invalid(error.to_string()))?;
    agreement.report("final");
    assert!(
        cache == cache_after,
        "no-autotune must preserve the frozen table"
    );
    assert_eq!(agreement.non_finite, 0, "all checked values must be finite");
    assert!(agreement.logits > 0, "logits must be compared");
    assert_eq!(agreement.top_disagreements, 0, "64/256 top token");
    assert!(
        agreement.maximum_kl <= MAXIMUM_LOGITS_KL,
        "64/256 KL {}",
        agreement.maximum_kl
    );
    Ok(())
}
