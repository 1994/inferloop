//! Opt-in vendor FP8 pilot: unchanged checkpoint codes, unchanged token quantizer, F32 output.
//! These ignored tests execute CUDA only when the serial safe-run owner invokes them.
use super::super::fp8_gemm::kernels;
use crate::device::{CudaDevice, vendor_fp8::Support};
use crate::tuning::fp8_gemm::Choice as Fp8GemmChoice;
use crate::tuning::resident_gemm::native_tile;
use cuda_core::f8e4m3fn;
use cutile::bench::BenchOptions;
use cutile::bench::do_bench_paired;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use infer_models::{DeviceWeight, QuantizedPackage, WeightEncoding, convert_float_bytes};
use std::env::var;
use std::error::Error;
use std::{sync::Arc, time::Duration};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;
const STAGING: u64 = 2 * 1024 * 1024 * 1024;
const ORACLE_SAMPLES: usize = 32;
const ROWS: [usize; 4] = [1, 12, 64, 128];

struct Weight {
    name: String,
    n: usize,
    k: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
    device_codes: Arc<Tensor<f8e4m3fn>>,
    device_scales: Arc<Tensor<f32>>,
}
impl Weight {
    fn new(
        device: &CudaDevice,
        name: String,
        n: usize,
        k: usize,
        codes: Vec<u8>,
        scales: Vec<f32>,
    ) -> TestResult<Self> {
        if codes.len() != n * k
            || scales.len() != n
            || codes.iter().any(|code| code & 127 == 127)
            || scales
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err("invalid finite FP8 channel checkpoint values".into());
        }
        Ok(Self {
            name,
            n,
            k,
            device_codes: device.upload(codes.iter().copied().map(f8e4m3fn).collect(), &[n, k])?,
            device_scales: device.upload(scales.clone(), &[n])?,
            codes,
            scales,
        })
    }
    fn synthetic(device: &CudaDevice, n: usize, k: usize) -> TestResult<Self> {
        let codes = (0..n * k)
            .map(|i| {
                // Broad finite magnitudes and independent signs; no all-zero or constant weight shortcut.
                24 + u8::try_from((i * 13 + i / 7) % 56).unwrap()
                    + if (i * 17 + i / 11) % 5 < 2 { 128 } else { 0 }
            })
            .collect();
        let scales = (0..n)
            .map(|i| f32::from(u16::try_from(i % 13 + 1).unwrap()) / 113.0)
            .collect();
        Self::new(device, format!("synthetic_{n}x{k}"), n, k, codes, scales)
    }
    fn checkpoint(
        device: &CudaDevice,
        package: &QuantizedPackage,
        name: &str,
        source: &DeviceWeight,
    ) -> TestResult<Self> {
        if source.encoding != WeightEncoding::Fp8Channel || !source.fp8_token_input {
            return Err(format!("{name}: not checkpoint-declared FP8 token/channel").into());
        }
        let [n, k] = source.shape.as_slice() else {
            return Err("FP8 checkpoint matrix rank".into());
        };
        let codes = package.read_view(&source.data, STAGING)?.to_vec();
        let scale = source
            .scale
            .as_ref()
            .ok_or("FP8 checkpoint scale missing")?;
        let converted = convert_float_bytes(package.read_view(scale, STAGING)?, scale.dtype)?;
        let scales = converted
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "kind": "checkpoint_source", "weight": name, "shape": [n, k],
                "encoded_source": source.data, "scale_source": scale,
                "arithmetic": "original E4M3 bytes + unchanged quantizer + raw F32 GEMM * xs * ws",
                "requantized_weight": false, "vendor_workspace_bytes": 0, "fast_accum": false,
            })
        );
        Self::new(device, name.to_owned(), *n, *k, codes, scales)
    }
}

struct Inputs<'a> {
    weight: &'a Weight,
    rows: usize,
    values: Vec<f32>,
    input: Arc<Tensor<f32>>,
    codes: Tensor<f8e4m3fn>,
    scales: Tensor<f32>,
    vendor: Support,
}
impl<'a> Inputs<'a> {
    fn new(device: &CudaDevice, weight: &'a Weight, rows: usize) -> TestResult<Self> {
        let values: Vec<f32> = (0..rows * weight.k)
            .map(|i| {
                if rows > 1 && i < weight.k {
                    0.0
                } else {
                    (f32::from(u16::try_from((i * 17 + i / weight.k * 23) % 131).unwrap()) - 65.0)
                        / 37.0
                }
            })
            .collect();
        Ok(Self {
            weight,
            rows,
            input: device.upload(values.clone(), &[rows, weight.k])?,
            values,
            codes: api::zeros::<f8e4m3fn>(&[rows, weight.k]).sync_on(&device.stream)?,
            scales: api::zeros::<f32>(&[rows, 1]).sync_on(&device.stream)?,
            vendor: Support::new_with(device, true),
        })
    }
    fn quantize(&mut self, scope: &Scope) -> Result<(), DeviceError> {
        let k = self.weight.k;
        scope.record(
            kernels::quantize(
                (&mut self.codes).partition([1, k.next_power_of_two()]),
                (&mut self.scales).partition([1, 1]),
                &self.input,
            )
            .generics(vec![k.to_string(), k.next_power_of_two().to_string()]),
        )?;
        Ok(())
    }
    fn matmul(
        &self,
        scope: &Scope,
        output: &mut Tensor<f32>,
        vendor: bool,
    ) -> Result<(), DeviceError> {
        if vendor {
            if !self.vendor.record(
                scope,
                &self.codes,
                &self.weight.device_codes,
                output,
                &self.scales,
                &self.weight.device_scales,
            )? {
                return Err(DeviceError::Internal(
                    "requested vendor shape was not warmed".into(),
                ));
            }
        } else {
            let tile = native_tile(self.rows);
            scope.record(
                kernels::matmul(
                    output.partition(tile),
                    &self.codes,
                    &self.weight.device_codes,
                    &self.scales,
                    &self.weight.device_scales,
                )
                .generics(vec![
                    self.weight.k.to_string(),
                    tile[0].to_string(),
                    tile[1].to_string(),
                ]),
            )?;
        }
        Ok(())
    }
}
fn decode(code: u8) -> f32 {
    let magnitude = code & 127;
    let exponent = magnitude >> 3;
    let mantissa = f32::from(magnitude & 7);
    let value = if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if code & 128 == 0 { value } else { -value }
}
fn encode(value: f32) -> u8 {
    let magnitude = (0..=126u8)
        .min_by(|&a, &b| {
            (decode(a) - value.abs().min(448.0))
                .abs()
                .total_cmp(&(decode(b) - value.abs().min(448.0)).abs())
                .then((a % 2).cmp(&(b % 2)))
        })
        .unwrap();
    magnitude | if value.is_sign_negative() { 128 } else { 0 }
}
fn validate_quantizer(
    inputs: &Inputs<'_>,
    encoded: &[f8e4m3fn],
    scales: &[f32],
    independent: bool,
) {
    let k = inputs.weight.k;
    for row in 0..inputs.rows {
        let span = &inputs.values[row * k..(row + 1) * k];
        let scale = (span.iter().copied().map(f32::abs).fold(0.0, f32::max) / 448.0).max(1e-12);
        assert!(
            (scales[row] - scale).abs() <= scale * 1e-6,
            "row={row}: token scale {} != {scale}",
            scales[row]
        );
        if independent {
            for column in 0..k {
                assert_eq!(
                    encoded[row * k + column].0,
                    encode(span[column] / scale),
                    "row={row} K={column}"
                );
            }
        }
    }
}
fn compare_outputs(reference: &[f32], actual: &[f32]) -> TestResult<serde_json::Value> {
    if reference.len() != actual.len() {
        return Err("output length changed".into());
    }
    let mut identical = 0_usize;
    let mut max_absolute = 0.0_f32;
    let mut max_normalized = 0.0_f32;
    let mut squared = 0.0_f64;
    let mut failed = 0_usize;
    let mut nonfinite = 0_usize;
    for (&left, &right) in reference.iter().zip(actual) {
        if !left.is_finite() || !right.is_finite() {
            nonfinite += 1;
            continue;
        }
        let absolute = (left - right).abs();
        let normalized = absolute / left.abs().max(1.0);
        max_absolute = max_absolute.max(absolute);
        max_normalized = max_normalized.max(normalized);
        squared += f64::from(absolute).powi(2);
        identical += usize::from(left.to_bits() == right.to_bits());
        failed += usize::from(normalized > 5e-4);
    }
    Ok(serde_json::json!({
        "outputs": actual.len(), "identical_outputs": identical,
        "all_outputs_identical": identical == actual.len(),
        "max_absolute_error": max_absolute, "max_normalized_error": max_normalized,
        "squared_error_sum": squared, "failed_outputs": failed, "nonfinite_outputs": nonfinite,
        "within_numerical_contract": failed == 0 && nonfinite == 0,
        "eligible_for_exact_arithmetic_profile": identical == actual.len(),
    }))
}
fn oracle(
    inputs: &Inputs<'_>,
    encoded: &[f8e4m3fn],
    scales: &[f32],
    native: &[f32],
    vendor: &[f32],
    full: bool,
) -> serde_json::Value {
    let weight = inputs.weight;
    let count = native.len();
    let samples = if full {
        count
    } else {
        ORACLE_SAMPLES.min(count)
    };
    let mut native_error = 0.0_f64;
    let mut vendor_error = 0.0_f64;
    let mut failed = 0_usize;
    let mut failures = Vec::new();
    for sample in 0..samples {
        let index = if full || samples == 1 {
            sample
        } else {
            sample * (count - 1) / (samples - 1)
        };
        let row = index / weight.n;
        let col = index % weight.n;
        let raw = (0..weight.k)
            .map(|k| {
                f64::from(decode(encoded[row * weight.k + k].0))
                    * f64::from(decode(weight.codes[col * weight.k + k]))
            })
            .sum::<f64>();
        let expected = raw * f64::from(scales[row]) * f64::from(weight.scales[col]);
        native_error = native_error.max((f64::from(native[index]) - expected).abs());
        vendor_error = vendor_error.max((f64::from(vendor[index]) - expected).abs());
        let tolerance = expected.abs().mul_add(2e-5, 2e-4);
        if (f64::from(native[index]) - expected).abs() > tolerance
            || (f64::from(vendor[index]) - expected).abs() > tolerance
        {
            failed += 1;
            if failures.len() < 4 {
                failures.push(serde_json::json!({
                    "row": row, "column": col, "native": native[index],
                    "vendor": vendor[index], "expected": expected, "tolerance": tolerance,
                }));
            }
        }
    }
    serde_json::json!({
        "arithmetic": "F64 sum of decoded original input/weight codes, then xs and ws",
        "samples": samples, "full_outputs": full,
        "native_max_absolute_error": native_error, "vendor_max_absolute_error": vendor_error,
        "failed_samples": failed, "failures": failures, "within_numerical_contract": failed == 0,
    })
}

fn compare_case(
    device: &CudaDevice,
    weight: &Weight,
    rows: usize,
    performance: bool,
    independent: bool,
) -> TestResult {
    let mut inputs = Inputs::new(device, weight, rows)?;
    let mut native = api::zeros::<f32>(&[rows, weight.n]).sync_on(&device.stream)?;
    let mut vendor = api::zeros::<f32>(&[rows, weight.n]).sync_on(&device.stream)?;
    assert!(
        inputs.vendor.is_enabled(),
        "FP8 Lt scalar/F32 API unavailable"
    );
    assert!(
        inputs
            .vendor
            .prepare(&inputs.codes, &weight.device_codes, &vendor)?,
        "Lt zero-workspace algorithm unavailable for {rows}x{}x{}",
        weight.n,
        weight.k
    );

    let quantizer = Graph::scope(&device.stream, |scope| inputs.quantize(scope))?;
    quantizer.launch().sync_on(&device.stream)?;
    let native_raw = Graph::scope(&device.stream, |scope| {
        inputs.matmul(scope, &mut native, false)
    })?;
    let vendor_raw = Graph::scope(&device.stream, |scope| {
        inputs.matmul(scope, &mut vendor, true)
    })?;
    let native_total = Graph::scope(&device.stream, |scope| {
        inputs.quantize(scope)?;
        inputs.matmul(scope, &mut native, false)
    })?;
    let vendor_total = Graph::scope(&device.stream, |scope| {
        inputs.quantize(scope)?;
        inputs.matmul(scope, &mut vendor, true)
    })?;
    // The inputs own the context, descriptors, codes and scales until all graphs below are dropped.
    native_raw.launch().sync_on(&device.stream)?;
    vendor_raw.launch().sync_on(&device.stream)?;
    let native = Arc::new(native);
    let vendor = Arc::new(vendor);
    let native_values = (&native).to_host_vec().sync_on(&device.stream)?;
    let vendor_values = (&vendor).to_host_vec().sync_on(&device.stream)?;
    let same_quantized = compare_outputs(&native_values, &vendor_values)?;
    // Snapshot FP8 codes outside timed scopes; the original owned tensor remains live
    // for subsequent graph replays. F32 scale readback borrows its owner directly.
    let encoded = inputs
        .codes
        .dup()
        .sync_on(&device.stream)?
        .to_host_vec()
        .sync_on(&device.stream)?;
    let scales = device.read_borrowed(&inputs.scales)?;
    validate_quantizer(&inputs, &encoded, &scales, independent);
    let cpu = oracle(
        &inputs,
        &encoded,
        &scales,
        &native_values,
        &vendor_values,
        independent,
    );
    native_total.launch().sync_on(&device.stream)?;
    vendor_total.launch().sync_on(&device.stream)?;
    let native_full = (&native).to_host_vec().sync_on(&device.stream)?;
    let vendor_full = (&vendor).to_host_vec().sync_on(&device.stream)?;
    let full_quantization = compare_outputs(&native_full, &vendor_full)?;
    let native_quantizer_stable = native_values
        .iter()
        .zip(&native_full)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let vendor_quantizer_stable = vendor_values
        .iter()
        .zip(&vendor_full)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    check_case_contract(
        weight,
        rows,
        (&same_quantized, &full_quantization, &cpu),
        (native_quantizer_stable, vendor_quantizer_stable),
    )?;
    if !performance {
        return Ok(());
    }
    measure_graphs(
        device,
        weight,
        rows,
        [
            (
                "prequantized GEMM plus scale epilogue",
                &native_raw,
                &vendor_raw,
            ),
            (
                "unchanged quantizer plus GEMM plus scale epilogue",
                &native_total,
                &vendor_total,
            ),
        ],
    )
}

fn device(performance: bool) -> TestResult<CudaDevice> {
    if performance {
        if cfg!(debug_assertions) {
            return Err("performance requires release build".into());
        }
        let compile = CompileOptions::default();
        if compile.device_debug || compile.opt_level.is_some_and(|level| level != 3) {
            return Err("performance requires CUDA JIT O3 without device debug".into());
        }
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    println!(
        "{}",
        serde_json::json!({
            "kind": "device", "device": device.profile()?, "release": !cfg!(debug_assertions),
            "requested_scope": "unchanged FP8 token/channel arithmetic",
            "fast_accum": false, "vendor_workspace_bytes": 0,
        })
    );
    Ok(device)
}
#[test]
#[ignore = "requires CUDA; run serially inside safe-run"]
fn vendor_fp8_same_quantized_correctness() -> TestResult {
    let device = device(false)?;
    for (n, k) in [(64, 256), (128, 512)] {
        let weight = Weight::synthetic(&device, n, k)?;
        for rows in ROWS {
            compare_case(&device, &weight, rows, false, true)?;
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA; run release serially inside safe-run"]
fn vendor_fp8_synthetic_performance_comparison() -> TestResult {
    let device = device(true)?;
    for (n, k) in [(17408, 5120), (5120, 17408), (5120, 5120)] {
        let weight = Weight::synthetic(&device, n, k)?;
        for rows in ROWS {
            compare_case(&device, &weight, rows, true, false)?;
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA and INFER_TEST_MODEL; run release serially inside safe-run"]
fn vendor_fp8_checkpoint_comparison() -> TestResult {
    let root = var("INFER_TEST_MODEL")?;
    let package = QuantizedPackage::open(root, infer_core::ModelId::ONE)?;
    let selected: Vec<(String, DeviceWeight)> = if let Ok(names) = var("INFER_TEST_FP8_WEIGHTS") {
        names
            .split(',')
            .map(|name| {
                package
                    .weights
                    .get(name)
                    .cloned()
                    .map(|source| (name.to_owned(), source))
                    .ok_or_else(|| format!("unknown canonical checkpoint weight {name}"))
            })
            .collect::<Result<_, _>>()?
    } else {
        let mut selected = Vec::new();
        for family in ["in_proj_qkv", "mlp.gate_proj", "mlp.down_proj", "lm_head"] {
            let (name, source) = package
                .weights
                .iter()
                .find(|(_, source)| {
                    source.encoding == WeightEncoding::Fp8Channel
                        && source.fp8_token_input
                        && source.data.name.contains(family)
                })
                .ok_or_else(|| {
                    format!("no checkpoint FP8 token/channel projection matching {family}")
                })?;
            selected.push((name.clone(), source.clone()));
        }
        selected
    };
    let device = device(true)?;
    for (name, source) in selected {
        let weight = Weight::checkpoint(&device, &package, &name, &source)?;
        for rows in ROWS {
            compare_case(&device, &weight, rows, true, false)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware and a real FP8 channel checkpoint; root runs release serially"]
fn vendor_fp8_verified_chooser_checkpoint_comparison() -> TestResult {
    use crate::{
        mlp::ProjectionWeight,
        tuning::{
            fp8_gemm::{self, Choice},
            resident_gemm::ResidentTuningSession,
        },
    };
    let package = QuantizedPackage::open(var("INFER_TEST_MODEL")?, infer_core::ModelId::ONE)?;
    let device = device(true)?;
    let session = ResidentTuningSession::default();
    for family in ["mlp.gate_proj", "mlp.down_proj", "in_proj_qkv", "lm_head"] {
        let (name, source) = package
            .weights
            .iter()
            .find(|(_, source)| {
                source.encoding == WeightEncoding::Fp8Channel
                    && source.fp8_token_input
                    && source.data.name.contains(family)
            })
            .ok_or_else(|| format!("no FP8 channel checkpoint family {family}"))?;
        let weight = Weight::checkpoint(&device, &package, name, source)?;
        let projection = ProjectionWeight::Fp8(
            Arc::clone(&weight.device_codes),
            Arc::clone(&weight.device_scales),
        );
        // The same model allowance applies to all shapes; larger prompt shapes exercise the measured Lt wins first.
        for rows in [128, 64, 12, 1] {
            let mut inputs = Inputs::new(&device, &weight, rows)?;
            let choice = fp8_gemm::select(
                &device,
                &mut inputs.vendor,
                &projection,
                [rows, weight.n, weight.k],
                true,
                &session,
            );
            let mut baseline = api::zeros::<f32>(&[rows, weight.n]).sync_on(&device.stream)?;
            let mut output = api::zeros::<f32>(&[rows, weight.n]).sync_on(&device.stream)?;
            let base = Graph::scope(&device.stream, |scope| {
                inputs.quantize(scope)?;
                Choice::baseline(rows).record(
                    scope,
                    &inputs.vendor,
                    &inputs.codes,
                    &weight.device_codes,
                    &mut baseline,
                    &inputs.scales,
                    &weight.device_scales,
                    weight.k,
                )
            })?;
            let candidate = Graph::scope(&device.stream, |scope| {
                inputs.quantize(scope)?;
                choice.record(
                    scope,
                    &inputs.vendor,
                    &inputs.codes,
                    &weight.device_codes,
                    &mut output,
                    &inputs.scales,
                    &weight.device_scales,
                    weight.k,
                )
            })?;
            base.launch().sync_on(&device.stream)?;
            candidate.launch().sync_on(&device.stream)?;
            let expected = device.read_borrowed(&baseline)?;
            let actual = device.read_borrowed(&output)?;
            assert!(
                expected
                    .iter()
                    .zip(&actual)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
            );
            measure_choice(&device, name, &weight, rows, &choice, &base, &candidate)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; root runs release serially"]
fn vendor_fp8_typed_algorithm_restore_roundtrip() -> TestResult {
    use crate::tuning::fp8_gemm::AlgorithmConfig;
    let device = device(true)?;
    let weight = Weight::synthetic(&device, 128, 512)?;
    let mut inputs = Inputs::new(&device, &weight, 128)?;
    let mut native = api::zeros::<f32>(&[128, 128]).sync_on(&device.stream)?;
    let mut restored_output = api::zeros::<f32>(&[128, 128]).sync_on(&device.stream)?;
    let quantizer = Graph::scope(&device.stream, |scope| inputs.quantize(scope))?;
    quantizer.launch().sync_on(&device.stream)?;
    let configs =
        inputs
            .vendor
            .prepare_candidates(&inputs.codes, &weight.device_codes, &restored_output)?;
    let selected = configs
        .first()
        .ok_or("no supported zero-workspace typed FP8 algorithm")?;
    let encoded_config = serde_json::to_string(selected)?;
    let config: AlgorithmConfig = serde_json::from_str(&encoded_config)?;
    assert_eq!(*selected, config);
    assert!(inputs.vendor.heuristic_query_count() > 0);
    let mut restored = Support::new_with(&device, true);
    restored.retain_channels(&weight.device_codes, &weight.device_scales);
    assert_eq!(restored.heuristic_query_count(), 0);
    assert!(
        restored.prepare_config(
            &inputs.codes,
            &weight.device_codes,
            &restored_output,
            config
        )?,
        "typed restore must execute SDK init/set/check and real warmup on a fresh handle"
    );
    assert_eq!(
        restored.heuristic_query_count(),
        0,
        "fresh typed restoration must work without a heuristic query"
    );
    assert!(Arc::strong_count(&weight.device_codes) >= 2);
    assert!(Arc::strong_count(&weight.device_scales) >= 2);
    let native_graph = Graph::scope(&device.stream, |scope| {
        inputs.quantize(scope)?;
        inputs.matmul(scope, &mut native, false)
    })?;
    let restored_graph = Graph::scope(&device.stream, |scope| {
        inputs.quantize(scope)?;
        if !restored.record_config(
            scope,
            config,
            &inputs.codes,
            &weight.device_codes,
            &mut restored_output,
            &inputs.scales,
            &weight.device_scales,
        )? {
            return Err(DeviceError::Internal(
                "typed FP8 algorithm was not captured".into(),
            ));
        }
        Ok(())
    })?;
    // Both graphs are declared after the lifetime owners and are destroyed before them.
    native_graph.launch().sync_on(&device.stream)?;
    restored_graph.launch().sync_on(&device.stream)?;
    let native_values = device.read_borrowed(&native)?;
    let restored_values = device.read_borrowed(&restored_output)?;
    let comparison = compare_outputs(&native_values, &restored_values)?;
    assert_eq!(comparison["all_outputs_identical"], true);
    assert_eq!(comparison["nonfinite_outputs"], 0);
    let codes = inputs
        .codes
        .dup()
        .sync_on(&device.stream)?
        .to_host_vec()
        .sync_on(&device.stream)?;
    let scales = device.read_borrowed(&inputs.scales)?;
    validate_quantizer(&inputs, &codes, &scales, true);
    let cpu = oracle(
        &inputs,
        &codes,
        &scales,
        &native_values,
        &restored_values,
        true,
    );
    assert_eq!(cpu["within_numerical_contract"], true);
    println!(
        "{}",
        serde_json::json!({"kind":"typed_algorithm_restore", "shape":[128,128,512],
        "algorithm":config,"fresh_handle":true,"restore_heuristic_queries":restored.heuristic_query_count(),
        "original_codes_and_scales_retained":true,"quantizer_inclusive":true,
        "finite_bitwise":comparison,"cpu_oracle":cpu,"fast_accum":false,"workspace_bytes":0,
        "profile_note":"this fixture validates typed restoration and numerical acceptance; SDK does not guarantee all-input same-K arithmetic"})
    );
    Ok(())
}

type GraphPair<'a> = (&'a str, &'a Graph, &'a Graph);

fn measure_graphs(
    device: &CudaDevice,
    weight: &Weight,
    rows: usize,
    graphs: [GraphPair<'_>; 2],
) -> TestResult {
    let rounds = var("INFER_TEST_FP8_ROUNDS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(3_usize);
    for round in 0..rounds {
        for clear_l2 in [true, false] {
            let options = BenchOptions {
                warmup: Duration::from_millis(50),
                rep: Duration::from_millis(100),
                min_reps: 30,
                max_reps: 100,
                clear_l2,
            };
            for (scope, left, right) in graphs {
                let (base, candidate) = do_bench_paired(
                    &device.stream,
                    &options,
                    |_| {
                        left.launch()
                            .sync_on(&device.stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    },
                    |_| {
                        right
                            .launch()
                            .sync_on(&device.stream)
                            .map_err(|e| tensor_error(&e.to_string()))
                    },
                )?;
                println!(
                    "{}",
                    serde_json::json!({
                        "kind": "performance", "weight": weight.name, "shape": [rows, weight.n, weight.k],
                        "round": round, "clear_l2": clear_l2, "scope": scope,
                        "native_tile": native_tile(rows), "vendor_workspace_bytes": 0,
                        "native_ms": base.median_ms(), "vendor_ms": candidate.median_ms(),
                        "speedup": base.median_ms() / candidate.median_ms(),
                        "native_p95_ms": base.quantile_ms(0.95), "vendor_p95_ms": candidate.quantile_ms(0.95),
                        "native_samples_ms": base.times_ms(), "vendor_samples_ms": candidate.times_ms(),
                    })
                );
            }
        }
    }
    Ok(())
}

fn measure_choice(
    device: &CudaDevice,
    name: &str,
    weight: &Weight,
    rows: usize,
    choice: &Fp8GemmChoice,
    base: &Graph,
    candidate: &Graph,
) -> TestResult {
    for clear_l2 in [true, false] {
        let options = BenchOptions {
            warmup: Duration::from_millis(50),
            rep: Duration::from_millis(100),
            min_reps: 10,
            max_reps: 30,
            clear_l2,
        };
        let (a, b) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                base.launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
            |_| {
                candidate
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
        )?;
        println!(
            "{}",
            serde_json::json!({"kind":"verified_chooser","weight":name,"shape":[rows,weight.n,weight.k],"choice":choice,
                    "finite_bitwise":true,"quantizer_inclusive":true,"clear_l2":clear_l2,"baseline_ms":a.times_ms(),"candidate_ms":b.times_ms(),"speedup":a.median_ms()/b.median_ms()})
        );
    }
    Ok(())
}

fn check_case_contract(
    weight: &Weight,
    rows: usize,
    reports: (&serde_json::Value, &serde_json::Value, &serde_json::Value),
    stable: (bool, bool),
) -> TestResult {
    let (same_quantized, full_quantization, cpu) = reports;
    let (native_quantizer_stable, vendor_quantizer_stable) = stable;
    println!(
        "{}",
        serde_json::json!({
            "kind": "correctness", "weight": weight.name, "shape": [rows, weight.n, weight.k],
            "same_quantized": same_quantized, "full_quantization": full_quantization, "cpu_oracle": cpu,
            "vendor_workspace_bytes": 0, "fast_accum": false,
            "native_quantizer_stable": native_quantizer_stable, "vendor_quantizer_stable": vendor_quantizer_stable,
            "profile_note": "non-bitwise results are not eligible to silently replace the original exact arithmetic profile",
        })
    );
    if same_quantized["within_numerical_contract"] != true
        || full_quantization["within_numerical_contract"] != true
        || cpu["within_numerical_contract"] != true
        || !native_quantizer_stable
        || !vendor_quantizer_stable
    {
        return Err(format!(
            "{} {rows}x{}x{} failed the recorded numerical contract",
            weight.name, weight.n, weight.k
        )
        .into());
    }
    Ok(())
}
