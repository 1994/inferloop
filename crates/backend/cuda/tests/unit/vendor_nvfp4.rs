//! Actual-weight FP4 cuBLASLt screen, including both GPU scale packs and preserving F32 output.
use crate::device::CudaDevice;
use crate::loading::{LoadOptions, LoadedModel};
use crate::mlp::ProjectionWeight;
use crate::resident::nvfp4_gemm::kernels;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use infer_core::ModelId;
use std::env::var;
use std::error::Error as StdError;
use std::time::Duration;

fn test_rows() -> TestResult<usize> {
    let rows = var("INFER_TEST_NVFP4_ROWS")
        .unwrap_or_else(|_| "1024".into())
        .parse::<usize>()?;
    if ![512, 1024, 2048].contains(&rows) {
        return Err("rows must be 512, 1024 or 2048".into());
    }
    Ok(rows)
}
const OUTPUTS: usize = 5120;
const COLUMNS: usize = 17408;
type TestResult<T = ()> = Result<T, Box<dyn StdError>>;
type Quantized = (Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>);

use super::super::context;
use super::{Context, Operands, Prepared, Shape, append, make_candidate};
use crate::loading::nvfp4_fixture::{Layer, layers};
struct Packed {
    scratch: Tensor<f8e4m3fn>,
}
fn capture(
    device: &CudaDevice,
    layers: &[Layer],
    input: &Tensor<f32>,
    quantized: &mut Quantized,
    outputs: &mut [Tensor<f32>],
    packed: &mut Packed,
    vendor: Option<(&Arc<Context>, &Arc<Prepared>)>,
) -> TestResult<CudaGraph<()>> {
    let (q, qs) = quantized;
    Ok(CudaGraph::scope(&device.stream, |scope| {
        for (layer, out) in layers.iter().zip(outputs) {
            scope.record(
                kernels::quantize_rows(
                    (&mut *q).partition([16, 256]),
                    (&mut *qs).partition([16, 32]),
                    input,
                    layer.scale,
                )
                .generics(vec![COLUMNS.to_string()]),
            )?;
            let ProjectionWeight::Fp4(w, s, g) = &layer.weight else {
                return Err(DeviceError::Launch("FP4 required".into()));
            };
            if let Some((context, prepared)) = vendor {
                let shape = Shape {
                    m: input.size() / COLUMNS,
                    n: OUTPUTS,
                    k: COLUMNS,
                };
                let mut op = Operands {
                    weight: &layer.weight,
                    scale: layer.scale,
                    q: &*q,
                    qs: &*qs,
                    scratch: &mut packed.scratch,
                    output: out,
                };
                assert_eq!(op.shape()?, Some(shape));
                append(scope, context, prepared, &mut op, shape)?;
            } else {
                scope.record(
                    kernels::packed_prefill(
                        out.partition([128, 128]),
                        &*q,
                        &*qs,
                        w,
                        s,
                        1.0 / (g * layer.scale),
                    )
                    .generics(
                        [COLUMNS, 128, 128]
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                    ),
                )?;
            }
        }
        Ok(())
    })?)
}

fn compare(
    device: &CudaDevice,
    a: &[Tensor<f32>],
    b: &[Tensor<f32>],
    algorithm: usize,
) -> TestResult {
    let mut elements = 0usize;
    let mut bit_differences = 0usize;
    let mut max_scaled_error = 0.0_f32;
    for (layer, (a, b)) in a.iter().zip(b).enumerate() {
        let a = device.read_borrowed(a)?;
        let b = device.read_borrowed(b)?;
        assert_eq!(a.len(), test_rows()? * OUTPUTS);
        assert_eq!(a.len(), b.len());
        for (index, (&x, &y)) in a.iter().zip(&b).enumerate() {
            assert!(
                x.is_finite() && y.is_finite(),
                "nonfinite layer {layer} at {index}"
            );
            let scaled = (x - y).abs() / x.abs().max(1.0);
            max_scaled_error = max_scaled_error.max(scaled);
            assert!(
                scaled <= 5.0e-4,
                "layer {layer} index {index}: {x} vs {y}, error {scaled}"
            );
            bit_differences += usize::from(x.to_bits() != y.to_bits());
        }
        elements += a.len();
    }
    println!(
        "{}",
        serde_json::json!({"kind":"nvfp4_cublas_full_output","algorithm":algorithm,
        "finite_elements":elements,"max_scaled_error":max_scaled_error,"bit_differences":bit_differences,
        "tolerance":5.0e-4,"reference":"existing native GEMM; independent scalar reference required before production"})
    );
    Ok(())
}

fn measure(
    device: &CudaDevice,
    a: &CudaGraph<()>,
    b: &CudaGraph<()>,
    algorithm: usize,
) -> TestResult {
    let rows = test_rows()?;
    let canary = BenchOptions {
        warmup: Duration::ZERO,
        rep: Duration::ZERO,
        min_reps: 1,
        max_reps: 1,
        clear_l2: false,
    };
    let (base, candidate) = do_bench_paired(
        &device.stream,
        &canary,
        |_| {
            a.launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
        |_| {
            b.launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
    )?;
    println!(
        "{}",
        serde_json::json!({"kind":"nvfp4_cublas_canary","algorithm":algorithm,"baseline_ms":base.median_ms(),
        "candidate_ms":candidate.median_ms(),"reject":candidate.median_ms()>base.median_ms()*1.10})
    );
    if candidate.median_ms() > base.median_ms() * 1.10 {
        return Ok(());
    }
    for reverse in [false, true] {
        let (first, second) = if reverse { (b, a) } else { (a, b) };
        let options = BenchOptions {
            warmup: Duration::from_millis(50),
            rep: Duration::from_millis(150),
            min_reps: 10,
            max_reps: 30,
            clear_l2: false,
        };
        let (x, y) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                first
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
            |_| {
                second
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|e| tensor_error(&e.to_string()))
            },
        )?;
        let (base, candidate) = if reverse { (y, x) } else { (x, y) };
        println!(
            "{}",
            serde_json::json!({
                "kind":"nvfp4_down_cublas","algorithm":algorithm,"shape":[rows,OUTPUTS,COLUMNS],
                "reverse":reverse,"layers":56,"baseline_ms":base.median_ms(),
                "candidate_ms":candidate.median_ms(),"speedup":base.median_ms()/candidate.median_ms(),
                "baseline_samples_ms":base.times_ms(),"candidate_samples_ms":candidate.times_ms(),
                "scope":"quantization plus paired SF GPU pack into reused F32 storage plus F32 GEMM, rotating all 56 actual checkpoint weights"
            })
        );
    }
    Ok(())
}

#[test]
#[ignore = "actual 27B checkpoint, CUDA, release and protected runner"]
fn actual_down_cublas_fp4_screen() -> TestResult {
    let options = CompileOptions::default();
    if cfg!(debug_assertions) || options.device_debug || options.opt_level.is_some_and(|v| v != 3) {
        return Err("release and CUDA O3 without device debug required".into());
    }
    let rows = test_rows()?;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let loaded = LoadedModel::open(
        device.clone(),
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let layers = layers(&device, &loaded)?;
    let weight_bytes = layers.len() * OUTPUTS * (COLUMNS / 2 + COLUMNS / 16);
    assert!(weight_bytes > usize::try_from(device.profile()?.l2_cache_bytes)?);
    let input = device.upload(
        (0..rows * COLUMNS)
            .map(|i| (f32::from(u16::try_from(i % 211).unwrap()) - 105.0) / 37.0)
            .collect::<Vec<_>>(),
        &[rows, COLUMNS],
    )?;
    let mut quantized = (
        api::zeros::<f4e2m1fnx2>(&[rows, COLUMNS / 2]).sync_on(&device.stream)?,
        api::zeros::<f8e4m3fn>(&[rows, COLUMNS / 16]).sync_on(&device.stream)?,
    );
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in &layers {
        a.push(api::zeros::<f32>(&[rows, OUTPUTS]).sync_on(&device.stream)?);
        b.push(api::zeros::<f32>(&[rows, OUTPUTS]).sync_on(&device.stream)?);
    }

    // Reuse storage initially allocated as F32, without byte assembly or a second allocation.
    let floats = Arc::new(api::zeros::<f32>(&[rows * COLUMNS]).sync_on(&device.stream)?);
    let float_pointer = floats.device_pointer().cu_deviceptr();
    let bytes = floats.reinterpret::<f8e4m3fn>(&[rows / 2, COLUMNS / 16 * 128])?;
    assert_eq!(float_pointer, bytes.device_pointer().cu_deviceptr());
    drop(floats);
    let scratch = Arc::try_unwrap(bytes).map_err(|_| "exclusive byte alias required")?;
    let mut packed = Packed { scratch };
    let context = context(&device).ok_or("cuBLASLt unavailable")?;
    let candidates = make_candidate(
        &context,
        Shape {
            m: rows,
            n: OUTPUTS,
            k: COLUMNS,
        },
        &packed.scratch,
    )?
    .into_iter()
    .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err("no zero-workspace F32 FP4 candidates".into());
    }
    println!(
        "{}",
        serde_json::json!({"kind":"nvfp4_cublas_candidates","count":candidates.len(),
       "packed_scratch_bytes":(OUTPUTS+rows)*COLUMNS/16,"rows":rows,"storage_reused_from":"F32 product-shaped allocation","storage_bytes":rows*COLUMNS*4,"same_pointer":true})
    );
    let ga = capture(
        &device,
        &layers,
        &input,
        &mut quantized,
        &mut a,
        &mut packed,
        None,
    )?;
    ga.launch().sync_on(&device.stream)?;
    for (algorithm, prepared) in candidates.iter().take(1).enumerate() {
        let gb = capture(
            &device,
            &layers,
            &input,
            &mut quantized,
            &mut b,
            &mut packed,
            Some((&context, prepared)),
        )?;
        gb.launch().sync_on(&device.stream)?;
        compare(&device, &a, &b, algorithm)?;
        measure(&device, &ga, &gb, algorithm)?;
    }
    Ok(())
}
