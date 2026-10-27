//! SDK FP4/F32 results checked against independent scalar dequantization.
use super::{Operands, Shape, append, make_candidate};
use crate::device::{CudaDevice, vendor_fp8};
use crate::mlp::ProjectionWeight;
use crate::tuning::resident_gemm::{GemmConfig, ResidentTuningSession};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use std::error::Error;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn fp4(code: u8) -> f64 {
    let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code & 7)];
    if code & 8 == 0 { magnitude } else { -magnitude }
}
fn fp8(code: u8) -> f64 {
    let exponent = i32::from((code >> 3) & 15);
    let mantissa = f64::from(code & 7);
    if exponent == 0 {
        mantissa * 2.0_f64.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f64.powi(exponent - 7)
    }
}
fn element(codes: &[u8], scales: &[u8], row: usize, column: usize, k: usize) -> f64 {
    let byte = codes[row * k / 2 + column / 2];
    let code = if column.is_multiple_of(2) {
        byte & 15
    } else {
        byte >> 4
    };
    fp4(code) * fp8(scales[row * (k / 16) + column / 16])
}
#[test]
#[ignore = "requires CUDA/Lt FP4 hardware, release, and protected runner"]
fn vendor_fp4_matches_independent_full_scalar_output() -> TestResult {
    if cfg!(debug_assertions) {
        return Err("release required".into());
    }
    let device = CudaDevice::new(0)?;
    CudaDevice::enable_kernel_cache()?;
    let (m, n, k) = (128usize, 128usize, 256usize);
    let qc: Vec<u8> = (0..m * k / 2)
        .map(|i| u8::try_from((i * 17 + i / 13) % 256).unwrap())
        .collect();
    let wc: Vec<u8> = (0..n * k / 2)
        .map(|i| u8::try_from((i * 29 + i / 7) % 256).unwrap())
        .collect();
    let qs: Vec<u8> = (0..m * k / 16)
        .map(|i| u8::try_from(i * 7 % 127).unwrap())
        .collect();
    let ws: Vec<u8> = (0..n * k / 16)
        .map(|i| u8::try_from(i * 13 % 127).unwrap())
        .collect();
    let q = device.upload(
        qc.iter().map(|&v| f4e2m1fnx2::from_bits(v)).collect(),
        &[m, k / 2],
    )?;
    let input_scales = device.upload(qs.iter().map(|&v| f8e4m3fn(v)).collect(), &[m, k / 16])?;
    let weight = ProjectionWeight::Fp4(
        device.upload(
            wc.iter().map(|&v| f4e2m1fnx2::from_bits(v)).collect(),
            &[n, k / 2],
        )?,
        device.upload(ws.iter().map(|&v| f8e4m3fn(v)).collect(), &[n, k / 16])?,
        1.0,
    );
    let floats = Arc::new(api::zeros::<f32>(&[m * k]).sync_on(&device.stream)?);
    let scratch = floats.reinterpret::<f8e4m3fn>(&[m / 2, k / 16 * 128])?;
    assert_eq!(
        scratch.device_pointer().cu_deviceptr(),
        floats.device_pointer().cu_deviceptr()
    );
    drop(floats);
    let mut scratch = Arc::try_unwrap(scratch).map_err(|_| "exclusive scratch")?;
    let mut output = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
    let shared = vendor_fp8::Support::new_with(&device, true);
    let context = shared.context.ok_or("Lt unavailable")?;
    let shape = Shape { m, n, k };
    let prepared =
        make_candidate(&context, shape, &scratch)?.ok_or("FP4 F32 algorithm unavailable")?;
    let mut op = Operands {
        weight: &weight,
        scale: 1.0,
        q: &q,
        qs: &input_scales,
        scratch: &mut scratch,
        output: &mut output,
    };
    assert!(op.shape()?.is_some());
    let graph = CudaGraph::scope(&device.stream, |scope| {
        append(scope, &context, &prepared, &mut op, shape)
    })?;
    graph.launch().sync_on(&device.stream)?;
    let actual = device.read_borrowed(&output)?;
    let mut max_scaled = 0.0_f64;
    for row in 0..m {
        for col in 0..n {
            let reference = (0..k)
                .map(|c| element(&qc, &qs, row, c, k) * element(&wc, &ws, col, c, k))
                .sum::<f64>();
            let value = f64::from(actual[row * n + col]);
            let scaled = (value - reference).abs() / reference.abs().max(1.0);
            max_scaled = max_scaled.max(scaled);
            assert!(
                value.is_finite() && scaled <= 5.0e-4,
                "({row},{col}) {value} != {reference}, error {scaled}"
            );
        }
    }
    println!(
        "independent full scalar output: {} values, max_scaled_error={max_scaled}",
        actual.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA; rejected operands must record no vendor nodes"]
fn vendor_fp4_rejects_overlap_and_unprepared_operands() -> TestResult {
    let device = CudaDevice::new(0)?;
    let (m, n, k) = (128usize, 128usize, 128usize);
    let q = api::zeros::<f4e2m1fnx2>(&[m, k / 2]).sync_on(&device.stream)?;
    let qs = api::zeros::<f8e4m3fn>(&[m, k / 16]).sync_on(&device.stream)?;
    let weight = ProjectionWeight::Fp4(
        device.upload(vec![f4e2m1fnx2::from_bits(0x22); n * k / 2], &[n, k / 2])?,
        device.upload(vec![f8e4m3fn(56); n * k / 16], &[n, k / 16])?,
        1.0,
    );
    let source = Arc::new(api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?);
    let bytes = source.reinterpret::<f8e4m3fn>(&[m / 2, k / 16 * 128])?;
    let output = source.reinterpret::<f32>(&[m, n])?;
    drop(source);
    let mut scratch = Arc::try_unwrap(bytes).map_err(|_| "scratch metadata")?;
    let mut output = Arc::try_unwrap(output).map_err(|_| "output metadata")?;
    let mut fresh = api::zeros::<f8e4m3fn>(&[m / 2, k / 16 * 128]).sync_on(&device.stream)?;
    let mut support = super::Support::new(
        &vendor_fp8::Support::new_with(&device, true),
        &Arc::new(ResidentTuningSession::default()),
        false,
    );
    {
        let mut op = Operands {
            weight: &weight,
            scale: 1.0,
            q: &q,
            qs: &qs,
            scratch: &mut fresh,
            output: &mut output,
        };
        let before = super::confirmations();
        assert!(
            support
                .candidate(&mut op, GemmConfig::baseline(m))?
                .is_none()
        );
        assert_eq!(
            super::confirmations(),
            before,
            "no-autotune must not measure a new candidate"
        );
    }
    let graph = CudaGraph::scope(&device.stream, |scope| {
        let op = Operands {
            weight: &weight,
            scale: 1.0,
            q: &q,
            qs: &qs,
            scratch: &mut scratch,
            output: &mut output,
        };
        assert!(op.shape()?.is_none());
        assert!(!support.record(scope, op)?);
        let op = Operands {
            weight: &weight,
            scale: 1.0,
            q: &q,
            qs: &qs,
            scratch: &mut fresh,
            output: &mut output,
        };
        assert!(op.shape()?.is_some());
        assert!(!support.record(scope, op)?);
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let op = Operands {
                weight: &weight,
                scale,
                q: &q,
                qs: &qs,
                scratch: &mut fresh,
                output: &mut output,
            };
            assert!(op.shape()?.is_none());
            assert!(!support.record(scope, op)?);
        }
        Ok(())
    })?;
    graph.launch().sync_on(&device.stream)?;
    assert!(
        device
            .read_borrowed(&output)?
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    Ok(())
}

pub(super) fn evidence(value: &serde_json::Value) {
    println!("{value}");
}
