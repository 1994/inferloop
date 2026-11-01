use super::*;
use crate::device::{CudaDevice, device_error};
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;
use std::time::Instant;

#[expect(
    clippy::cast_precision_loss,
    reason = "a test-only generator; the low mantissa bits of the state are not meaningful"
)]
fn sample(seed: u64, index: usize) -> f32 {
    let mut state = seed
        .wrapping_add(index as u64)
        .wrapping_mul(6_364_136_223_846_793_005);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 29;
    ((state >> 40) as f32 / 8_388_608.0) - 1.0
}

/// cuBLAS must reproduce a projection against an f64 reference on the shapes the dense models
/// use, before anything is allowed to delegate to it.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
fn gemm_bf16_matches_reference() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    if !available() {
        eprintln!("cuBLAS unavailable; skipping");
        return Ok(());
    }
    for (m, n, k) in [
        (4_usize, 256_usize, 256_usize),
        (12, 2048, 2048),
        (128, 2048, 2048),
    ] {
        let activations: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32(sample(1, i))).collect();
        let weights: Vec<bf16> = (0..n * k).map(|i| bf16::from_f32(sample(2, i))).collect();
        let host_activations: Vec<f32> = activations.iter().map(|v| v.to_f32()).collect();
        let host_weights: Vec<f32> = weights.iter().map(|v| v.to_f32()).collect();
        let a = device.upload(activations, &[m, k])?;
        let b = device.upload(weights, &[n, k])?;
        let out = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
        // The first launch pays cuBLAS initialisation and its internal algorithm search, so
        // warm the shape and then time repeats.
        gemm_bf16(&device, &a, &b, &out)?;
        device.reclaim_barrier()?;
        let repeats = 50;
        let started = Instant::now();
        for _ in 0..repeats {
            gemm_bf16(&device, &a, &b, &out)?;
        }
        device.reclaim_barrier()?;
        let elapsed = started.elapsed() / repeats;
        let (worst, scale) =
            reference_error(&out, &host_activations, &host_weights, (m, n, k), &device)?;
        eprintln!(
            "m={m} n={n} k={k}: worst {worst:.3e} against reference scale {scale:.1} in {:.3} ms",
            elapsed.as_secs_f64() * 1.0e3
        );
        eprintln!(
            "device {}: captured gemm worst {worst:.3e} of {scale:.1}",
            device.ordinal()
        );
        assert!(
            worst <= 1.0e-2 * scale.max(1.0),
            "a per-instance cuBLAS context produced wrong output"
        );
    }
    Ok(())
}

/// The engine runs captured graphs, so the launch has to be recordable and replayable, with
/// its scalars living on the device rather than frozen into the graph.
#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test extents are small literals, so the narrowing to the kernel's i32 is exact"
)]
fn gemm_bf16_replays_inside_a_captured_graph() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    if !available() {
        eprintln!("cuBLAS unavailable; skipping");
        return Ok(());
    }
    let (rows, columns, contraction) = (12_usize, 2048_usize, 2048_usize);
    let activations: Vec<bf16> = (0..rows * contraction)
        .map(|i| bf16::from_f32(sample(3, i)))
        .collect();
    let weights: Vec<bf16> = (0..columns * contraction)
        .map(|i| bf16::from_f32(sample(4, i)))
        .collect();
    let a = device.upload(activations, &[rows, contraction])?;
    let b = device.upload(weights, &[columns, contraction])?;
    let out = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
    let context = context_for(&device).ok_or("no cuBLAS context")?;
    // cuBLAS does host-side work on the first use of a configuration (workspace and algorithm
    // selection), which a capturing stream rejects; run the shape once before recording it.
    gemm_bf16(&device, &a, &b, &out)?;
    device.reclaim_barrier()?;
    let graph = Graph::scope(&device.stream, |scope| {
        let op = GemmBf16::new(
            context,
            rows as i32,
            columns as i32,
            contraction as i32,
            &a,
            &b,
            &out,
        )
        .map_err(|error| DeviceError::Internal(error.to_string()))?;
        scope.record(op)?;
        Ok(())
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    device.reclaim_barrier()?;
    let captured = out.to_host_vec().sync_on(&device.stream)?;
    let direct = api::zeros::<f32>(&[rows, columns]).sync_on(&device.stream)?;
    gemm_bf16(&device, &a, &b, &direct)?;
    device.reclaim_barrier()?;
    let reference = direct.to_host_vec().sync_on(&device.stream)?;
    let gap = captured
        .iter()
        .zip(&reference)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("captured versus direct launch: max_abs={gap:e}");
    assert_eq!(gap, 0.0, "the captured launch differs from the direct one");
    Ok(())
}

/// Worst absolute error of a device result against an f64 reference, with the reference scale.
fn reference_error(
    out: &Tensor<f32>,
    host_activations: &[f32],
    host_weights: &[f32],
    (rows, columns, contraction): (usize, usize, usize),
    device: &CudaDevice,
) -> Result<(f64, f64), Box<dyn Error>> {
    let actual = device
        .read_borrowed(out)
        .map_err(|error| error.to_string())?;
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..rows {
        for j in 0..columns {
            let mut expected = 0.0_f64;
            for d in 0..contraction {
                expected = f64::mul_add(
                    f64::from(host_activations[i * contraction + d]),
                    f64::from(host_weights[j * contraction + d]),
                    expected,
                );
            }
            scale = scale.max(expected.abs());
            worst = worst.max((f64::from(actual[i * columns + j]) - expected).abs());
        }
    }
    Ok((worst, scale))
}
