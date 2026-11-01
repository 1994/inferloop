//! Head-to-head between the prompt GEMM and the vendor one, on the shapes the 2B model hits.
//!
//! Adopting a delegated projection changes model arithmetic, so it is worth measuring against the
//! kernel it would replace before any wiring happens. The F32 activation has to be narrowed to
//! BF16 first, because cuBLAS wants both operands in one type, and that cast is part of the price.
use super::gemm;
use crate::device::cublaslt::{GemmBf16, Support, available};
use crate::device::{CudaDevice, device_error};
use crate::kernels::linear::cast_bf16;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

const CAST_TILE: i32 = 1024;

#[expect(
    clippy::cast_precision_loss,
    reason = "a benchmark-only generator; the low mantissa bits of the state are not meaningful"
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

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run"]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "benchmark extents are small literals, so the narrowing to the kernel's i32 is exact"
)]
fn cublas_versus_the_prompt_gemm_on_model_shapes() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    if !available() {
        eprintln!("cuBLAS unavailable; skipping");
        return Ok(());
    }
    let mut support = Support::new_with(&device, true);
    if !support.is_enabled() {
        eprintln!("cuBLAS context unavailable; skipping");
        return Ok(());
    }
    let m = 128_usize;
    for (n, k) in [(2048_usize, 2048_usize), (12288, 2048), (2048, 8192)] {
        let inputs: Vec<f32> = (0..m * k).map(|i| sample(5, i) * 0.1).collect();
        let weights: Vec<bf16> = (0..n * k).map(|i| bf16::from_f32(sample(6, i))).collect();
        let input = device.upload(inputs, &[m, k])?;
        let weight = device.upload(weights, &[n, k])?;
        let mut native = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
        // Our prompt kernel tiles the output 32x64 and the K loop by 64.
        let kernel_graph = Graph::scope(&device.stream, |scope| {
            scope.record(
                gemm::dense((&mut native).partition([32, 64]), &input, &weight)
                    .generics(vec![k.to_string()]),
            )?;
            Ok(())
        })
        .map_err(device_error)?;

        let mut narrowed = api::zeros::<bf16>(&[m * k]).sync_on(&device.stream)?;
        let delegated = api::zeros::<f32>(&[m, n]).sync_on(&device.stream)?;
        let cast_graph = Graph::scope(&device.stream, |scope| {
            scope.record(
                cast_bf16(
                    (&mut narrowed).partition([CAST_TILE as usize]),
                    &input.view(&[m * k])?,
                )
                .generics(vec![CAST_TILE.to_string()]),
            )?;
            Ok(())
        })
        .map_err(device_error)?;
        // cuBLAS does host-side work the first time it sees a configuration and a capturing
        // stream rejects it, so every shape is warmed here, outside the graph, exactly as the
        // production path does it. Warming after the capture would not have helped.
        support.warm(m, &weight, n, k)?;
        let gemm_graph = Graph::scope(&device.stream, |scope| {
            let op = GemmBf16::new(
                Arc::clone(
                    support
                        .context()
                        .ok_or_else(|| DeviceError::Internal("no cuBLAS context".to_string()))?,
                ),
                m as i32,
                n as i32,
                k as i32,
                &narrowed,
                &weight,
                &delegated,
            )
            .map_err(|error| DeviceError::Internal(error.to_string()))?;
            scope.record(op)?;
            Ok(())
        })
        .map_err(device_error)?;
        // Warm the native and cast paths before timing; the vendor one was warmed above.
        kernel_graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        cast_graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        device.reclaim_barrier()?;
        let repeats = 30;
        let time = |graph: &Graph| -> Result<f64, Box<dyn Error>> {
            let started = Instant::now();
            for _ in 0..repeats {
                graph
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(device_error)?;
            }
            device.reclaim_barrier()?;
            Ok(started.elapsed().as_secs_f64() / f64::from(repeats) * 1.0e6)
        };
        let ours = time(&kernel_graph)?;
        let cast = time(&cast_graph)?;
        let theirs = time(&gemm_graph)?;
        eprintln!(
            "m={m} n={n} k={k}: ours {ours:.1} us, cast {cast:.1} us + cuBLAS {theirs:.1} us = {:.1} us ({:.2}x)",
            cast + theirs,
            ours / (cast + theirs)
        );
    }
    Ok(())
}
