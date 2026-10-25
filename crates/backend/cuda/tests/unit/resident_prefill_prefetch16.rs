//! Long-K down GEMM: Prefetch8 vs Prefetch16, exact bits and paired timing.

use super::kernels;
use crate::device::CudaDevice;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use std::error::Error as StdError;
use std::time::Duration;

#[test]
#[ignore = "requires CUDA hardware; run inside safe-run with real weights"]
fn prefetch16_long_k_down_preserves_bits_and_measures_paired_time() -> Result<(), Box<dyn StdError>>
{
    if cfg!(debug_assertions) {
        return Err("requires cargo test --release".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (rows, outputs, columns) = (1024usize, 5120usize, 17408usize);
    let input = device.upload(
        vec![f4e2m1fnx2::from_bits(0x21); rows * columns / 2],
        &[rows, columns / 2],
    )?;
    let input_scales = device.upload(
        vec![f8e4m3fn(56); rows * columns / 16],
        &[rows, columns / 16],
    )?;
    let weight = device.upload(
        vec![f4e2m1fnx2::from_bits(0x31); outputs * columns / 2],
        &[outputs, columns / 2],
    )?;
    let scales = device.upload(
        vec![f8e4m3fn(56); outputs * columns / 16],
        &[outputs, columns / 16],
    )?;
    let mut shallow = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let mut deep = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let tile = [128usize, 128usize];
    let generics = vec![columns.to_string(), rows.to_string(), outputs.to_string()];
    let shallow_graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::packed_prefill(
                (&mut shallow).partition(tile),
                &input,
                &input_scales,
                &weight,
                &scales,
                1.0,
            )
            .generics(generics.clone()),
        )?;
        Ok(())
    })?;
    let deep_graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::packed_prefill16(
                (&mut deep).partition(tile),
                &input,
                &input_scales,
                &weight,
                &scales,
                1.0,
            )
            .generics(generics),
        )?;
        Ok(())
    })?;
    shallow_graph.launch().sync_on(&device.stream)?;
    deep_graph.launch().sync_on(&device.stream)?;
    let shallow_bits = shallow.to_host_vec().sync_on(&device.stream)?;
    let deep_bits = deep.to_host_vec().sync_on(&device.stream)?;
    if shallow_bits != deep_bits {
        return Err("Prefetch16 changed output bits".into());
    }
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(200),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    let (baseline, candidate) = do_bench_paired(
        &device.stream,
        &options,
        |_| {
            shallow_graph
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
        |_| {
            deep_graph
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
    )?;
    println!(
        "{}",
        serde_json::json!({
            "prefill_down_prefetch16": true,
            "shape": [rows, outputs, columns],
            "bits_equal": true,
            "prefetch8_ms": baseline.median_ms(),
            "prefetch16_ms": candidate.median_ms(),
            "speedup": baseline.median_ms() / candidate.median_ms(),
        })
    );
    Ok(())
}
