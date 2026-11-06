use super::kernels;
use crate::device::CudaDevice;
use crate::device::device_error;
use crate::resident::buffers::buffer as operand;
use crate::resident::buffers::cutile;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_cuda_kernels::tile::kernels as contract_kernels;
use infer_cuda_kernels::tile::runtime::Graph;
use std::error::Error;
use std::slice::from_ref;
use std::time::Duration;

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn recurrent_chunk_performance_gate() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("benchmark requires release".into());
    }
    let compile = CompileOptions::default();
    if compile.device_debug || compile.opt_level.is_some_and(|x| x != 3) {
        return Err("benchmark requires CUDA O3".into());
    }
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    // 64 lanes is the width the 27B prompt graph captures, so the serving number has a
    // same-width isolated counterpart to compare against instead of an interpolation.
    for (kh, vh, dim, lanes) in [(2, 4, 32, 32), (16, 48, 128, 64), (16, 48, 128, 256)] {
        bench_case(&device, (kh, vh, dim, lanes))?;
    }
    Ok(())
}
fn production_block(dim: usize, lanes: usize) -> usize {
    if dim == super::ORDERED_DIM && super::ORDERED_WIDTHS.contains(&lanes) {
        super::ORDERED_BLOCK
    } else {
        dim
    }
}

fn bench_case(
    device: &CudaDevice,
    geometry: (usize, usize, usize, usize),
) -> Result<(), Box<dyn Error>> {
    let (kh, vh, dim, lanes) = geometry;
    let cols = (2 * kh + vh) * dim;
    let qkv = device.upload(
        super::tests::values(lanes * cols, 17),
        &[lanes, 2 * kh + vh, dim],
    )?;
    let beta = device.upload(vec![0.2_f32; lanes * vh], &[lanes, vh])?;
    let alpha = device.upload(vec![-0.1_f32; lanes * vh], &[lanes, vh])?;
    let a_log = device.upload(vec![-1.0_f32; vh], &[vh])?;
    let bias = device.upload(vec![0.1_f32; vh], &[vh])?;
    let mut state = api::zeros::<f32>(&[vh, dim, dim]).sync_on(&device.stream)?;
    let mut old_outputs = Vec::new();
    let mut metadata = Vec::new();
    for lane in 0..lanes {
        old_outputs.push(
            api::zeros::<f32>(&[vh, dim])
                .sync_on(&device.stream)?
                .reshape(&[vh, 1, dim])?,
        );
        metadata.push(device.upload(vec![i32::try_from(lane)?, 0, i32::try_from(lane)?, 0], &[4])?);
    }
    let baseline = per_lane_baseline(
        device,
        geometry,
        [&qkv, &beta, &alpha, &a_log, &bias],
        &mut state,
        &mut old_outputs,
        &metadata,
    )?;
    let metadata = device
        .runtime()
        .upload(vec![0, i32::try_from(lanes)?, 0, 0])?;
    let mut scratch = api::zeros::<f32>(&[vh, lanes, dim]).sync_on(&device.stream)?;
    let mut output = api::zeros::<f32>(&[lanes, vh, dim]).sync_on(&device.stream)?;
    let block = production_block(dim, lanes);
    let generics = [kh, vh, dim, lanes, block]
        .map(|value| value.to_string())
        .to_vec();
    let prepared = (dim == super::ORDERED_DIM)
        .then(|| super::prepared::Inputs::new(device, (kh, vh, dim), lanes))
        .transpose()?;
    let candidate = Graph::scope(&device.stream, |scope| {
        if dim == 128 {
            prepared.as_ref().expect("ordered preparation").record(
                scope,
                contract_kernels(0).recurrent.as_ref(),
                [
                    operand(&qkv)?,
                    operand(&beta)?,
                    operand(&alpha)?,
                    operand(&a_log)?,
                    operand(&bias)?,
                ],
                (&state, &scratch),
                &metadata,
                (kh, vh, dim, lanes, block),
            )?;
        } else {
            cutile(scope)?.record(
                kernels::delta(
                    (&mut state).partition([1, dim, dim]),
                    (&mut scratch).partition([1, lanes, dim]),
                    &qkv,
                    &beta,
                    &alpha,
                    &a_log,
                    &bias,
                    metadata.tensor(),
                )
                .generics(generics),
            )?;
        }
        cutile(scope)?.record(
            kernels::transpose((&mut output).partition([1, 1, dim]), &scratch)
                .generics(vec![dim.to_string()]),
        )?;
        Ok(())
    })?;
    compare_graphs(device, &baseline, &candidate, geometry)
}

/// The per-lane reference recurrence the batched kernel must match.
fn per_lane_baseline(
    device: &CudaDevice,
    (kh, vh, dim, lanes): (usize, usize, usize, usize),
    [qkv, beta, alpha, a_log, bias]: [&Arc<Tensor<f32>>; 5],
    state: &mut Tensor<f32>,
    old_outputs: &mut [Tensor<f32>],
    metadata: &[Arc<Tensor<i32>>],
) -> infer_core::Result<Graph> {
    let cols = (2 * kh + vh) * dim;
    Graph::scope(&device.stream, |scope| {
        for (lane, output) in old_outputs.iter_mut().enumerate() {
            cutile(scope)?.record(
                super::super::recurrent::recurrent::delta(
                    output.partition([1, 1, dim]),
                    state.partition([1, dim, dim]),
                    &qkv.view(&[lanes * cols])
                        .map_err(device_error)?
                        .slice(from_ref(&(lane * cols..(lane + 1) * cols)))
                        .map_err(device_error)?,
                    &beta
                        .view(&[lanes * vh])
                        .map_err(device_error)?
                        .slice(from_ref(&(lane * vh..(lane + 1) * vh)))
                        .map_err(device_error)?,
                    &alpha
                        .view(&[lanes * vh])
                        .map_err(device_error)?
                        .slice(from_ref(&(lane * vh..(lane + 1) * vh)))
                        .map_err(device_error)?,
                    a_log,
                    bias,
                    &metadata[lane],
                )
                .generics(vec![
                    kh.to_string(),
                    vh.to_string(),
                    dim.to_string(),
                    dim.to_string(),
                ]),
            )?;
        }
        Ok(())
    })
}

fn compare_graphs(
    device: &CudaDevice,
    baseline: &Graph,
    candidate: &Graph,
    geometry: (usize, usize, usize, usize),
) -> Result<(), Box<dyn Error>> {
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(100),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    let (old, new) = do_bench_paired(
        &device.stream,
        &options,
        |_| {
            baseline
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
        |_| {
            candidate
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
    )?;
    println!(
        "geometry={geometry:?} old_ms={} new_ms={} speedup={}",
        old.median_ms(),
        new.median_ms(),
        old.median_ms() / new.median_ms()
    );
    assert!(
        new.median_ms() <= old.median_ms() * 1.05,
        "recurrent chunk regresses beyond 5% noise allowance"
    );
    Ok(())
}
