//! The device contract keeps recorded memory alive, and replay costs what raw cuTile replay costs.
use crate::device::{CudaDevice, asynchronous::block_on, device_error};
use crate::resident::pool_gemv::{
    cell_tests::acceptance::{Case, OUTPUTS, ROWS},
    kernels as pool_gemv,
};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::PoolGemvPlan;
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
use std::sync::Arc;
use std::time::Instant;

const COLUMNS: usize = 64;
const TILE: (usize, usize) = (16, 64);
/// Launches per timing block and alternating blocks per side.
const LAUNCHES: u32 = 2000;
const BLOCKS: usize = 11;
/// The wrapper may not cost more than this, against replay-to-replay noise.
const MAX_RATIO: f64 = 1.03;

fn upload<T: DType>(device: &CudaDevice, values: Vec<T>, shape: &[usize]) -> Result<Allocation<T>> {
    Allocation::adopt(device.upload(values, shape)?)
}

struct Fixture {
    graph: Graph,
    output: Allocation<f32>,
}

fn capture(device: &CudaDevice, case: Case) -> Result<Fixture> {
    let columns = case.columns;
    let input = upload(device, case.input(), &[ROWS, columns])?;
    let bits = case
        .weight_bits()
        .into_iter()
        .map(bf16::from_bits)
        .collect();
    let weight = upload(device, bits, &[OUTPUTS, columns])?;
    let output = upload(device, vec![f32::NAN; ROWS * OUTPUTS], &[ROWS, OUTPUTS])?;
    let plan = PoolGemvPlan {
        rows: ROWS,
        outputs: OUTPUTS,
        columns,
        tile_rows: TILE.0,
        tile_columns: TILE.1,
    };
    let kernels = device.kernels();
    let graph = device.runtime().capture(|recording| {
        kernels.projection.pool_gemv_bf16(
            recording,
            &plan,
            output.buffer(),
            input.buffer(),
            weight.bf16_buffer(),
        )
    })?;
    Ok(Fixture { graph, output })
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_graph_keeps_recorded_memory_alive() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let case = Case { columns: COLUMNS };
    // `capture` dropped its input and weight handles; only the graph holds them now.
    let fixture = capture(&device, case)?;
    let _churn = device.upload(vec![0.0f32; 1 << 20], &[1 << 20])?;
    block_on(device.runtime().run(fixture.graph.launch()))?;
    case.check(&device.read(fixture.output.tensor())?)
        .map_err(Error::invariant)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "requires a CUDA device; timing"]
fn wrapped_replay_costs_what_raw_replay_costs() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let case = Case { columns: COLUMNS };
    let wrapped = capture(&device, case)?;
    let input = device.upload(case.input(), &[ROWS, COLUMNS])?;
    let bits = case
        .weight_bits()
        .into_iter()
        .map(bf16::from_bits)
        .collect();
    let weight = device.upload(bits, &[OUTPUTS, COLUMNS])?;
    let mut output = device.upload(vec![0.0f32; ROWS * OUTPUTS], &[ROWS, OUTPUTS])?;
    let output = Arc::get_mut(&mut output).ok_or_else(|| Error::invariant("shared output"))?;
    let raw = CudaGraph::scope(&device.stream, |scope| {
        scope
            .record(
                pool_gemv::dense(
                    output.partition([1, TILE.0]),
                    input.as_ref(),
                    weight.as_ref(),
                )
                .generics(vec![
                    bf16::DTYPE.as_str().into(),
                    TILE.0.to_string(),
                    TILE.1.to_string(),
                    COLUMNS.to_string(),
                ]),
            )
            .map(drop)
    })
    .map_err(device_error)?;
    let runtime = device.runtime();
    let (mut raw_times, mut wrapped_times) = (Vec::new(), Vec::new());
    block_on(async {
        for _ in 0..BLOCKS {
            let start = Instant::now();
            for _ in 0..LAUNCHES {
                device.run_op(raw.launch()).await?;
            }
            raw_times.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            for _ in 0..LAUNCHES {
                runtime.run(wrapped.graph.launch()).await?;
            }
            wrapped_times.push(start.elapsed().as_secs_f64());
        }
        Ok::<(), Error>(())
    })?;
    let per_launch = 1e6 / f64::from(LAUNCHES);
    let (raw, wrapped) = (median(raw_times), median(wrapped_times));
    println!(
        "raw {:.2} us, wrapped {:.2} us per launch, ratio {:.4}",
        raw * per_launch,
        wrapped * per_launch,
        wrapped / raw
    );
    assert!(wrapped / raw < MAX_RATIO, "wrapped replay costs more");
    Ok(())
}
