//! The cuTile source of the RMS norm cell, through the contract, against the shared cases.
#[path = "../../../cuda-kernels/acceptance/norm.rs"]
pub mod acceptance;

use super::*;
use crate::device::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use crate::resident::capture::error;
use acceptance::Case;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{NormBuffers, NormPlan};
use std::time::Instant;

/// Replays timed per case after the checked one.
const TIMED_REPLAYS: u32 = 200;
const EPSILON: f32 = 1.0e-6;

fn cases() -> Vec<Case> {
    let case = |rows, dimension, gated, offset| Case {
        rows,
        dimension,
        gated,
        epsilon: EPSILON,
        offset,
    };
    vec![
        // Row tiles: a narrow power-of-two head over whole tiles of rows.
        case(
            NORM_ROWS * NORM_ROWS,
            NORM_ROWS_MAX_DIMENSION / 2,
            true,
            0.0,
        ),
        case(NORM_ROWS * NORM_ROWS, NORM_ROWS_MAX_DIMENSION, false, 1.0),
        // Blocks: a wide hidden row, and a row the block pads past its width.
        case(4, 5120, false, 1.0),
        case(3, 1000, true, 0.0),
    ]
}

fn check(device: &CudaDevice, case: Case) -> Result<()> {
    let shape = [case.rows, case.dimension];
    let input = device.upload(case.input(), &shape)?;
    let gate = device.upload(case.gate(), &shape)?;
    let weight = device.upload(case.weight(), &[case.dimension])?;
    let output = device.upload(vec![f32::NAN; case.rows * case.dimension], &shape)?;
    let plan = NormPlan {
        layout: layout(case.dimension, case.rows),
        rows: case.rows,
        dimension: case.dimension,
        gated: case.gated,
        epsilon: case.epsilon,
        offset: case.offset,
        occupancy: None,
    };
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels
            .norm
            .norm(
                scope,
                &plan,
                NormBuffers {
                    out: buffer(output.as_ref())?,
                    x: buffer(input.as_ref())?,
                    weight: buffer(weight.as_ref())?,
                    gate: buffer(gate.as_ref())?,
                },
            )
            .map_err(error)
    })?;
    let replay = || {
        graph
            .launch()
            .sync_on(device.stream())
            .map_err(device_error)
    };
    replay()?;
    case.check(&device.read_borrowed(&output)?)
        .map_err(Error::invariant)?;
    let start = Instant::now();
    for _ in 0..TIMED_REPLAYS {
        replay()?;
    }
    eprintln!(
        "NORM_CELL {case:?} {:?} {:.2} us/replay",
        plan.layout,
        start.elapsed().as_secs_f64() * 1e6 / f64::from(TIMED_REPLAYS)
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn tile_source_passes_the_shared_cases() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    cases()
        .into_iter()
        .try_for_each(|case| check(&device, case))
}
