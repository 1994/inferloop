//! The cuTile source of the rotary cell, through the contract, against the shared cases.
#[path = "../../../cuda-kernels/acceptance/rope.rs"]
pub mod acceptance;

use crate::device::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use crate::resident::capture::error;
use acceptance::{Case, Positions};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{RopeBuffers, RopeMode, RopePlan};
use std::time::Instant;

/// Replays timed per case after the checked one.
const TIMED_REPLAYS: u32 = 200;
/// Metadata fields per pooled lane, as the resident metadata lays them out.
const FIELDS: usize = 4;

fn cases() -> Vec<Case> {
    let case = |positions, rows, heads, half, base| Case {
        positions,
        rows,
        heads,
        head_dim: 256,
        half,
        base,
    };
    vec![
        case(Positions::Single, 24, 24, 32, 4093),
        case(Positions::Prompt, 64 * 4, 4, 64, 17),
        case(Positions::Pooled { fields: FIELDS }, 3 * 8, 8, 32, 8191),
    ]
}

fn check(device: &CudaDevice, case: Case) -> Result<()> {
    let shape = [case.rows, case.head_dim];
    let input = device.upload(case.input(), &shape)?;
    let frequencies = device.upload(case.frequencies(), &[case.half])?;
    let metadata = case.metadata();
    let metadata = device.upload(metadata.clone(), &[metadata.len()])?;
    let output = device.upload(vec![f32::NAN; case.rows * case.head_dim], &shape)?;
    let plan = RopePlan {
        mode: match case.positions {
            Positions::Single => RopeMode::Single,
            Positions::Prompt => RopeMode::Prompt,
            Positions::Pooled { fields } => RopeMode::Pooled { fields },
        },
        rows: case.rows,
        heads: case.heads,
        head_dim: case.head_dim,
        half: case.half,
    };
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels
            .rope
            .rope(
                scope,
                &plan,
                RopeBuffers {
                    out: buffer(output.as_ref())?,
                    x: buffer(input.as_ref())?,
                    frequencies: buffer(frequencies.as_ref())?,
                    axes: None,
                    metadata: buffer(metadata.as_ref())?,
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
        "ROPE_CELL {case:?} {:.2} us/replay",
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
