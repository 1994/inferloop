//! The cuTile sources of the element-wise state cells, through the contract, against the shared
//! cases.
#[path = "../../../cuda-kernels/acceptance/elementwise.rs"]
pub mod acceptance;

use crate::device::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use crate::resident::capture::error;
use acceptance::{Case, Op};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{BinaryOp, UnaryOp};
use std::time::Instant;

/// Replays timed per case after the checked one.
const TIMED_REPLAYS: u32 = 200;
/// A hidden row, and a length that leaves a partial tile.
const LENGTHS: [usize; 2] = [5120, 1000];

fn check(device: &CudaDevice, case: Case) -> Result<()> {
    let shape = [case.elements];
    let first = device.upload(case.first(), &shape)?;
    let second = device.upload(case.second(), &shape)?;
    let output = device.upload(vec![f32::NAN; case.elements], &shape)?;
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        let (out, x, y) = (
            buffer(output.as_ref())?,
            buffer(first.as_ref())?,
            buffer(second.as_ref())?,
        );
        let state = &kernels.state;
        match case.op {
            Op::Silu => state.unary(scope, UnaryOp::Silu, out, x),
            Op::Sigmoid => state.unary(scope, UnaryOp::Sigmoid, out, x),
            Op::Add => state.binary(scope, BinaryOp::Add, out, x, y),
            Op::Multiply => state.binary(scope, BinaryOp::Multiply, out, x, y),
        }
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
        "ELEMENTWISE_CELL {case:?} {:.2} us/replay",
        start.elapsed().as_secs_f64() * 1e6 / f64::from(TIMED_REPLAYS)
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn tile_sources_pass_the_shared_cases() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for op in [Op::Silu, Op::Sigmoid, Op::Add, Op::Multiply] {
        for elements in LENGTHS {
            check(&device, Case { op, elements })?;
        }
    }
    Ok(())
}
