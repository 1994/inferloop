//! The cuTile sources of the recurrent step cells, through the contract, against the shared
//! cases.
#[path = "../../../cuda-kernels/acceptance/recurrent.rs"]
pub mod acceptance;

use crate::device::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use crate::resident::capture::error;
use acceptance::{CONV_WIDTH, Conv, Delta};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{ConvBuffers, ConvKernel, DeltaBuffers, DeltaStepPlan};
use infer_cuda_kernels::runtime::Graph;
use std::time::Instant;

/// Replays timed per case after the checked one.
const TIMED_REPLAYS: u32 = 200;

fn replay(device: &CudaDevice, graph: &Graph) -> Result<()> {
    graph
        .launch()
        .sync_on(device.stream())
        .map_err(device_error)
}

fn time(device: &CudaDevice, graph: &Graph, label: &str) -> Result<()> {
    let start = Instant::now();
    for _ in 0..TIMED_REPLAYS {
        replay(device, graph)?;
    }
    eprintln!(
        "RECURRENT_CELL {label} {:.2} us/replay",
        start.elapsed().as_secs_f64() * 1e6 / f64::from(TIMED_REPLAYS)
    );
    Ok(())
}

fn check_conv(device: &CudaDevice, case: Conv) -> Result<()> {
    let channels = case.channels;
    let input = device.upload(case.input(), &[channels])?;
    let [h0, h1, h2] = case.history().map(|rows| device.upload(rows, &[channels]));
    let (h0, h1, h2) = (h0?, h1?, h2?);
    let weight = device.upload(case.weight(), &[channels, CONV_WIDTH])?;
    let metadata = device.upload(case.metadata(), &[acceptance::METADATA_FIELDS])?;
    let output = device.upload(vec![f32::NAN; channels], &[channels])?;
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels
            .recurrent
            .conv(
                scope,
                ConvKernel::Step,
                channels,
                ConvBuffers {
                    output: buffer(output.as_ref())?,
                    history: [
                        buffer(h0.as_ref())?,
                        buffer(h1.as_ref())?,
                        buffer(h2.as_ref())?,
                    ],
                    input: buffer(input.as_ref())?,
                    weight: buffer(weight.as_ref())?,
                    metadata: buffer(metadata.as_ref())?,
                },
            )
            .map_err(error)
    })?;
    replay(device, &graph)?;
    let history = [h0, h1, h2].map(|rows| device.read_borrowed(&rows));
    let [h0, h1, h2] = history;
    case.check(
        &device.read_borrowed(&output)?,
        [h0?.as_slice(), h1?.as_slice(), h2?.as_slice()],
    )
    .map_err(Error::invariant)?;
    time(device, &graph, &format!("{case:?}"))
}

fn check_delta(device: &CudaDevice, case: Delta) -> Result<()> {
    let (heads, dim) = (case.value_heads, case.dim);
    let qkv = case.qkv();
    let qkv = device.upload(qkv.clone(), &[qkv.len()])?;
    let state = device.upload(case.state(), &[heads, dim, dim])?;
    let [beta, alpha, decay, bias] = case.gates().map(|gate| device.upload(gate, &[heads]));
    let (beta, alpha, decay, bias) = (beta?, alpha?, decay?, bias?);
    let metadata = device.upload(case.metadata(), &[acceptance::METADATA_FIELDS])?;
    let output = device.upload(vec![f32::NAN; heads * dim], &[heads * dim])?;
    let plan = DeltaStepPlan {
        key_heads: case.key_heads,
        value_heads: heads,
        dim,
        block: dim,
        lanes: None,
    };
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels
            .recurrent
            .delta_step(
                scope,
                &plan,
                DeltaBuffers {
                    state: buffer(state.as_ref())?,
                    output: buffer(output.as_ref())?,
                    qkv: buffer(qkv.as_ref())?,
                    beta: buffer(beta.as_ref())?,
                    alpha: buffer(alpha.as_ref())?,
                    decay: buffer(decay.as_ref())?,
                    bias: buffer(bias.as_ref())?,
                    metadata: buffer(metadata.as_ref())?,
                },
            )
            .map_err(error)
    })?;
    replay(device, &graph)?;
    case.check(
        &device.read_borrowed(&output)?,
        &device.read_borrowed(&state)?,
    )
    .map_err(Error::invariant)?;
    time(device, &graph, &format!("{case:?}"))
}

#[test]
#[ignore = "requires a CUDA device"]
fn tile_sources_pass_the_shared_cases() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for active in [true, false] {
        check_conv(
            &device,
            Conv {
                channels: 1024,
                active,
            },
        )?;
        check_delta(
            &device,
            Delta {
                key_heads: 2,
                value_heads: 4,
                dim: 128,
                active,
            },
        )?;
    }
    Ok(())
}
