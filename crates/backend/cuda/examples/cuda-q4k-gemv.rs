//! `Q4_K` GEMV prototype: correctness against the host oracle and 5090 bandwidth.

use infer_backend_cuda::device::CudaDevice;
use infer_cuda_kernels::contract::{DownPlan, Q4KWeight};
use infer_gguf::{Q4KDevice, Q4KTiled};
use std::error::Error;
use std::mem::size_of;
use std::time::Instant;

/// Output rows and reduction width of the synthetic matrix.
const ROWS: usize = 4096;
const COLUMNS: usize = 5120;
/// Replays used for the bandwidth sample.
const REPLAYS: usize = 100;
/// RTX 5090 peak memory bandwidth in GB/s, for the roofline percentage.
const HBM_GBPS: f64 = 1792.0;

const fn half(value: f32) -> [u8; 2] {
    // Only the two exact binary16 values this fixture uses.
    match value.to_bits() {
        0x3f00_0000 => [0x00, 0x38],
        _ => [0x00, 0x34],
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "synthetic fixture bytes wrap intentionally"
)]
fn blocks() -> Vec<u8> {
    let mut blocks = Vec::new();
    for row in 0..ROWS {
        for block in 0..(COLUMNS / 256) {
            let mut bytes = vec![0u8; 144];
            bytes[..2].copy_from_slice(&half(0.5));
            bytes[2..4].copy_from_slice(&half(0.25));
            for index in 0..12 {
                bytes[4 + index] = (row as u8)
                    .wrapping_mul(3)
                    .wrapping_add(block as u8)
                    .wrapping_add(index as u8)
                    | 0x11;
            }
            for (index, quant) in bytes[16..].iter_mut().enumerate() {
                *quant = (row as u8)
                    .wrapping_mul(17)
                    .wrapping_add((block * 32 + index) as u8)
                    | 0x21;
            }
            blocks.extend_from_slice(&bytes);
        }
    }
    blocks
}

#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark fixture and bandwidth reporting"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let blocks = blocks();
    let matrix = Q4KDevice::from_blocks(ROWS, COLUMNS, &blocks)?;
    let tiled = Q4KTiled::from_device(&matrix)?;

    let x: Vec<f32> = (0..COLUMNS)
        .map(|index| ((index as f32) * 0.013).sin())
        .collect();
    let expected: Vec<f32> = (0..ROWS)
        .map(|row| matrix.row_dot(row, &x))
        .collect::<Result<_, _>>()?;

    let x_alloc = runtime.upload(x)?;
    let codes_alloc = runtime.upload(tiled.codes().to_vec())?;
    let factors_alloc = runtime.upload(tiled.factors().to_vec())?;
    let biases_alloc = runtime.upload(tiled.biases().to_vec())?;
    let out_alloc = runtime.zeros::<f32>(ROWS)?;

    let plan = DownPlan {
        tile_rows: 16,
        tile_columns: 64,
        columns: COLUMNS,
    };
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels.projection.q4_k_row(
            scope,
            &plan,
            out_alloc.buffer(),
            x_alloc.buffer(),
            Q4KWeight {
                codes: codes_alloc.buffer(),
                factors: factors_alloc.buffer(),
                biases: biases_alloc.buffer(),
            },
        )
    })?;
    device.tile().sync(graph.launch())?;
    let actual = runtime.read(&out_alloc)?;
    let mut max_relative = 0.0f64;
    for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let error = (actual - expected).abs();
        let relative = f64::from(error) / f64::from(expected.abs().max(1.0));
        max_relative = max_relative.max(relative);
        if relative > 1e-3 {
            return Err(format!(
                "row {row}: actual {actual} expected {expected} relative {relative:.3e}"
            )
            .into());
        }
    }
    println!(
        "Q4_K GEMV correctness passed over {ROWS}x{COLUMNS} (max relative {max_relative:.3e})"
    );

    let started = Instant::now();
    for _ in 0..REPLAYS {
        device.tile().sync(graph.launch())?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let bytes = (tiled.codes().len()
        + (tiled.factors().len() + tiled.biases().len()) * size_of::<f32>()) as f64;
    let bandwidth = bytes * REPLAYS as f64 / elapsed / 1e9;
    println!(
        "Q4_K GEMV bandwidth {bandwidth:.1} GB/s ({:.1}% of {HBM_GBPS:.0} GB/s roofline)",
        bandwidth / HBM_GBPS * 100.0
    );

    Ok(())
}
