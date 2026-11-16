//! `Q6_K` GEMV prototype: correctness against the reference decode and 5090 bandwidth.
use infer_backend_cuda::device::CudaDevice;
use infer_cuda_kernels::contract::{GroupedIntPlan, WeightRef};
use infer_format::{GgufType, dequantize};
use infer_models::Q6KTiled;
use std::error::Error;
use std::mem::size_of_val;
use std::time::Instant;

/// Output rows and reduction width.
const ROWS: usize = 4096;
const COLUMNS: usize = 5120;
/// Replays used for the bandwidth sample.
const REPLAYS: usize = 100;
/// RTX 5090 peak memory bandwidth in GB/s.
const HBM_GBPS: f64 = 1792.0;

fn blocks() -> Vec<u8> {
    let per_row = COLUMNS / 256;
    let mut blocks = Vec::with_capacity(ROWS * per_row * 210);
    for row in 0..ROWS {
        for block in 0..per_row {
            let mut bytes = vec![0u8; 210];
            bytes[208..210].copy_from_slice(&[0x00, 0x38]); // 0.5
            for scale in &mut bytes[192..208] {
                *scale = u8::try_from(row % 256)
                    .unwrap_or(0)
                    .wrapping_add(u8::try_from(block % 256).unwrap_or(0))
                    % 15
                    + 1;
            }
            for (index, quant) in bytes[..128].iter_mut().enumerate() {
                let row = u8::try_from(row % 256).unwrap_or(0);
                let offset = u8::try_from((block * 128 + index) % 256).unwrap_or(0);
                *quant = row.wrapping_mul(7).wrapping_add(offset) | 0x21;
            }
            for (index, quant) in bytes[128..192].iter_mut().enumerate() {
                let row = u8::try_from(row % 256).unwrap_or(0);
                let offset = u8::try_from((block * 64 + index) % 256).unwrap_or(0);
                *quant = row.wrapping_mul(3).wrapping_add(offset) | 0x49;
            }
            blocks.extend_from_slice(&bytes);
        }
    }
    blocks
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let blocks = blocks();
    let tiled = Q6KTiled::from_blocks(ROWS, COLUMNS, &blocks)?;
    let x: Vec<f32> = (0..COLUMNS)
        .map(|index| {
            f32::from(u16::try_from(index % u16::MAX as usize).unwrap_or(0))
                .mul_add(0.011, 0.0)
                .sin()
        })
        .collect();
    let mut values = vec![0.0f32; ROWS * COLUMNS];
    dequantize(GgufType::Q6K, &blocks, &mut values)?;
    let expected: Vec<f32> = (0..ROWS)
        .map(|row| {
            values[row * COLUMNS..][..COLUMNS]
                .iter()
                .zip(&x)
                .map(|(value, activation)| value * activation)
                .sum()
        })
        .collect();

    let x_alloc = runtime.upload(x)?;
    let codes_alloc = runtime.upload(tiled.codes().to_vec())?;
    let factors_alloc = runtime.upload(tiled.factors().to_vec())?;
    let out_alloc = runtime.zeros::<f32>(ROWS)?;
    let plan = GroupedIntPlan {
        tile_rows: 16,
        columns: COLUMNS,
    };
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels.projection.grouped_int6_row(
            scope,
            &plan,
            out_alloc.buffer(),
            x_alloc.buffer(),
            WeightRef::GroupedInt6 {
                codes: codes_alloc.buffer(),
                factors: factors_alloc.buffer(),
            },
        )
    })?;
    device.tile().sync(graph.launch())?;
    let actual = runtime.read(&out_alloc)?;
    let mut max_relative = 0.0f64;
    for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let relative = f64::from((actual - expected).abs()) / f64::from(expected.abs().max(1.0));
        max_relative = max_relative.max(relative);
        if relative > 1e-3 {
            return Err(format!("row {row}: actual {actual} expected {expected}").into());
        }
    }
    println!(
        "Q6_K GEMV correctness passed over {ROWS}x{COLUMNS} (max relative {max_relative:.3e})"
    );

    let started = Instant::now();
    for _ in 0..REPLAYS {
        device.tile().sync(graph.launch())?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let bytes = tiled.codes().len() + size_of_val(tiled.factors());
    let bytes = f64::from(u32::try_from(bytes).unwrap_or(u32::MAX));
    let replays = f64::from(u32::try_from(REPLAYS).unwrap_or(u32::MAX));
    let bandwidth = bytes * replays / elapsed / 1e9;
    println!(
        "Q6_K GEMV bandwidth {bandwidth:.1} GB/s ({:.1}% of {HBM_GBPS:.0} GB/s roofline)",
        bandwidth / HBM_GBPS * 100.0
    );
    Ok(())
}
