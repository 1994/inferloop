//! Capture an AOT projection into a graph, replay it, and compare with a CPU reference.
#![expect(
    unsafe_code,
    reason = "Launches through the raw-address kernel contract with buffers owned by this function"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "The fixture values are small integers that fit exactly in f32"
)]

use infer_cuda_kernels::aot::{AotProjection, Device};
use infer_cuda_kernels::contract::{DevicePtr, PoolGemvPlan, Projection};
use std::env;
use std::error::Error;
use std::mem::size_of;
use std::path::Path;

/// Activation rows of every fixture.
const ROWS: usize = 4;
/// Output elements of every fixture.
const OUTPUTS: usize = 8;
/// Reduction lengths around the warp width, including partial warps.
const COLUMNS: [usize; 5] = [1, 31, 32, 64, 129];
/// Relative tolerance: a few F32 rounding units on the summed magnitude of the element.
const RELATIVE_TOLERANCE: f32 = 1.0e-5;
/// Replays of one captured graph; every replay must reproduce the same bits.
const REPLAYS: usize = 3;
/// Bits a BF16 keeps from the top of an F32.
const BF16_SHIFT: u32 = 16;

fn check(
    device: &Device,
    projection: &AotProjection,
    columns: usize,
) -> Result<(), Box<dyn Error>> {
    let x: Vec<f32> = (0..ROWS * columns)
        .map(|i| ((i * 7) % 13) as f32 / 8.0 - 0.5)
        .collect();
    // Eighths are exact in BF16, so truncating the F32 bits loses nothing.
    let w: Vec<f32> = (0..OUTPUTS * columns)
        .map(|i| ((i * 11) % 17) as f32 / 8.0 - 1.0)
        .collect();
    let bits: Vec<u8> = w
        .iter()
        .flat_map(|value| u16::try_from(value.to_bits() >> BF16_SHIFT).map(u16::to_ne_bytes))
        .flatten()
        .collect();
    let x_buffer = device.alloc(ROWS * columns * size_of::<f32>())?;
    let w_buffer = device.alloc(bits.len())?;
    let out_buffer = device.alloc(ROWS * OUTPUTS * size_of::<f32>())?;
    device.upload_f32(&x_buffer, &x)?;
    device.upload(&w_buffer, &bits)?;
    let plan = PoolGemvPlan {
        rows: ROWS,
        outputs: OUTPUTS,
        columns,
        tile_rows: 1,
        tile_columns: columns,
    };
    let graph = device.capture(|recording| {
        // SAFETY: the three buffers match the plan, do not alias, and are declared before `graph`,
        // so they outlive it and every replay.
        unsafe {
            projection.pool_gemv_bf16(
                recording,
                &plan,
                DevicePtr(out_buffer.pointer()),
                DevicePtr(x_buffer.pointer()),
                DevicePtr(w_buffer.pointer()),
            )
        }
    })?;
    let mut first: Option<Vec<u32>> = None;
    let mut actual = vec![0.0f32; ROWS * OUTPUTS];
    for _ in 0..REPLAYS {
        device.upload_f32(&out_buffer, &[f32::NAN; ROWS * OUTPUTS])?;
        graph.launch()?;
        device.download_f32(&out_buffer, &mut actual)?;
        let replay: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
        if first.get_or_insert_with(|| replay.clone()) != &replay {
            return Err(format!("{columns} columns: replays differ").into());
        }
    }
    for (index, value) in actual.iter().enumerate() {
        let (row, element) = (index / OUTPUTS, index % OUTPUTS);
        let terms = (0..columns).map(|k| w[element * columns + k] * x[row * columns + k]);
        let expected: f32 = terms.clone().sum();
        let magnitude: f32 = terms.map(f32::abs).sum();
        if (value - expected).abs() > magnitude * RELATIVE_TOLERANCE {
            return Err(
                format!("{columns} columns, [{row}, {element}]: {value} vs {expected}").into(),
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = env::args()
        .nth(1)
        .unwrap_or_else(|| "target/kernels/aot/sm_120".to_owned());
    let device = Device::open(0)?;
    let projection = AotProjection::load(&device, Path::new(&root))?;
    for columns in COLUMNS {
        check(&device, &projection, columns)?;
    }
    println!("AOT graph replay passed: {ROWS}x{OUTPUTS} over {COLUMNS:?} columns");
    Ok(())
}
