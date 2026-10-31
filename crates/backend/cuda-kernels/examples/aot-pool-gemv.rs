//! Capture an AOT projection into a graph, replay it, and compare with a CPU reference.
#![expect(
    unsafe_code,
    reason = "Launches through the raw-address kernel contract with buffers owned by this function"
)]

#[path = "../acceptance/pool_gemv.rs"]
mod acceptance;

use acceptance::{Case, OUTPUTS, ROWS};
use infer_cuda_kernels::aot::{AotProjection, Device};
use infer_cuda_kernels::contract::{DevicePtr, PoolGemvPlan, Projection};
use std::env;
use std::error::Error;
use std::mem::size_of;
use std::path::Path;

/// Reduction lengths around the warp width, including partial warps.
const COLUMNS: [usize; 5] = [1, 31, 32, 64, 129];
/// Replays of one captured graph; every replay must reproduce the same bits.
const REPLAYS: usize = 3;

fn check(
    device: &Device,
    projection: &AotProjection,
    columns: usize,
) -> Result<(), Box<dyn Error>> {
    let case = Case { columns };
    let x = case.input();
    let bits: Vec<u8> = case
        .weight_bits()
        .iter()
        .flat_map(|value| value.to_ne_bytes())
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
    case.check(&actual)?;
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
