//! Load an AOT projection artifact and compare it with a CPU reference.
#![expect(
    unsafe_code,
    reason = "Launches through the raw-address kernel contract with buffers owned by this function"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "The fixture values are small integers that fit exactly in f32"
)]

use infer_cuda_kernels::aot::{AotProjection, Device};
use infer_cuda_kernels::contract::{Arch, DevicePtr, LinearPlan, Profile, Projection, Target};
use std::env;
use std::error::Error;
use std::mem::size_of;
use std::path::Path;

/// Output rows of every fixture.
const ROWS: usize = 8;
/// Reduction lengths around the warp width, including partial warps.
const COLUMNS: [usize; 5] = [1, 31, 32, 64, 129];
/// Relative tolerance: a few F32 rounding units on the summed magnitude of the row.
const RELATIVE_TOLERANCE: f32 = 1.0e-5;

fn check(
    device: &Device,
    projection: &AotProjection,
    columns: usize,
) -> Result<(), Box<dyn Error>> {
    let x: Vec<f32> = (0..columns)
        .map(|i| ((i * 7) % 13) as f32 / 8.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..ROWS * columns)
        .map(|i| ((i * 11) % 17) as f32 / 8.0 - 1.0)
        .collect();
    let x_buffer = device.alloc(columns * size_of::<f32>())?;
    let w_buffer = device.alloc(ROWS * columns * size_of::<f32>())?;
    let out_buffer = device.alloc(ROWS * size_of::<f32>())?;
    device.upload_f32(&x_buffer, &x)?;
    device.upload_f32(&w_buffer, &w)?;
    let plan = LinearPlan {
        rows: ROWS,
        columns,
        block: columns,
        tile_rows: 1,
    };
    // SAFETY: the three buffers match the plan, do not alias, and outlive the synchronization
    // performed by the download below.
    unsafe {
        projection.dense_f32(
            &plan,
            DevicePtr(out_buffer.pointer()),
            DevicePtr(x_buffer.pointer()),
            DevicePtr(w_buffer.pointer()),
        )?;
    }
    let mut actual = vec![0.0f32; ROWS];
    device.download_f32(&out_buffer, &mut actual)?;
    for (row, value) in actual.iter().enumerate() {
        let terms = (0..columns).map(|k| w[row * columns + k] * x[k]);
        let expected: f32 = terms.clone().sum();
        let magnitude: f32 = terms.map(f32::abs).sum();
        if (value - expected).abs() > magnitude * RELATIVE_TOLERANCE {
            return Err(format!("{columns} columns, row {row}: {value} vs {expected}").into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = env::args()
        .nth(1)
        .unwrap_or_else(|| "target/kernels/aot/sm_120".to_owned());
    let device = Device::open(0)?;
    let projection = AotProjection::load(
        &device,
        Path::new(&root),
        Target {
            arch: Arch::Sm120,
            profile: Profile::Aot,
        },
    )?;
    for columns in COLUMNS {
        check(&device, &projection, columns)?;
    }
    println!("AOT projection passed: {ROWS} rows x {COLUMNS:?} columns");
    Ok(())
}
