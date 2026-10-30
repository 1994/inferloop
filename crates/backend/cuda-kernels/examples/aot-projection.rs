//! Load an AOT projection artifact and compare it with a CPU reference.
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
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    let root = env::args()
        .nth(1)
        .unwrap_or_else(|| "target/kernels/aot/sm_120".to_owned());
    let device = Arc::new(Device::open(0)?);
    let rows = 8usize;
    let columns = 64usize;
    let x: Vec<f32> = (0..columns)
        .map(|i| ((i * 7) % 13) as f32 / 8.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..rows * columns)
        .map(|i| ((i * 11) % 17) as f32 / 8.0 - 1.0)
        .collect();
    let x_buffer = device.alloc(columns * size_of::<f32>())?;
    let w_buffer = device.alloc(rows * columns * size_of::<f32>())?;
    let out_buffer = device.alloc(rows * size_of::<f32>())?;
    device.upload_f32(&x_buffer, &x)?;
    device.upload_f32(&w_buffer, &w)?;
    let projection = AotProjection::load(
        device.clone(),
        Path::new(&root),
        Target {
            arch: Arch::Sm120,
            profile: Profile::Aot,
        },
    )?;
    let plan = LinearPlan {
        rows,
        columns,
        block: columns,
        tile_rows: 1,
    };
    projection.dense_f32(
        &plan,
        DevicePtr(out_buffer.pointer()),
        DevicePtr(x_buffer.pointer()),
        DevicePtr(w_buffer.pointer()),
    )?;
    device.synchronize()?;
    let mut actual = vec![0.0f32; rows];
    device.download_f32(&out_buffer, &mut actual)?;
    for row in 0..rows {
        let expected: f32 = (0..columns).map(|k| w[row * columns + k] * x[k]).sum();
        let tolerance = 1.0e-3 * columns as f32;
        assert!(
            (actual[row] - expected).abs() <= tolerance,
            "row {row}: {} vs {expected}",
            actual[row]
        );
    }
    println!("AOT projection passed: {rows}x{columns}");
    Ok(())
}
