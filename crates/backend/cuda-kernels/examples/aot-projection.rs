//! Load an AOT projection artifact and compare it with a CPU reference.
#![expect(
    unsafe_code,
    reason = "The example converts typed slices to bytes for device copies"
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
use std::slice;
use std::sync::Arc;

const fn bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: every call passes f32 data, which has no padding and is valid for any bit pattern.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) }
}

const fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: every call passes f32 data; the host buffer is writable for its full length.
    unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), size_of_val(values)) }
}

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
    device.upload(&x_buffer, bytes(&x))?;
    device.upload(&w_buffer, bytes(&w))?;
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
    device.download(&out_buffer, bytes_mut(&mut actual))?;
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
