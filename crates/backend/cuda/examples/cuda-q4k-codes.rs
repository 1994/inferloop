//! Isolate `Q4_K` code unpacking: one 64-value group through the int8 MMA probe.
use infer_backend_cuda::device::CudaDevice;
use infer_models::{GgufPackage, Q4KTiled};
use std::env::args;
use std::error::Error;

const ROWS: usize = 16;
const COLS: usize = 16;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: cuda-q4k-codes <gguf> <slot>")?;
    let slot_name = arguments
        .next()
        .ok_or("usage: cuda-q4k-codes <gguf> <slot>")?;
    let package = GgufPackage::open(path)?;
    let layout = Q4KTiled::from_device(&package.q4k(&slot_name, u64::MAX)?)?;
    let mut codes = vec![0i8; 64 * COLS];
    for lane in 0..32 {
        for row in 0..COLS {
            let packed = layout.codes()[lane * 16 + row];
            codes[lane * COLS + row] = (packed & 0x0F).cast_signed();
            codes[(lane + 32) * COLS + row] = (packed >> 4).cast_signed();
        }
    }
    let x: Vec<i8> = (0..ROWS * 64)
        .map(|index| i8::try_from(index % 19).unwrap_or(0) - 9)
        .collect();
    let x_f32: Vec<f32> = x.iter().map(|value| f32::from(*value)).collect();
    let scales = [1.0f32; ROWS];
    let mut sums = [0.0f32; ROWS];
    for row in 0..ROWS {
        sums[row] = x_f32[row * 64..(row + 1) * 64].iter().sum();
    }
    let values = package.dense(&slot_name, u64::MAX)?;
    let mut expected = vec![0i32; ROWS * COLS];
    for row in 0..ROWS {
        for column in 0..COLS {
            expected[row * COLS + column] = (0..64)
                .map(|k| i32::from(x[row * 64 + k]) * i32::from(codes[k * COLS + column]))
                .sum();
        }
    }
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let x_alloc = runtime.upload(x)?;
    let codes_alloc = runtime.upload(codes)?;
    let out_alloc = runtime.zeros::<i32>(ROWS * COLS)?;
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels.projection.int8_probe(
            scope,
            x_alloc.buffer(),
            codes_alloc.buffer(),
            out_alloc.buffer(),
        )
    })?;
    device.tile().sync(graph.launch())?;
    let actual = runtime.read(&out_alloc)?;
    let mut worst = 0i32;
    for (actual, expected) in actual.iter().zip(&expected) {
        worst = worst.max((actual - expected).abs());
    }
    if worst != 0 {
        return Err(format!("Q4_K code product differs by {worst}").into());
    }
    let mut contribution = vec![0.0f32; ROWS * COLS];
    let mut reference = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        for output in 0..COLS {
            let factor = layout.factors()[output];
            let bias = layout.biases()[output];
            contribution[row * COLS + output] = {
                let product = f32::from(i16::try_from(actual[row * COLS + output]).unwrap_or(0));
                factor.mul_add(product * scales[row], -bias * sums[row])
            };
            reference[row * COLS + output] = values[output * layout.columns()..][..64]
                .iter()
                .zip(&x_f32[row * 64..][..64])
                .map(|(weight, activation)| weight * activation)
                .sum();
        }
    }
    let mut relative = 0.0f64;
    for (actual, expected) in contribution.iter().zip(&reference) {
        relative =
            relative.max(f64::from((actual - expected).abs()) / f64::from(expected.abs().max(1.0)));
    }
    println!("Q4_K code unpack exact; group-scale diagnostic worst relative {relative:.3e}");
    Ok(())
}
