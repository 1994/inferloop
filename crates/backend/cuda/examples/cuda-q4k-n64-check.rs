//! Check the 64-output two-term int8 MMQ against the reference on a real `Q4_K` tensor.
use infer_backend_cuda::device::CudaDevice;
use infer_cuda_kernels::contract::Q4kInt8TwoTermProbe;
use infer_models::{GgufPackage, Q4KTiled};
use std::env::args;
use std::error::Error;

const ROWS: usize = 16;
const OUT: usize = 64;
const RESIDUAL_SCALE: f32 = 0.001;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: cuda-q4k-n64-check <gguf> <slot>")?;
    let slot_name = arguments
        .next()
        .ok_or("usage: cuda-q4k-n64-check <gguf> <slot>")?;
    let package = GgufPackage::open(path)?;
    let slot = package
        .slots()
        .get(&slot_name)
        .ok_or("missing slot")?
        .clone();
    let columns = slot.shape[1];
    let groups = columns / 32;
    let layout = Q4KTiled::from_device(&package.q4k(&slot_name, u64::MAX)?)?;
    let values = package.dense(&slot_name, u64::MAX)?;
    let (main, residual, x, sums) = inputs(columns, groups);
    let (codes, factors, biases) = packed(&layout, columns, groups);
    let mut expected = vec![0.0f32; ROWS * OUT];
    for row in 0..ROWS {
        for output in 0..OUT {
            expected[row * OUT + output] = values[output * columns..][..columns]
                .iter()
                .zip(&x[row * columns..][..columns])
                .map(|(weight, activation)| weight * activation)
                .sum();
        }
    }
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let x_alloc = runtime.upload(main)?;
    let residual_alloc = runtime.upload(residual)?;
    let codes_alloc = runtime.upload(codes)?;
    let factors_alloc = runtime.upload(factors)?;
    let biases_alloc = runtime.upload(biases)?;
    let scales_alloc = runtime.upload(vec![1.0f32; groups * ROWS])?;
    let fine_alloc = runtime.upload(vec![RESIDUAL_SCALE; groups * ROWS])?;
    let sums_alloc = runtime.upload(sums)?;
    let out_alloc = runtime.zeros::<f32>(ROWS * OUT)?;
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels.projection.q4k_int8_two_term_n64(
            scope,
            &Q4kInt8TwoTermProbe {
                groups,
                out: out_alloc.buffer(),
                x: x_alloc.buffer(),
                residual: residual_alloc.buffer(),
                codes: codes_alloc.buffer(),
                factors: factors_alloc.buffer(),
                biases: biases_alloc.buffer(),
                scales: scales_alloc.buffer(),
                residual_scales: fine_alloc.buffer(),
                sums: sums_alloc.buffer(),
            },
        )
    })?;
    device.tile().sync(graph.launch())?;
    let actual = runtime.read(&out_alloc)?;
    let mut worst = 0.0f32;
    for (actual, expected) in actual.iter().zip(&expected) {
        worst = worst.max((actual - expected).abs());
    }
    println!("N64 real check max-abs={worst:.3e}");
    Ok(())
}

fn inputs(columns: usize, groups: usize) -> (Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>) {
    let mut main = vec![0i8; ROWS * columns];
    let mut residual = vec![0i8; ROWS * columns];
    for index in 0..ROWS * columns {
        main[index] = i8::try_from(index % 19).unwrap_or(0) - 9;
        residual[index] = i8::try_from(index % 7).unwrap_or(0) - 3;
    }
    let x: Vec<f32> = main
        .iter()
        .zip(&residual)
        .map(|(main, residual)| {
            f32::mul_add(f32::from(*residual), RESIDUAL_SCALE, f32::from(*main))
        })
        .collect();
    let mut sums = vec![0.0f32; groups * ROWS];
    for row in 0..ROWS {
        for group in 0..groups {
            sums[group * ROWS + row] = x[row * columns + group * 32..][..32].iter().sum();
        }
    }
    (main, residual, x, sums)
}

fn packed(layout: &Q4KTiled, columns: usize, groups: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let mut codes = vec![0i8; groups * 32 * OUT];
    let mut factors = vec![0.0f32; groups * OUT];
    let mut biases = vec![0.0f32; groups * OUT];
    let groups64 = columns / 64;
    for tile in 0..OUT / 16 {
        for group in 0..groups {
            for lane in 0..32 {
                for row in 0..16 {
                    let packed =
                        layout.codes()[(tile * groups64 + group / 2) * 512 + lane * 16 + row];
                    let code = if group.is_multiple_of(2) {
                        packed & 0x0F
                    } else {
                        packed >> 4
                    };
                    codes[(group * 32 + lane) * OUT + tile * 16 + row] = code.cast_signed();
                }
            }
            for row in 0..16 {
                factors[group * OUT + tile * 16 + row] =
                    layout.factors()[tile * groups * 16 + group * 16 + row];
                biases[group * OUT + tile * 16 + row] =
                    layout.biases()[tile * groups * 16 + group * 16 + row];
            }
        }
    }
    (codes, factors, biases)
}
