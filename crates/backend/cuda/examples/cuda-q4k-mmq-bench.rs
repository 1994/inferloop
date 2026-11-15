//! Two-term int8 MMQ N64 bandwidth probe on synthetic Q4_K-packed data.
use infer_backend_cuda::device::CudaDevice;
use infer_cuda_kernels::contract::Q4kInt8TwoTermProbe;
use std::error::Error;
use std::time::Instant;

const ROWS: usize = 16;
const OUTPUTS: usize = 6144;
const COLUMNS: usize = 2048;
const GROUPS: usize = COLUMNS / 32;
const TILES: usize = OUTPUTS / 64;
const REPLAYS: usize = 20;
const HBM_GBPS: f64 = 1792.0;

fn main() -> Result<(), Box<dyn Error>> {
    let codes: Vec<i8> = (0..TILES * GROUPS * 32 * 64)
        .map(|index| i8::try_from(index % 16).unwrap_or(0))
        .collect();
    let factors: Vec<f32> = (0..TILES * GROUPS * 64)
        .map(|index| f32::from(u16::try_from(index % 97).unwrap_or(0)) * 0.001)
        .collect();
    let biases: Vec<f32> = (0..TILES * GROUPS * 64)
        .map(|index| f32::from(u16::try_from(index % 31).unwrap_or(0)) * 0.0001)
        .collect();
    let scales = vec![0.01f32; TILES * GROUPS * ROWS];
    let fine_scales = vec![0.0001f32; TILES * GROUPS * ROWS];
    let sums = vec![0.25f32; TILES * GROUPS * ROWS];
    let x: Vec<i8> = (0..ROWS * COLUMNS)
        .map(|index| i8::try_from(index % 19).unwrap_or(0) - 9)
        .collect();
    let residual: Vec<i8> = (0..ROWS * COLUMNS)
        .map(|index| i8::try_from(index % 7).unwrap_or(0) - 3)
        .collect();
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let x_alloc = runtime.upload(x)?;
    let residual_alloc = runtime.upload(residual)?;
    let codes_alloc = runtime.upload(codes)?;
    let factors_alloc = runtime.upload(factors)?;
    let biases_alloc = runtime.upload(biases)?;
    let scales_alloc = runtime.upload(scales)?;
    let fine_alloc = runtime.upload(fine_scales)?;
    let sums_alloc = runtime.upload(sums)?;
    let out_alloc = runtime.zeros::<f32>(ROWS * OUTPUTS)?;
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels.projection.q4k_int8_two_term_n64(
            scope,
            &Q4kInt8TwoTermProbe {
                groups: GROUPS,
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
    let started = Instant::now();
    for _ in 0..REPLAYS {
        device.tile().sync(graph.launch())?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let bytes = codes_alloc.len() * size_of::<i8>()
        + factors_alloc.len() * size_of::<f32>()
        + biases_alloc.len() * size_of::<f32>();
    let bytes = f64::from(u32::try_from(bytes).unwrap_or(u32::MAX));
    let replays = f64::from(u32::try_from(REPLAYS).unwrap_or(u32::MAX));
    let bandwidth = bytes * replays / elapsed / 1e9;
    println!(
        "N64 MMQ {bandwidth:.1} GB/s ({:.1}% roofline) per-launch {:.3} ms",
        bandwidth / HBM_GBPS * 100.0,
        elapsed * 1000.0 / replays
    );
    Ok(())
}
