//! int8 MMA probe: validate cuTile `mmai` against a scalar integer reference.
use infer_backend_cuda::device::CudaDevice;
use std::error::Error;

const M: usize = 16;
const K: usize = 64;
const N: usize = 16;

fn main() -> Result<(), Box<dyn Error>> {
    let lhs: Vec<i8> = (0..M * K)
        .map(|index| i8::try_from((index * 7) % 31).unwrap_or(0) - 15)
        .collect();
    let rhs: Vec<i8> = (0..K * N)
        .map(|index| i8::try_from((index * 5) % 27).unwrap_or(0) - 13)
        .collect();
    let mut expected = vec![0i32; M * N];
    for row in 0..M {
        for column in 0..N {
            expected[row * N + column] = (0..K)
                .map(|k| i32::from(lhs[row * K + k]) * i32::from(rhs[k * N + column]))
                .sum();
        }
    }
    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let lhs_alloc = runtime.upload(lhs)?;
    let rhs_alloc = runtime.upload(rhs)?;
    let out_alloc = runtime.zeros::<i32>(M * N)?;
    let kernels = device.kernels();
    let graph = device.tile().capture(|scope| {
        kernels.projection.int8_probe(
            scope,
            lhs_alloc.buffer(),
            rhs_alloc.buffer(),
            out_alloc.buffer(),
        )
    })?;
    device.tile().sync(graph.launch())?;
    let actual = runtime.read(&out_alloc)?;
    if actual != expected {
        return Err(format!("int8 MMA mismatch: {actual:?} vs {expected:?}").into());
    }
    println!("int8 MMA 16x64x16 exact match");
    Ok(())
}
