#[cfg(cuda_backend)]
use infer_backend_cuda::device::CudaDevice;
#[cfg(cuda_backend)]
use infer_backend_cuda::mlp::Matrix;
#[cfg(cuda_backend)]
fn main() -> infer_core::Result<()> {
    let device = CudaDevice::new(0)?;
    let x = device.runtime().upload(vec![1.0f32, 2.0, 3.0, 4.0])?;
    let w = Matrix::upload(&device, vec![1.0f32; 32], &[8, 4])?;
    let y = device.matvec(&x, &w)?;
    let result = device.runtime().read(&y)?;
    if result != vec![10.0f32; 8] {
        return Err(infer_core::Error::invariant(format!(
            "CUDA matvec mismatch: {result:?}"
        )));
    }
    println!("Rust CUDA matrix-vector kernel passed");
    quantized(&device)?;
    Ok(())
}

#[cfg(cuda_backend)]
fn quantized(device: &CudaDevice) -> infer_core::Result<()> {
    use cuda_core::{f4e2m1fnx2, f8e4m3fn};
    use infer_backend_cuda::strategy::{DecodeSearch, LinearStrategy};
    let runtime = device.runtime();
    let x = runtime.upload(vec![1.0f32; 32])?;
    let fp8 = Matrix::upload(device, vec![f8e4m3fn(0x38); 7 * 32], &[7, 32])?;
    let channel = runtime.upload(vec![0.5f32; 7])?;
    let fp4 = Matrix::upload(device, vec![f4e2m1fnx2::from_bits(0x21); 7 * 16], &[7, 16])?;
    let blocks = Matrix::upload(device, vec![f8e4m3fn(0x38); 7 * 2], &[7, 2])?;
    for tile in DecodeSearch.candidates(32) {
        let y8 = device.fp8_matvec(&x, &fp8, &channel, tile)?;
        let y4 = device.nvfp4_matvec(&x, &fp4, &blocks, 2.0, tile)?;
        if runtime.read(&y8)? != vec![16.0f32; 7] || runtime.read(&y4)? != vec![12.0f32; 7] {
            return Err(infer_core::Error::invariant(
                "quantized CUDA numerical mismatch",
            ));
        }
    }
    println!("Rust CUDA FP8/NVFP4 reference kernels passed all tile candidates and row tails");
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA kernels require Linux");
    exit(1);
}
