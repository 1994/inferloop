//! Compare the `Q4_K` device-layout GEMV oracle with the reference decode on a real tensor.
use infer_gguf::{GgufFile, GgufType, Q4KDevice, dequantize};
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: gguf-q4k-gemv <file> <tensor>")?;
    let name = arguments
        .next()
        .ok_or("usage: gguf-q4k-gemv <file> <tensor>")?;
    let file = GgufFile::open(path)?;
    let info = file
        .tensor(&name)
        .ok_or_else(|| format!("missing tensor {name}"))?
        .clone();
    if info.ty() != GgufType::Q4K {
        return Err(format!("tensor is {}, not q4_k", info.ty().name()).into());
    }
    let shape = info.shape()?;
    let [rows, columns] = shape.as_slice() else {
        return Err("Q4_K tensor rank".into());
    };
    let blocks = file.read_tensor(&name, u64::MAX)?;
    let device = Q4KDevice::from_blocks(*rows, *columns, &blocks)?;
    let mut values = vec![0.0f32; *rows * *columns];
    dequantize(GgufType::Q4K, &blocks, &mut values)?;
    let ones = vec![1.0f32; *columns];
    for row in 0..(*rows).min(4) {
        let expected: f32 = values[row * *columns..][..*columns].iter().sum();
        let actual = device.row_dot(row, &ones)?;
        let error = (actual - expected).abs();
        let relative = error / expected.abs().max(1.0);
        println!(
            "row {row}: oracle={:08x} reference={:08x} abs={error:.3e} rel={relative:.3e}",
            actual.to_bits(),
            expected.to_bits()
        );
        if relative > 1e-4 {
            return Err(format!("row {row} differs by {relative:.3e}").into());
        }
    }
    Ok(())
}
