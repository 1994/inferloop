//! Decode a GGUF tensor with the reference decoder and print its first values as F32 bits.
use infer_format::{GgufFile, dequantize};
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: gguf-dequant <file> <tensor> [count]")?;
    let name = arguments
        .next()
        .ok_or("usage: gguf-dequant <file> <tensor> [count]")?;
    let count: usize = arguments.next().unwrap_or_else(|| "16".into()).parse()?;
    let file = GgufFile::open(path)?;
    let info = file
        .tensor(&name)
        .ok_or_else(|| format!("missing tensor {name}"))?
        .clone();
    let mut values = vec![0f32; usize::try_from(info.elements()?)?];
    let bytes = file.read_tensor(&name, u64::MAX)?;
    dequantize(info.ty(), &bytes, &mut values)?;
    for value in values.iter().take(count) {
        println!("{:08x}", value.to_bits());
    }
    Ok(())
}
