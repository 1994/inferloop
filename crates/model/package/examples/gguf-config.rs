//! Print the HF configuration synthesized from a text GGUF and, when given, its mmproj.
use infer_models::{GgufPackage, default_registry};
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let text = arguments
        .next()
        .ok_or("usage: gguf-config <text.gguf> [mmproj.gguf]")?;
    let package = match arguments.next() {
        Some(mmproj) => GgufPackage::open_with_vision(default_registry(), text, mmproj)?,
        None => GgufPackage::open(text)?,
    };
    println!("{}", String::from_utf8_lossy(package.config()));
    println!(
        "architecture={} slots={} vision_slots={}",
        package.architecture(),
        package.slots().len(),
        package.vision().map_or(0, |vision| vision.slots().len())
    );
    Ok(())
}
