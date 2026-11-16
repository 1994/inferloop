//! Open a native GGUF text package and count its slots per grouped-int format.
use infer_core::ModelId;
use infer_models::{GgufPackage, MultimodalAssets, QuantizedPackage, TextAssets, default_registry};
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = args().nth(1).ok_or("usage: gguf-package <model.gguf>")?;
    let package = GgufPackage::open(&path)?;
    println!("slots={}", package.slots().len());
    let mut formats = std::collections::BTreeMap::<&str, usize>::new();
    for slot in package.slots().values() {
        if let Some(format) = slot.dtype.grouped_int() {
            *formats.entry(format.label()).or_default() += 1;
        }
    }
    println!("grouped={formats:?}");
    println!(
        "config={} tokenizer={}",
        package.config().len(),
        package.tokenizer().len()
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(package.tokenizer())
        .map_err(|error| format!("tokenizer: {error}"))?;
    println!(
        "tokens={:?}",
        tokenizer
            .encode("Hello, world!", false)
            .map_err(|error| format!("encode: {error}"))?
            .get_ids()
    );
    let package = QuantizedPackage::open_gguf(
        default_registry(),
        std::path::Path::new(&path),
        ModelId::ONE,
    )?;
    let mut grouped = 0;
    for weight in package.weights.values() {
        if matches!(
            weight.encoding(),
            infer_models::WeightEncoding::GroupedInt(_)
        ) {
            grouped += 1;
        }
    }
    println!(
        "graph weights={} grouped={grouped} vision={}",
        package.weights.len(),
        package.vision_source_count()
    );
    let assets = TextAssets::open(&path, 4096)?;
    println!("assets tokens={:?}", assets.encode("Hello, world!", false)?);
    println!(
        "multimodal={}",
        MultimodalAssets::open(std::path::Path::new(&path))?.is_some()
    );
    Ok(())
}
