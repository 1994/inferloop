//! Open a native GGUF text package, list slots and repack one Q4_K/Q6_K tensor.
use infer_core::ModelId;
use infer_models::{GgufPackage, QuantizedPackage, default_registry};
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = args().nth(1).ok_or("usage: gguf-package <model.gguf>")?;
    let package = GgufPackage::open(&path)?;
    println!("slots={}", package.slots().len());
    let mut q4k = 0;
    let mut q6k = 0;
    for slot in package.slots().values() {
        match slot.dtype {
            infer_format::GgufType::Q4K => q4k += 1,
            infer_format::GgufType::Q6K => q6k += 1,
            _ => {}
        }
    }
    println!("q4k={q4k} q6k={q6k}");
    if let Some(slot) = package
        .slots()
        .values()
        .find(|slot| slot.dtype == infer_format::GgufType::Q4K)
    {
        let layout = package.q4k(&slot.slot, u64::MAX)?;
        println!(
            "q4k slot={} rows={} columns={}",
            slot.slot,
            layout.rows(),
            layout.columns()
        );
    }
    if let Some(slot) = package
        .slots()
        .values()
        .find(|slot| slot.dtype == infer_format::GgufType::Q6K)
    {
        let layout = package.q6k(&slot.slot, u64::MAX)?;
        println!(
            "q6k slot={} rows={} columns={}",
            slot.slot,
            layout.rows(),
            layout.columns()
        );
    }
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
            infer_models::WeightEncoding::GroupedInt4 | infer_models::WeightEncoding::GroupedInt6
        ) {
            grouped += 1;
        }
    }
    println!(
        "graph weights={} grouped={grouped} vision={}",
        package.weights.len(),
        package.vision_source_count()
    );
    Ok(())
}
