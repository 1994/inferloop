//! Print the HF configuration synthesized from a Qwen3-VL text GGUF and its mmproj.
use infer_format::GgufFile;
use infer_models::TensorSpec;
use std::env::args;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let text = arguments
        .next()
        .ok_or("usage: gguf-config <text.gguf> <mmproj.gguf>")?;
    let mmproj = arguments
        .next()
        .ok_or("usage: gguf-config <text.gguf> <mmproj.gguf>")?;
    let text = GgufFile::open(text)?;
    let mmproj = GgufFile::open(mmproj)?;
    let config = infer_models::config(text.metadata(), mmproj.metadata())?;
    println!("{}", String::from_utf8_lossy(&config));

    let unmapped: Vec<&str> = text
        .tensors()
        .keys()
        .filter(|name| infer_models::text_tensor(name).is_none())
        .map(String::as_str)
        .collect();
    println!(
        "text tensors={} unmapped={unmapped:?}",
        text.tensors().len()
    );
    let unmapped: Vec<&str> = mmproj
        .tensors()
        .keys()
        .filter(|name| infer_models::vision_tensor(name, &[5, 11, 17]).is_none())
        .map(String::as_str)
        .collect();
    println!(
        "vision tensors={} unmapped={unmapped:?}",
        mmproj.tensors().len()
    );
    println!(
        "vision slots={}",
        infer_models::vision_slots(mmproj.metadata())?.len()
    );
    let text_specs = specs(&text)?;
    infer_models::validate_text(text.metadata(), &text_specs)?;
    let vision_specs = specs(&mmproj)?;
    infer_models::validate_vision(mmproj.metadata(), &vision_specs)?;
    println!(
        "validated text={} vision={}",
        text_specs.len(),
        vision_specs.len()
    );
    Ok(())
}

fn specs(file: &GgufFile) -> Result<Vec<TensorSpec>, Box<dyn Error>> {
    file.tensors()
        .values()
        .map(|tensor| {
            Ok(TensorSpec {
                name: tensor.name().to_owned(),
                shape: tensor.shape()?,
                ty: tensor.ty(),
            })
        })
        .collect()
}
