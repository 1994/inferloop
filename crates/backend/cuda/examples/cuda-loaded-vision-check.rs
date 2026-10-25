//! Loaded-model vision binding check: the tower is bound by `LoadedModel` and encodes a golden image.
//!
//! The text model is loaded alongside the tower, so this exercises the real binding path rather
//! than a standalone weight map.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use std::env::args as env_args;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use infer_backend_cuda::{
        device::CudaDevice,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{Error, ModelId};
    use infer_models::{PromptImage, SafetensorsFile};

    let mut args = env_args().skip(1);
    let root = args
        .next()
        .ok_or_else(|| Error::invalid("model package path required"))?;
    let golden = args
        .next()
        .ok_or_else(|| Error::invalid("golden safetensors path required"))?;
    enable_default()?;
    let device = CudaDevice::new(0)?;
    let options = LoadOptions {
        autotune: false,
        ..LoadOptions::default()
    };
    let loaded = LoadedModel::open(device, &root, ModelId::ONE, options)?;
    let vision = loaded
        .vision()
        .ok_or_else(|| Error::unsupported("package declares no vision encoder"))?;
    let geometry = vision.encoder().clone();

    let mut file = SafetensorsFile::open(&golden)?;
    let pixels = file.read_f32("c0/pixel_values", 1 << 30)?;
    let reference = file.read_f32("c0/pooler_output", 1 << 30)?;
    let patches = pixels
        .shape
        .first()
        .copied()
        .ok_or_else(|| Error::invalid("golden pixel shape"))?;
    let image = PromptImage {
        pixels: pixels.data,
        grid: (1, 4, 4),
    };
    let encoded = vision.encode(loaded.device(), &image)?;
    if encoded.len() != reference.data.len() {
        return Err(Error::invalid("encoded vision shape").into());
    }
    let mut max_abs = 0f32;
    for (value, gold) in encoded.iter().zip(reference.data.iter()) {
        max_abs = max_abs.max((value - gold).abs());
    }
    let scale = reference
        .data
        .iter()
        .fold(0f32, |scale, value| scale.max(value.abs()));
    let relative = if scale > 0.0 { max_abs / scale } else { 0.0 };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": relative <= 1e-2,
            "package": root,
            "patches": patches,
            "merged_tokens": encoded.len() / geometry.out_hidden_size,
            "out_hidden_size": geometry.out_hidden_size,
            "vision_max_abs": max_abs,
            "vision_max_rel": relative,
            "scope": "LoadedModel vision binding, tower and merger on the golden image",
        }))?
    );
    if relative > 1e-2 {
        return Err(Error::invariant(format!("loaded vision deviates by {relative}")).into());
    }
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA vision checks require Linux");
    exit(1);
}
