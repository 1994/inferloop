//! Loaded-model vision binding check: the tower is bound by `LoadedModel` and encodes a golden image.
//!
//! The text model is loaded alongside the tower, so this exercises the real binding path rather
//! than a standalone weight map.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use std::env;
#[cfg(cuda_backend)]
use std::env::args as env_args;
use std::error::Error as StdError;
use std::fs;
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

    let grids = [(1, 4, 4), (1, 4, 6), (2, 4, 4), (1, 8, 8), (1, 16, 16)];
    let mut file = SafetensorsFile::open(&golden)?;
    let mut cases = Vec::new();
    let mut worst = 0.0f32;
    for (index, grid) in grids.into_iter().enumerate() {
        let pixels = file.read_f32(&format!("c{index}/pixel_values"), 1 << 30)?;
        let reference = file.read_f32(&format!("c{index}/pooler_output"), 1 << 30)?;
        let patches = pixels
            .shape
            .first()
            .copied()
            .ok_or_else(|| Error::invalid("golden pixel shape"))?;
        let expected_patches = grid.0 * grid.1 * grid.2;
        if patches != expected_patches {
            return Err(Error::invalid(format!(
                "case {index}: golden has {patches} patches, grid expects {expected_patches}"
            ))
            .into());
        }
        let image = PromptImage {
            pixels: pixels.data,
            grid,
        };
        let encoded = vision.encode(loaded.device(), &image)?;
        if encoded.len() != reference.data.len() {
            return Err(Error::invalid(format!("case {index}: encoded vision shape")).into());
        }
        if !encoded.iter().all(|value| value.is_finite()) {
            return Err(Error::invalid(format!("case {index}: non-finite vision value")).into());
        }
        if let Ok(directory) = env::var("VISION_DUMP") {
            fs::create_dir_all(&directory)?;
            let bytes: Vec<u8> = encoded
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            fs::write(format!("{directory}/c{index}.f32"), bytes)?;
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
        if relative > 1e-2 {
            return Err(Error::invariant(format!("case {index} deviates by {relative}")).into());
        }
        worst = worst.max(relative);
        cases.push(serde_json::json!({
            "index": index,
            "grid_thw": [grid.0, grid.1, grid.2],
            "patches": patches,
            "merged_tokens": encoded.len() / geometry.out_hidden_size,
            "vision_max_abs": max_abs,
            "vision_max_rel": relative,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": true,
            "package": root,
            "golden": golden,
            "out_hidden_size": geometry.out_hidden_size,
            "vision_inventory_bytes": vision.inventory_bytes(),
            "cases": cases,
            "worst_vision_max_rel": worst,
            "scope": "LoadedModel full vision tower and merger over all official golden cases",
        }))?
    );
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA vision checks require Linux");
    exit(1);
}
