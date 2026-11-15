//! Execute the native GGUF vision tower on a synthetic image and check two images diverge.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use infer_backend_cuda::device::CudaDevice;
#[cfg(cuda_backend)]
use infer_backend_cuda::loading::{LoadOptions, LoadedModel};
#[cfg(cuda_backend)]
use infer_core::{Error, ModelId};
#[cfg(cuda_backend)]
use infer_models::PromptImage;
#[cfg(cuda_backend)]
use std::env::args;
#[cfg(cuda_backend)]
use std::error::Error as StdError;

#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    let root = args().nth(1).ok_or_else(|| Error::invalid("gguf path"))?;
    enable_default()?;
    let device = CudaDevice::new(0)?;
    let loaded = LoadedModel::open(
        device,
        &root,
        ModelId::ONE,
        LoadOptions {
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    let vision = loaded
        .vision()
        .ok_or_else(|| Error::unsupported("no vision tower"))?;
    let encoder = vision.encoder().clone();
    let grid = (1usize, 4usize, 4usize);
    let patches = grid.0 * grid.1 * grid.2;
    let width =
        encoder.in_channels * encoder.temporal_patch_size * encoder.patch_size * encoder.patch_size;
    let pixels = |offset: f32| -> Vec<f32> {
        (0..patches * width)
            .map(|index| phase(index, offset))
            .collect()
    };
    let first = vision.encode(
        loaded.device(),
        &PromptImage {
            pixels: pixels(0.0),
            grid,
        },
    )?;
    let second = vision.encode(
        loaded.device(),
        &PromptImage {
            pixels: pixels(0.7),
            grid,
        },
    )?;
    if first.len() != second.len() || first.is_empty() {
        return Err(Error::invariant("vision output shape").into());
    }
    if !first.iter().chain(&second).all(|value| value.is_finite()) {
        return Err(Error::invariant("non-finite vision output").into());
    }
    let shift = first
        .iter()
        .zip(&second)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    if shift <= 1e-3 {
        return Err(Error::invariant("images produced identical vision output").into());
    }
    println!(
        "gguf vision passed: merged={} width={} shift={shift:.3e}",
        first.len() / encoder.out_hidden_size,
        encoder.out_hidden_size
    );
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    eprintln!("cuda backend required");
}

#[cfg(cuda_backend)]
fn phase(index: usize, offset: f32) -> f32 {
    let bounded = u16::try_from(index % usize::from(u16::MAX)).unwrap_or(0);
    f32::from(bounded).mul_add(0.013, offset).sin()
}
