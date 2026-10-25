//! End-to-end multimodal generation: a real image drives resident decoding.
//!
//! The prompt is assembled by the provider's own processor and tokenizer, the image is encoded by
//! the bound vision tower, the encodings are placed at the media tokens, and the resident program
//! decodes greedily. Two different images are generated from to show the media actually reaches the
//! language model rather than being silently dropped.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use infer_backend_cuda::loading::LoadedModel as LoadingLoadedModel;
#[cfg(cuda_backend)]
use infer_backend_cuda::resident::DeviceProgram;
#[cfg(cuda_backend)]
use std::env::args as env_args;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
use std::fs::read;
#[cfg(cuda_backend)]
use std::ops::Range;
#[cfg(cuda_backend)]
use std::path::Path;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use infer_backend_cuda::{
        device::CudaDevice,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{Error, ModelId};
    use infer_models::{ChatMessage, ChatOptions, ImageProcessor, TextAssets};

    let mut args = env_args().skip(1);
    let root = args
        .next()
        .ok_or_else(|| Error::invalid("model package path required"))?;
    let budget: usize = match args.next() {
        Some(value) => value
            .parse()
            .map_err(|error| Error::invalid(format!("generation budget: {error}")))?,
        None => 8,
    };
    enable_default()?;
    let device = CudaDevice::new(0)?;
    let options = LoadOptions {
        // A prompt graph turns the media prefill into fused chunks instead of one decode step
        // per token; zero asks the loader to pick the widest graph the package can hold.
        prefill_width: 0,
        autotune: false,
        ..LoadOptions::default()
    };
    let loaded = LoadedModel::open(device, &root, ModelId::ONE, options)?;
    if loaded.vision().is_none() {
        return Err(Error::unsupported("package declares no vision encoder").into());
    }
    let hidden = loaded.model().hidden_size;

    let assets = TextAssets::open(&root, 4096)?;
    let processor = ImageProcessor::from_json(
        &read(Path::new(&root).join("preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let message = ChatMessage::new(
        "user",
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image in a few words.",
    );
    let text = assets.render_chat(&[message], &ChatOptions::default())?;
    let mut program = loaded.sequence(4096)?;
    let mut cases = Vec::new();
    for kind in 0..2u8 {
        cases.push(run_case(
            &mut program,
            &loaded,
            &assets,
            &processor,
            &text,
            kind,
            budget,
            hidden,
        )?);
    }
    // Two different images must move the final prompt logits: identical logits would mean the
    // media placeholders were executed as ordinary tokens and the images were silently dropped.
    let (Some(first), Some(second)) = (cases.first(), cases.get(1)) else {
        return Err(Error::invariant("both image runs must produce logits").into());
    };
    let (first, second) = (&first.logits, &second.logits);
    if first.len() != second.len() {
        return Err(Error::invariant("logit rows differ in width").into());
    }
    let mut difference = 0f32;
    for (left, right) in first.iter().zip(second.iter()) {
        difference = difference.max((left - right).abs());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": difference > MINIMUM_SHIFT,
            "package": root,
            "hidden_size": hidden,
            "runs": cases.iter().map(|case| case.summary.clone()).collect::<Vec<_>>(),
            "logit_shift": difference,
            "minimum_shift": MINIMUM_SHIFT,
            "scope": "image -> vision tower -> resident prefill with media overrides -> greedy decode",
        }))?
    );
    if difference <= MINIMUM_SHIFT {
        return Err(Error::invariant("the images did not move the prompt logits").into());
    }
    Ok(())
}

/// Smallest logit shift that counts as the media reaching the model.
#[cfg(cuda_backend)]
const MINIMUM_SHIFT: f32 = 1e-3;

/// One image's prompt, encodings, final prompt logits and generated continuation.
#[cfg(cuda_backend)]
struct Case {
    summary: serde_json::Value,
    logits: Vec<f32>,
}

/// Assemble the prompt for one image, encode it and generate a continuation.
#[cfg(cuda_backend)]
#[expect(
    clippy::too_many_arguments,
    reason = "The example threads the loaded model, its assets and the generation settings through"
)]
fn run_case(
    program: &mut DeviceProgram,
    loaded: &LoadingLoadedModel,
    assets: &infer_models::TextAssets,
    processor: &infer_models::ImageProcessor,
    text: &str,
    kind: u8,
    budget: usize,
    hidden: usize,
) -> Result<Case, Box<dyn StdError>> {
    use infer_backend_cuda::vision;
    use infer_core::Error;
    use infer_ir::Modality;
    let imported = loaded.imported();
    let vision = loaded
        .vision()
        .ok_or_else(|| Error::unsupported("package declares no vision encoder"))?;
    let prepared = infer_models::prepare(
        assets,
        processor,
        imported,
        text,
        &[synthetic_image(kind)],
        &Modality::Image,
    )?;
    let image = prepared
        .images
        .first()
        .ok_or_else(|| Error::invariant("prepared prompt without an image"))?;
    let encoded = vision.encode(loaded.device(), image)?;
    let encodings = [encoded];
    let prefill = vision::prefill(
        program,
        imported,
        &prepared,
        &Modality::Image,
        &encodings,
        hidden,
        0,
    )?;
    let logits = prefill.logits.clone();
    let generated = vision::decode(program, prepared.tokens.len(), &prefill, budget, &[])?;
    let media_tokens = prepared.spans.iter().map(Range::len).sum::<usize>();
    Ok(Case {
        summary: serde_json::json!({
            "image": kind,
            "grid": [image.grid.0, image.grid.1, image.grid.2],
            "prompt_tokens": prepared.tokens.len(),
            "media_tokens": media_tokens,
            "generated_tokens": generated,
            "generated_text": assets.decode(&generated, true)?,
        }),
        logits,
    })
}

/// Deterministic RGB8 test images: a smooth gradient plus a second, differently ordered gradient.
#[cfg(cuda_backend)]
fn synthetic_image(kind: u8) -> infer_models::RawImage {
    /// Height of the synthetic test image, above the processor's 32-pixel floor.
    const HEIGHT: usize = 60;
    /// Width of the synthetic test image.
    const WIDTH: usize = 100;
    let mut pixels = vec![0u8; HEIGHT * WIDTH * 3];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let slot = (y * WIDTH + x) * 3;
            let vertical = u8::try_from(y * 255 / HEIGHT).unwrap_or(0);
            let horizontal = u8::try_from(x * 255 / WIDTH).unwrap_or(0);
            let (red, green, blue) = if kind == 0 {
                (vertical, horizontal, 128)
            } else {
                (horizontal, 255 - vertical, 64)
            };
            pixels[slot] = red;
            pixels[slot + 1] = green;
            pixels[slot + 2] = blue;
        }
    }
    infer_models::RawImage {
        height: HEIGHT,
        width: WIDTH,
        pixels,
    }
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA multimodal generation requires Linux");
    exit(1);
}
