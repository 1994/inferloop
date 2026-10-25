//! Multi-image and video input through the fused media prefill.
//!
//! One prompt carries two images, and a second run swaps their order. A third and fourth run
//! prefill the same clip forwards and backwards. The token streams are identical within each
//! pair, so a logit shift proves the extra media actually reaches the language model: the pair
//! checks that both images occupy their own spans, the clip pair checks that timestamped
//! temporal groups carry real motion instead of a repeated still.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use infer_backend_cuda::loading::LoadedModel as LoadingLoadedModel;
#[cfg(cuda_backend)]
use infer_backend_cuda::resident::DeviceProgram;
#[cfg(cuda_backend)]
use std::env::args as env_args;
#[cfg(cuda_backend)]
use std::env::var;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
use std::fs::read;
#[cfg(cuda_backend)]
use std::ops::Range;
#[cfg(cuda_backend)]
use std::path::Path;
#[cfg(cuda_backend)]
use std::slice::from_ref;
#[cfg(cuda_backend)]
use std::time::Instant;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use infer_backend_cuda::{
        device::CudaDevice,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{Error, ModelId};
    use infer_models::{ImageProcessor, TextAssets, VideoProcessor};

    let (root, budget) = arguments()?;
    enable_default()?;
    // A prompt graph turns the media prefill into fused chunks; `INFER_MEDIA_PREFILL_WIDTH=1`
    // selects the per-token fallback so the two paths can be timed against each other.
    let prefill_width = var("INFER_MEDIA_PREFILL_WIDTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let device = CudaDevice::new(0)?;
    let options = LoadOptions {
        prefill_width,
        autotune: false,
        ..LoadOptions::default()
    };
    let loaded = LoadedModel::open(device, &root, ModelId::ONE, options)?;
    if loaded.vision().is_none() {
        return Err(Error::unsupported("package declares no vision encoder").into());
    }
    let hidden = loaded.model().hidden_size;
    let assets = TextAssets::open(&root, 4096)?;
    let images = ImageProcessor::from_json(
        &read(Path::new(&root).join("preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let video = VideoProcessor::from_json(
        &read(Path::new(&root).join("video_preprocessor_config.json"))
            .map_err(|error| Error::invalid(error.to_string()))?,
    )?;
    let mut program = loaded.sequence(4096)?;

    let (multi_image, pair_shift) =
        multi_image_case(&mut program, &loaded, &assets, &images, budget)?;
    let (video, video_shift) = video_case(&mut program, &loaded, &assets, &video, budget)?;
    let passed = pair_shift > MINIMUM_SHIFT && video_shift > MINIMUM_SHIFT;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": passed,
            "package": root,
            "hidden_size": hidden,
            "prefill_width": program.prefill_width(),
            "multi_image": multi_image,
            "video": video,
            "minimum_shift": MINIMUM_SHIFT,
            "scope": "two images in one prompt and a multi-frame clip -> vision tower -> fused media prefill -> greedy decode",
        }))?
    );
    if !passed {
        return Err(Error::invariant("media did not move the prompt logits").into());
    }
    Ok(())
}

/// Smallest logit shift that counts as the media reaching the model.
#[cfg(cuda_backend)]
const MINIMUM_SHIFT: f32 = 1e-3;

/// Model package path and greedy budget from the command line.
#[cfg(cuda_backend)]
fn arguments() -> Result<(String, usize), Box<dyn StdError>> {
    use infer_core::Error;

    let mut args = env_args().skip(1);
    let root = args
        .next()
        .ok_or_else(|| Error::invalid("model package path required"))?;
    let budget: usize = match args.next() {
        Some(value) => value
            .parse()
            .map_err(|error| Error::invalid(format!("generation budget: {error}")))?,
        None => 6,
    };
    Ok((root, budget))
}

/// Two images in one prompt, then the same images in the opposite order.
#[cfg(cuda_backend)]
fn multi_image_case(
    program: &mut DeviceProgram,
    loaded: &LoadingLoadedModel,
    assets: &infer_models::TextAssets,
    processor: &infer_models::ImageProcessor,
    budget: usize,
) -> Result<(serde_json::Value, f32), Box<dyn StdError>> {
    use infer_ir::Modality;
    use infer_models::{ChatMessage, ChatOptions, prepare};

    let text = "<|vision_start|><|image_pad|><|vision_end|>\
                <|vision_start|><|image_pad|><|vision_end|>\
                Do these two images differ?";
    let prompt = assets.render_chat(&[ChatMessage::new("user", text)], &ChatOptions::default())?;
    let images = [synthetic_image(0), synthetic_image(1)];
    let forward = prepare(
        assets,
        processor,
        loaded.imported(),
        &prompt,
        &images,
        &Modality::Image,
    )?;
    let reversed = prepare(
        assets,
        processor,
        loaded.imported(),
        &prompt,
        &[images[1].clone(), images[0].clone()],
        &Modality::Image,
    )?;
    let forward = run(program, loaded, &forward, &Modality::Image, budget)?;
    let reversed = run(program, loaded, &reversed, &Modality::Image, budget)?;
    let shift = logit_shift(&forward.logits, &reversed.logits)?;
    let grids: Vec<[usize; 3]> = forward
        .prepared
        .images
        .iter()
        .map(|image| image.grid.into())
        .collect();
    let summary = serde_json::json!({
        "grids": grids,
        "prompt_tokens": forward.prepared.tokens.len(),
        "media_tokens": media_tokens(&forward.prepared),
        "logit_shift": shift,
        "prefill_millis": forward.millis,
        "forward": {
            "tokens": forward.tokens,
            "text": assets.decode(&forward.tokens, true)?,
        },
        "reversed": {
            "tokens": reversed.tokens,
            "text": assets.decode(&reversed.tokens, true)?,
        },
    });
    Ok((summary, shift))
}

/// One clip forwards and backwards: identical tokens, different motion.
#[cfg(cuda_backend)]
fn video_case(
    program: &mut DeviceProgram,
    loaded: &LoadingLoadedModel,
    assets: &infer_models::TextAssets,
    processor: &infer_models::VideoProcessor,
    budget: usize,
) -> Result<(serde_json::Value, f32), Box<dyn StdError>> {
    use infer_core::Error;
    use infer_ir::Modality;
    use infer_models::{ChatMessage, ChatOptions, prepare_video};

    let text = "<|vision_start|><|video_pad|><|vision_end|>Describe the motion in this clip.";
    let prompt = assets.render_chat(&[ChatMessage::new("user", text)], &ChatOptions::default())?;
    let mut clip = synthetic_video();
    let forward = prepare_video(
        assets,
        processor,
        loaded.imported(),
        &prompt,
        from_ref(&clip),
        &Modality::Video,
    )?;
    clip.frames.reverse();
    let reversed = prepare_video(
        assets,
        processor,
        loaded.imported(),
        &prompt,
        &[clip],
        &Modality::Video,
    )?;
    if forward.images.len() < 2 {
        return Err(Error::invariant("the clip must span several temporal groups").into());
    }
    let forward = run(program, loaded, &forward, &Modality::Video, budget)?;
    let reversed = run(program, loaded, &reversed, &Modality::Video, budget)?;
    let shift = logit_shift(&forward.logits, &reversed.logits)?;
    let groups = forward.prepared.images.len();
    let grid = [
        groups,
        forward.prepared.images[0].grid.1,
        forward.prepared.images[0].grid.2,
    ];
    let summary = serde_json::json!({
        "groups": groups,
        "grid": grid,
        "prompt_tokens": forward.prepared.tokens.len(),
        "media_tokens": media_tokens(&forward.prepared),
        "logit_shift": shift,
        "prefill_millis": forward.millis,
        "forward": {
            "tokens": forward.tokens,
            "text": assets.decode(&forward.tokens, true)?,
        },
        "reversed": {
            "tokens": reversed.tokens,
            "text": assets.decode(&reversed.tokens, true)?,
        },
    });
    Ok((summary, shift))
}

/// One completed run: its prompt, logits, greedy tokens and prefill time.
#[cfg(cuda_backend)]
struct Run {
    prepared: infer_models::PreparedPrompt,
    logits: Vec<f32>,
    tokens: Vec<u32>,
    millis: f64,
}

/// Encode every prepared media item, prefill once and decode greedily.
#[cfg(cuda_backend)]
fn run(
    program: &mut DeviceProgram,
    loaded: &LoadingLoadedModel,
    prepared: &infer_models::PreparedPrompt,
    modality: &infer_ir::Modality,
    budget: usize,
) -> Result<Run, Box<dyn StdError>> {
    use infer_backend_cuda::vision;
    use infer_core::Error;

    let tower = loaded
        .vision()
        .ok_or_else(|| Error::unsupported("package declares no vision encoder"))?;
    let hidden = loaded.model().hidden_size;
    let mut encodings = Vec::with_capacity(prepared.images.len());
    for image in &prepared.images {
        encodings.push(tower.encode(loaded.device(), image)?);
    }
    let started = Instant::now();
    let prefilled = vision::prefill(
        program,
        loaded.imported(),
        prepared,
        modality,
        &encodings,
        hidden,
        0,
    )?;
    let millis = started.elapsed().as_secs_f64() * 1_000.0;
    let logits = prefilled.logits.clone();
    let tokens = vision::decode(program, prepared.tokens.len(), &prefilled, budget, &[])?;
    Ok(Run {
        prepared: prepared.clone(),
        logits,
        tokens,
        millis,
    })
}

/// Number of prompt positions the prepared media occupies.
#[cfg(cuda_backend)]
fn media_tokens(prepared: &infer_models::PreparedPrompt) -> usize {
    prepared.spans.iter().map(Range::len).sum()
}

/// Largest absolute logit difference between two runs of the same token stream.
#[cfg(cuda_backend)]
fn logit_shift(left: &[f32], right: &[f32]) -> Result<f32, Box<dyn StdError>> {
    use infer_core::Error;

    if left.len() != right.len() {
        return Err(Error::invariant("logit rows differ in width").into());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0f32, f32::max))
}

/// Deterministic RGB8 test image: a vertical or horizontal gradient.
#[cfg(cuda_backend)]
fn synthetic_image(kind: u8) -> infer_models::RawImage {
    /// Height of the synthetic test image, above the processor's patch floor.
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

/// A moving stripe: the frame index changes the stripe position and the background.
#[cfg(cuda_backend)]
fn synthetic_video() -> infer_models::RawVideo {
    /// Frames decoded from the synthetic clip.
    const FRAMES: usize = 8;
    /// Height of the synthetic test clip.
    const HEIGHT: usize = 60;
    /// Width of the synthetic test clip.
    const WIDTH: usize = 100;
    /// Source frame rate the sampler reads.
    const FPS: f64 = 30.0;
    let frames = (0..FRAMES)
        .map(|frame| {
            let mut pixels = vec![0u8; HEIGHT * WIDTH * 3];
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let slot = (y * WIDTH + x) * 3;
                    let stripe = if x.abs_diff(frame * WIDTH / FRAMES) < 10 {
                        200
                    } else {
                        40
                    };
                    pixels[slot] = u8::try_from((y * 255 / HEIGHT + frame * 20) % 256).unwrap_or(0);
                    pixels[slot + 1] = u8::try_from(stripe).unwrap_or(0);
                    pixels[slot + 2] =
                        u8::try_from((x * 255 / WIDTH + frame * 10) % 256).unwrap_or(0);
                }
            }
            infer_models::RgbImage {
                height: HEIGHT,
                width: WIDTH,
                pixels,
            }
        })
        .collect();
    infer_models::RawVideo { fps: FPS, frames }
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA multimodal checks require Linux");
    exit(1);
}
