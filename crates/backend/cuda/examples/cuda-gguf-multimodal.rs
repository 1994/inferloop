//! GGUF multimodal generation: native text blocks + native mmproj drive one image prompt.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use infer_backend_cuda::device::CudaDevice;
#[cfg(cuda_backend)]
use infer_backend_cuda::loading::{LoadOptions, LoadedModel};
#[cfg(cuda_backend)]
use infer_backend_cuda::resident::DeviceProgram;
#[cfg(cuda_backend)]
use infer_core::{Error, ModelId};
#[cfg(cuda_backend)]
use infer_models::{ChatMessage, ChatOptions, ImageProcessor, PackageSource, TextAssets};
#[cfg(cuda_backend)]
use std::env::args;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
use std::ops::Range;

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
            prefill_width: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    if loaded.vision().is_none() {
        return Err(Error::unsupported("no vision tower").into());
    }
    let hidden = loaded.model().hidden_size;
    let assets = TextAssets::open(&root, 4096)?;
    let source = PackageSource::open(&root)?;
    let processor = ImageProcessor::from_json(source.preprocessor_config()?)?;
    let message = ChatMessage::new(
        "user",
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image in a few words.",
    );
    let text = assets.render_chat(&[message], &ChatOptions::default())?;
    let mut program = loaded.sequence(4096)?;
    let mut logits = Vec::new();
    for kind in 0..2u8 {
        let case = run_case(
            &mut program,
            &loaded,
            &assets,
            &processor,
            &text,
            kind,
            hidden,
        )?;
        println!(
            "image={kind} tokens={} media={} generated={:?} text={:?}",
            case.tokens, case.media, case.generated, case.text
        );
        logits.push(case.logits);
    }
    let shift = logits[0]
        .iter()
        .zip(&logits[1])
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    if shift <= 1e-3 {
        return Err(Error::invariant("images did not move the logits").into());
    }
    println!("logit_shift={shift:.4}");
    Ok(())
}

#[cfg(cuda_backend)]
struct Case {
    tokens: usize,
    media: usize,
    generated: Vec<u32>,
    text: String,
    logits: Vec<f32>,
}

#[cfg(cuda_backend)]
fn run_case(
    program: &mut DeviceProgram,
    loaded: &LoadedModel,
    assets: &TextAssets,
    processor: &ImageProcessor,
    text: &str,
    kind: u8,
    hidden: usize,
) -> Result<Case, Box<dyn StdError>> {
    use infer_backend_cuda::vision;
    use infer_ir::Modality;
    let imported = loaded.imported();
    let vision = loaded
        .vision()
        .ok_or_else(|| Error::unsupported("no vision tower"))?;
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
    let generated = vision::decode(program, prepared.tokens.len(), &prefill, 8, &[])?;
    let media = prepared.spans.iter().map(Range::len).sum();
    Ok(Case {
        tokens: prepared.tokens.len(),
        media,
        text: assets.decode(&generated, true)?,
        generated,
        logits,
    })
}

#[cfg(cuda_backend)]
fn synthetic_image(kind: u8) -> infer_models::RawImage {
    const HEIGHT: usize = 60;
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
    eprintln!("cuda backend required");
}
