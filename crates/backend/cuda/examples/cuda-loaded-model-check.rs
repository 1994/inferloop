//! Native library loader and independent request state, using an actual model package.
#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use infer_backend_cuda::resident::DeviceProgram;
#[cfg(cuda_backend)]
use std::env::args;
#[cfg(cuda_backend)]
use std::error::Error as StdError;
#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn StdError>> {
    use infer_backend_cuda::{
        device::CudaDevice,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{Error, ModelId};
    use infer_models::{ChatMessage, ChatOptions, TextAssets};
    let root = args()
        .nth(1)
        .ok_or_else(|| Error::invalid("model path is required"))?;
    enable_default()?;
    let assets = TextAssets::open(&root, 4096)?;
    let input = assets.encode_chat(
        &[ChatMessage::new("user", "只输出17乘23的结果。")],
        &ChatOptions {
            enable_thinking: false,
            ..Default::default()
        },
    )?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        &root,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 32,
            ..Default::default()
        },
    )?;
    let mut first = loaded.sequence(input.len() + 16)?;
    let mut second = loaded.sequence(input.len() + 16)?;
    let nodes = loaded.graph().nodes.len();
    drop(loaded); // Captured graphs must retain their shared immutable weights.
    let first_logits = prefill(&mut first, &input)?;
    let second_logits = prefill(&mut second, &input)?;
    if first_logits != second_logits {
        return Err(Error::invariant("independent request prefill mismatch").into());
    }
    let a = generate(
        &mut first,
        &input,
        first_logits,
        &assets
            .generation
            .resolve(&infer_models::SamplingOverrides::default())?
            .sampling,
    )?;
    // Mutating the first request must not mutate the second request's recurrent state.
    let b = generate(
        &mut second,
        &input,
        second_logits,
        &assets
            .generation
            .resolve(&infer_models::SamplingOverrides::default())?
            .sampling,
    )?;
    let text = assets.decode(&a, true)?;
    if a != b || text.trim() != "391" {
        return Err(Error::invariant("library model generation/state isolation mismatch").into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": true, "model": root, "nodes": nodes, "requests": 2,
            "token_ids": a, "text": text, "prefill_width": 32,
            "scope": "library-only loading, shared weight lifetime, independent state and greedy decode; not production BackendProvider acceptance"
        }))?
    );
    Ok(())
}

#[cfg(cuda_backend)]
fn prefill(program: &mut DeviceProgram, tokens: &[u32]) -> infer_core::Result<Vec<f32>> {
    let mut logits = Vec::new();
    for (index, chunk) in tokens.chunks(32).enumerate() {
        let outputs = program.prefill_batch(chunk, index * 32, (index + 1) * 32 >= tokens.len())?;
        logits = outputs
            .into_iter()
            .last()
            .ok_or_else(|| infer_core::Error::invariant("prompt output"))?
            .1;
    }
    Ok(logits)
}

#[cfg(cuda_backend)]
fn generate(
    program: &mut DeviceProgram,
    prompt: &[u32],
    mut logits: Vec<f32>,
    sampling: &infer_ir::Sampling,
) -> infer_core::Result<Vec<u32>> {
    let mut result = Vec::new();
    for i in 0..16 {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
            .map(|(i, _)| u32::try_from(i))
            .transpose()
            .map_err(|e| infer_core::Error::invalid(e.to_string()))?
            .ok_or_else(|| infer_core::Error::invariant("empty logits"))?;
        result.push(token);
        if sampling.is_eos(token) {
            break;
        }
        logits = program
            .step(token, prompt.len() + i, prompt.len() + i, None, true)?
            .1;
    }
    Ok(result)
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA model check requires Linux and an NVIDIA device");
    exit(1);
}
