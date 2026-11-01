#[cfg(cuda_backend)]
use cutile::jit_cache::enable_default;
#[cfg(cuda_backend)]
use std::error::Error;
#[cfg(cuda_backend)]
mod model_smoke;

#[cfg(cuda_backend)]
fn main() -> Result<(), Box<dyn Error>> {
    use clap::Parser;
    use infer_models::{ChatMessage, ChatOptions, TextAssets};
    use model_smoke::{mtp::Mtp, options::Options};
    use std::time::Instant;
    let options = Options::parse();
    enable_default()?;
    if options.dataset.is_some() {
        return model_smoke::suite::run(&options);
    }
    options.validate()?;
    let text = TextAssets::open(&options.model, 4096)?;
    let resolved = text.resolve_chat_generation(&options.sampling())?;
    let input = text.encode_chat(
        &[ChatMessage::new("user", options.prompt.clone())],
        &ChatOptions {
            enable_thinking: resolved.enable_thinking,
            ..Default::default()
        },
    )?;
    let started = Instant::now();
    let capacity = input.len() + options.max_new_tokens + options.mtp;
    let mut model = model_smoke::Model::load(&options.model, capacity)?;
    let mut mtp = if options.mtp > 0 {
        Some(Mtp::load(&options.model, &model, capacity)?)
    } else {
        None
    };
    if options.tune_projections {
        return model_smoke::tuning::run(&model, mtp.as_ref()).map_err(Into::into);
    }
    model_smoke::prepare_graphs(&mut model, mtp.as_mut(), &options)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let start = Instant::now();
    let logits = model_smoke::prefill::run(&mut model, &mut mtp, &input, options.prefill_batch)?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let result = model_smoke::decode::decode(
        &mut model,
        mtp.as_mut(),
        &input,
        logits,
        &resolved.sampling,
        options.max_new_tokens,
        options.mtp,
    )?;
    let decode_seconds = start.elapsed().as_secs_f64();
    let result = serde_json::json!({
        "mode": options.execution_mode(),
        "limitation": options.limitation(),
        "precision": options.limitation(),
        "mtp_depth": options.mtp, "generation": resolved,
        "mlp_graph": options.mlp_graph,
        "device_graph": options.device_graph, "prefill_batch": options.prefill_batch, "prefill_math": options.prefill_math(),
        "nvfp4_loading_policy": model.package.imported.precision.resolve(|dtype| model.device.target().supports_compute(dtype)).storage,
        "target_kv_cache": if options.fp8_kv { "fp8-e4m3-static-scales" } else { "f32" },
        "tuning": options.tuning,
        "projection_tuning": if options.tuning.is_some() { "file override" } else { "built-in default tiling" },
        "mlp_pdl": options.mlp_pdl,
        "verification": options.verification(),
        "timing_note": "diagnostic wall time includes host work, transfers and first-use JIT; not a production benchmark",
        "target": model.device.target().name(), "model": options.model, "prompt": options.prompt,
        "input_tokens": input, "decode": result, "text": text.decode(&result.tokens, true)?,
        "load_seconds": load_seconds, "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(not(cuda_backend))]
fn main() {
    use std::process::exit;

    eprintln!("CUDA model diagnostics require Linux");
    exit(1);
}
