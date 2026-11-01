//! Resident-model, sequential dataset diagnostic. Warmup and model load are excluded.
use super::{Model, decode, mtp::Mtp, options::Options};
use infer_models::{ChatMessage, ChatOptions, TextAssets};
use serde_json::{Value, json};
use std::error::Error;
use std::fs::read;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    options.validate()?;
    let manifest: Value =
        serde_json::from_slice(&read(options.dataset.as_ref().ok_or("dataset required")?)?)?;
    let records = manifest["samples"].as_array().ok_or("samples required")?;
    if records.is_empty() {
        return Err("empty dataset".into());
    }
    let text = TextAssets::open(&options.model, 4096)?;
    let resolved = text.resolve_chat_generation(&options.sampling())?;
    let mut inputs = Vec::new();
    for record in records {
        inputs.push(text.encode_chat(
            &[ChatMessage::new(
                "user",
                record["prompt"].as_str().ok_or("prompt required")?,
            )],
            &ChatOptions {
                enable_thinking: resolved.enable_thinking,
                ..Default::default()
            },
        )?);
    }
    let capacity =
        inputs.iter().map(Vec::len).max().unwrap_or(0) + options.max_new_tokens + options.mtp;
    let started = Instant::now();
    let mut model = Model::load(&options.model, capacity)?;
    let mut draft = load_draft(options, &model, capacity)?;
    super::prepare_graphs(&mut model, draft.as_mut(), options)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let empty = model.state.clone();
    let draft_empty = draft.as_ref().map(Mtp::checkpoint);
    let mut results = Vec::new();
    let mut suite_started = Instant::now();
    let mut measured_start_unix = 0.0;
    // One complete request warms kernels; state is restored before every request.
    for index in 0..=inputs.len() {
        model.state.clone_from(&empty);
        model.hidden.clear();
        if let (Some(draft), Some(empty)) = (&mut draft, &draft_empty) {
            draft.restore(empty.clone())?;
            draft.model.hidden.clear();
        }
        if index == 1 {
            suite_started = Instant::now();
            measured_start_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        }
        let sample = index.saturating_sub(1);
        let input = &inputs[sample];
        let request_start_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let started = Instant::now();
        let logits = super::prefill::run(&mut model, &mut draft, input, options.prefill_batch)?;
        let prefill_seconds = started.elapsed().as_secs_f64();
        let decoded = decode::decode(
            &mut model,
            draft.as_mut(),
            input,
            logits,
            &resolved.sampling,
            options.max_new_tokens,
            options.mtp,
        )?;
        let seconds = started.elapsed().as_secs_f64();
        if index > 0 {
            results.push(json!({"id": records[sample]["id"], "input_tokens": input,
                "text": text.decode(&decoded.tokens, true)?, "decode": decoded,
                "prefill_seconds": prefill_seconds, "wall_seconds": seconds,
                "started_unix": request_start_unix,
                "finish_reason": if decoded.tokens.last().is_some_and(|t| resolved.sampling.is_eos(*t)) {
                    "stop" } else { "length" }}));
        }
        eprintln!("dataset {index}/{}: {seconds:.3}s", inputs.len());
    }
    let wall_seconds = suite_started.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "framework": if options.device_graph { "rust-resident" } else { "rust-diagnostic" }, "model": options.model, "target": model.device.target().name(),
            "limitation": options.limitation(), "verification": options.verification(),
            "dataset": manifest, "generation": resolved, "mtp_depth": options.mtp,
        "max_new_tokens": options.max_new_tokens, "concurrency": 1, "warmup_requests": 1,
        "mlp_graph": options.mlp_graph,
        "device_graph": options.device_graph, "prefill_batch": options.prefill_batch, "prefill_math": options.prefill_math(),
        "nvfp4_loading_policy": model.package.imported.precision.resolve(|dtype| model.device.target().supports_compute(dtype)).storage,
        "target_kv_cache": if options.fp8_kv { "fp8-e4m3-static-scales" } else { "f32" },
        "tuning": options.tuning,
        "projection_tuning": if options.tuning.is_some() { "file override" } else { "built-in default tiling" },
        "mlp_pdl": options.mlp_pdl,
        "load_seconds": load_seconds, "wall_seconds": wall_seconds, "results": results,
        "measured_start_unix": measured_start_unix,
            "completed": true
        }))?
    );
    Ok(())
}

fn load_draft(
    options: &Options,
    model: &Model,
    capacity: usize,
) -> infer_core::Result<Option<Mtp>> {
    if options.mtp > 0 {
        Ok(Some(Mtp::load(&options.model, model, capacity)?))
    } else {
        Ok(None)
    }
}
