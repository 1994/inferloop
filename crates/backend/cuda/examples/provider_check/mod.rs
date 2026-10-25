mod engine;
mod lifecycle;
mod pool;
use cutile::jit_cache::enable_default;
use infer_backend_cuda::{
    device::CudaDevice,
    executor::CudaBackend,
    loading::{LoadOptions, LoadedModel},
    resident::DeviceProgram,
};
use infer_core::{Error, ModelId, ProgramId, RequestId, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, OutputReadout};
use infer_models::{ChatMessage, ChatOptions, TextAssets};
use infer_spi::BackendProvider;
use std::env::args as env_args;
use std::error::Error as StdError;
use std::time::Instant;

pub fn run() -> Result<(), Box<dyn StdError>> {
    let root = env_args()
        .nth(1)
        .ok_or_else(|| Error::invalid("model path required"))?;
    enable_default()?;
    let assets = TextAssets::open(&root, 4096)?;
    let prompt = prompt(&assets)?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        &root,
        ModelId::ONE,
        LoadOptions {
            fp8_kv: env_args().any(|arg| arg == "--fp8-kv").then_some(true),
            prefill_width: 32,
            mtp_depth: mtp_depth(),
            ..Default::default()
        },
    )?;
    let estimate = loaded.sequence_budget(prompt.len() + 16, OutputReadout::Full, true)?;
    // The MTP draft graph, fusion weights and shared tensors must capture on real hardware.
    let draft = loaded.draft(prompt.len() + 16)?;
    if draft.is_none() && mtp_depth() > 0 {
        return Err(Error::invariant("MTP draft program missing").into());
    }
    let profile = loaded.profile().clone();
    let mut backend = pool::backend(loaded)?;
    let program = backend.compile(ProgramId::ONE)?;
    backend.validate_program(backend.model(), &program)?;
    let states = [StateId::ONE, StateId::new(2)?];
    let capacity = prompt.len() + 16;
    backend.reserve_state_for(states[0], capacity, OutputReadout::Full)?;
    backend.reserve_state_for(states[1], capacity, OutputReadout::Logits)?;
    lifecycle::invalid_reservations(&mut backend, states[0], capacity)?;
    let tasks = prefill_tasks(&prompt, states)?;
    lifecycle::invalid_submissions(&mut backend, &program, &tasks)?;
    let outputs = lifecycle::submit(&mut backend, &program, &tasks, prompt.len())?;
    if outputs[0].output.logits != outputs[1].output.logits
        || outputs[0].output.hidden.len() != prompt.len()
        || !outputs[1].output.hidden.is_empty()
    {
        return Err(Error::invariant("CUDA SPI readout/state isolation").into());
    }
    backend.validate_state_ownership(&[
        (states[0], capacity, prompt.len()),
        (states[1], capacity, prompt.len()),
    ])?;
    let baseline = outputs[0].output.logits.clone();
    let draft_prime = prime_draft(draft, &prompt, &outputs[0].output.hidden)?;
    let sampling = sampling(&assets)?;
    let sequences = generate(
        &mut backend,
        &program,
        outputs,
        states,
        prompt.len(),
        &sampling,
    )?;
    let text = assets.decode(&sequences[0], true)?;
    if sequences[0] != sequences[1] || text.trim() != "391" {
        return Err(Error::invariant("CUDA SPI generation").into());
    }
    backend.reset_state(states[0])?;
    let replay = lifecycle::submit(
        &mut backend,
        &program,
        &[ExecutionTask {
            plain_greedy: false,
            request: RequestId::ONE,
            state: states[0],
            tokens: prompt.clone().into(),

            sampling: None,

            media: None,
        }],
        prompt.len(),
    )?;
    if replay[0].output.logits != baseline {
        return Err(Error::invariant("CUDA reset/replay mismatch").into());
    }
    let pool = pool::check(&mut backend, &program, &prompt, &baseline, states, capacity)?;
    let identity = backend.identity().to_owned();
    let capabilities = backend.capabilities();
    engine::run(backend, &prompt, &sampling, &sequences[0])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": true, "model": root, "backend": identity, "capabilities": capabilities,
            "nodes": program.operations.len(), "requests": 2, "token_ids": sequences[0], "text": text,
            "state_admission_bytes": estimate, "prefill_width": 32, "pool": pool,
            "fp8_kv": env_args().any(|arg| arg == "--fp8-kv"),
            "device_profile": profile,
            // Compiled ops are the per-step dispatch census the fusion estimate rests on.
            "operations": program.operations.len(),
            "op_census": op_census(&program),
            "graph_nodes": program.dataflow.nodes.len(),
            "mtp_depth": mtp_depth(), "prompt_tokens": prompt.len(),
            "draft_prime": draft_prime.map(|(steps, micros)| serde_json::json!({"steps": steps, "micros": micros})),
            "pool_budget_pressure": env_args().any(|arg| arg == "--pool-budget-pressure"),
            "checks": ["readout", "state isolation", "cursor rejection", "kernel rejection", "duplicate reservation", "capacity rejection", "completion ownership", "one-shot poll", "reset replay", "release", "Engine two-request scheduling"],
            "scope": "BackendProvider actual-model acceptance with completion polling, greedy functional check; not throughput baseline or continuous batching"
        }))?
    );
    Ok(())
}

/// Two independent prefill submissions: compatibility `Full` readout and incremental `Logits`.
fn prefill_tasks(prompt: &[u32], states: [StateId; 2]) -> Result<Vec<ExecutionTask>, Error> {
    let input: infer_ir::TokenBuffer = prompt.to_vec().into();
    Ok(vec![
        ExecutionTask {
            plain_greedy: false,
            request: RequestId::ONE,
            state: states[0],
            tokens: prompt.to_vec().into(),
            sampling: None,

            media: None,
        },
        ExecutionTask {
            plain_greedy: false,
            request: RequestId::new(2)?,
            state: states[1],
            tokens: ExecutionInput::Prefill {
                span: input.span(0..input.len())?,
                readout: OutputReadout::Logits,
            },
            sampling: None,

            media: None,
        },
    ])
}

fn generate(
    backend: &mut CudaBackend,
    program: &infer_ir::ExecutionProgram,
    outputs: Vec<infer_ir::TaskOutput>,
    states: [StateId; 2],
    prompt_len: usize,
    sampling: &infer_ir::Sampling,
) -> infer_core::Result<Vec<Vec<u32>>> {
    let mut sequences = Vec::new();
    for (index, output) in outputs.into_iter().enumerate() {
        let mut logits = output.output.logits;
        let mut generated = Vec::new();
        for offset in 0..16 {
            let token = lifecycle::argmax(&logits)?;
            generated.push(token);
            if sampling.is_eos(token) {
                break;
            }
            logits = lifecycle::submit(
                backend,
                program,
                &[ExecutionTask {
                    plain_greedy: false,
                    request: RequestId::new(index as u64 + 1)?,
                    state: states[index],
                    tokens: ExecutionInput::Decode {
                        position: prompt_len + offset,
                        token,
                    },

                    sampling: None,

                    media: None,
                }],
                1,
            )?
            .remove(0)
            .output
            .logits;
        }
        sequences.push(generated);
    }
    Ok(sequences)
}

/// Prime the MTP draft over the prompt, mirroring the diagnostic decode timeline.
fn prime_draft(
    draft: Option<DeviceProgram>,
    prompt: &[u32],
    hidden: &[Vec<f32>],
) -> Result<Option<(usize, u128)>, Error> {
    let Some(mut draft) = draft else {
        return Ok(None);
    };
    if hidden.len() != prompt.len() {
        return Err(Error::invariant("MTP priming hidden rows"));
    }
    let started = Instant::now();
    let mut steps = 0;
    for (position, token) in prompt.iter().enumerate().skip(1) {
        draft.step(
            *token,
            position,
            position - 1,
            Some(&hidden[position - 1]),
            false,
        )?;
        steps += 1;
    }
    Ok(Some((steps, started.elapsed().as_micros())))
}

/// Dispatch census by op kind: the fusion opportunity is the count of small per-layer ops.
fn op_census(program: &infer_ir::ExecutionProgram) -> serde_json::Value {
    let nodes = &program.dataflow.nodes;
    let count =
        |kind: fn(&infer_ir::TensorOp) -> bool| nodes.iter().filter(|n| kind(&n.op)).count();
    serde_json::json!({
        "linear": count(|op| matches!(op, infer_ir::TensorOp::Linear)),
        "norm": count(|op| matches!(op, infer_ir::TensorOp::Norm { .. })),
        "gated_norm": count(|op| matches!(op, infer_ir::TensorOp::GatedNorm { .. })),
        "silu": count(|op| matches!(op, infer_ir::TensorOp::Silu)),
        "multiply": count(|op| matches!(op, infer_ir::TensorOp::Multiply)),
        "add": count(|op| matches!(op, infer_ir::TensorOp::Add)),
        "split": count(|op| matches!(op, infer_ir::TensorOp::Split { .. })),
        "embedding": count(|op| matches!(op, infer_ir::TensorOp::Embedding)),
    })
}

/// `--mtp-depth N` selects the speculative draft depth; absent or unparsable means disabled.
fn mtp_depth() -> usize {
    let mut args = env_args();
    while let Some(arg) = args.next() {
        if arg == "--mtp-depth" {
            return args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
        }
    }
    0
}

fn sampling(assets: &TextAssets) -> infer_core::Result<infer_ir::Sampling> {
    let sampling = assets
        .generation
        .resolve(&infer_models::SamplingOverrides::default())?
        .sampling;
    Ok(sampling)
}

fn prompt(assets: &TextAssets) -> infer_core::Result<Vec<u32>> {
    let prompt = assets.encode_chat(
        &[ChatMessage::new("user", "只输出17乘23的结果。")],
        &ChatOptions {
            enable_thinking: false,
            ..Default::default()
        },
    )?;
    Ok(prompt)
}
