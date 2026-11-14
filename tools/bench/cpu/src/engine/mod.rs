mod backend;
mod kernels;
use crate::allocator::{self, Counts};
use infer_core::{CancelSource, Error, ModelId, RequestId, Result};
use infer_ir::{
    BackboneKind, CanonicalRequest, DType, FeedForward, Head, Mixer, Modality, ModelIr,
    PositionSpec, PrecisionPlan, StateKind, StateRequirement,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use serde::Serialize;
#[derive(Serialize)]
pub struct Case {
    requests: usize,
    batch: usize,
    prefill: Counts,
    decode: Counts,
    completion: Counts,
    cancellation: Counts,
}
/// The protocol scene's model: an explicit descriptor with the dimensions the benchmark has
/// always measured, so its allocation profile stays comparable. The backend is a device-contract
/// double that computes nothing, so the kernels below are declarations and no weights are loaded.
fn protocol_model() -> ModelIr {
    // Two attention layers, hidden 8, intermediate 16, vocabulary 32.
    const HIDDEN: usize = 8;
    const LAYERS: usize = 2;
    const KV_ELEMENTS_PER_HEAD: usize = 2;
    ModelIr {
        id: ModelId::ONE,
        backbone: BackboneKind::Decoder,
        vocab_size: 32,
        hidden_size: HIDDEN,
        max_sequence: 32768,
        mixers: vec![
            Mixer::Attention {
                query_heads: 1,
                kv_heads: 1,
                head_dim: HIDDEN,
                sliding_window: None,
                output_gate: false,
                qk_norm: false,
            };
            LAYERS
        ],
        feed_forward: FeedForward::Dense { intermediate: 16 },
        position: PositionSpec {
            rope_theta: 10_000.0,
            rotary_fraction: 1.0,
            multimodal_sections: vec![],
            interleaved: false,
        },
        norm_epsilon: 1e-5,
        norm_weight_offset: 0.0,
        heads: vec![Head::LanguageModel, Head::Embedding],
        modalities: vec![Modality::Text],
        state: (0..LAYERS)
            .map(|layer| StateRequirement {
                layer,
                kind: StateKind::AttentionKv,
                dtype: DType::F32,
                elements: HIDDEN * KV_ELEMENTS_PER_HEAD,
                per_token: true,
            })
            .collect(),
        tied_embeddings: false,
    }
}

fn engine(requests: usize, batch: usize) -> Result<Engine<backend::Backend>> {
    let model = protocol_model();
    let backend = backend::Backend::new(requests, batch, model.vocab_size)?;
    let mut registry = KernelRegistry::default();
    registry.register(&kernels::DeclaredKernels)?;
    Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig {
            max_requests: requests,
            max_num_seqs: batch,
            candidate_limit: requests,
            history_capacity: 32,
            admission: infer_ir::AdmissionConfig {
                default_tenant: infer_ir::TenantQuota {
                    max_reserved_tokens: requests * 32768,
                    max_state_pages: requests * 2048,
                    ..infer_ir::TenantQuota::default()
                },
                ..infer_ir::AdmissionConfig::default()
            },
            max_num_batched_tokens: batch * 64,
            num_gpu_blocks: requests * 2048,
            ..RuntimeConfig::default()
        },
    )
}
fn request(id: usize, tokens: usize, generated: usize, slo: bool) -> Result<CanonicalRequest> {
    let mut request: CanonicalRequest = serde_json::from_value(serde_json::json!({
        "id":id,"model":1,"input":{"Sequence":{"tokens":vec![1; tokens]}},
        "sampling":{"temperature":0.7,"top_k":8,"seed":42},
        "workload":{"Generate":{"max_new_tokens":generated}}
    }))
    .map_err(|error| Error::invalid(error.to_string()))?;
    if slo {
        request.qos.deadline_us = Some(60_000_000);
        request.qos.ttft_slo_us = Some(30_000_000);
        request.qos.tpot_slo_us = Some(1_000);
    }
    Ok(request)
}
fn measured(operation: impl FnOnce() -> Result<()>) -> Result<Counts> {
    allocator::start();
    let result = operation();
    let counts = allocator::stop();
    result?;
    Ok(counts)
}
const fn add(total: &mut Counts, sample: Counts) {
    total.allocations += sample.allocations;
    total.reallocations += sample.reallocations;
    total.deallocations += sample.deallocations;
    total.requested_bytes += sample.requested_bytes;
}
pub fn run(requests: usize, batch: usize) -> Result<Case> {
    run_inner(requests, batch, false)
}
pub fn run_slo(requests: usize, batch: usize) -> Result<Case> {
    run_inner(requests, batch, true)
}
fn run_inner(requests: usize, batch: usize, slo: bool) -> Result<Case> {
    let mut engine = engine(requests, batch)?;
    for id in 1..=requests {
        engine.submit(request(id, 1024, 16384, slo)?)?;
    }
    let mut output = Vec::with_capacity(batch * 2);
    let mut prefill = Counts::default();
    let mut decode = Counts::default();
    allocator::arm_trap();
    for tick in 1..=10000 {
        let generating = !engine.request(RequestId::ONE)?.generated.is_empty();
        let counts = measured(|| engine.tick_into(tick, &mut output))?;
        add(
            if generating {
                &mut decode
            } else {
                &mut prefill
            },
            counts,
        );
        output.clear();
    }
    if engine.request(RequestId::ONE)?.status.terminal() {
        return Err(Error::invariant("cancellation test lost its live request"));
    }
    let mut cancellation = Counts::default();
    for id in 1..=requests {
        add(
            &mut cancellation,
            measured(|| {
                engine.cancel_into(
                    RequestId::new(id as u64)?,
                    CancelSource::Requested,
                    &mut output,
                )
            })?,
        );
    }
    for tick in 10001..11000 {
        add(
            &mut cancellation,
            measured(|| engine.tick_into(tick, &mut output))?,
        );
        output.clear();
        if engine.is_idle() {
            break;
        }
    }
    engine.check_invariants()?;
    if !engine.is_idle() {
        return Err(Error::invariant("live cancellation did not drain"));
    }
    let mut terminal = self::engine(requests, batch)?;
    for id in 1..=requests {
        terminal.submit(request(id, 1, 1, slo)?)?;
    }
    let mut completion = Counts::default();
    for tick in 1..1000 {
        add(
            &mut completion,
            measured(|| terminal.tick_into(tick, &mut output))?,
        );
        output.clear();
        if terminal.is_idle() {
            break;
        }
    }
    terminal.check_invariants()?;
    if !terminal.is_idle() {
        return Err(Error::invariant("completion did not drain"));
    }
    let case = Case {
        requests,
        batch,
        prefill,
        decode,
        completion,
        cancellation,
    };
    for (phase, counts) in [
        ("prefill", &case.prefill),
        ("decode", &case.decode),
        ("completion", &case.completion),
        ("cancellation", &case.cancellation),
    ] {
        if counts.allocations != 0 || counts.reallocations != 0 || counts.deallocations != 0 {
            return Err(Error::invariant(format!(
                "Engine CPU allocation gate failed in {phase} for R={requests}, B={batch}: {} alloc, {} realloc, {} dealloc\n{}",
                counts.allocations,
                counts.reallocations,
                counts.deallocations,
                allocator::trapped_stack().unwrap_or("no trapped stack")
            )));
        }
    }
    Ok(case)
}
