//! Exercise the same provider through scheduler, incremental input and completion handling.
use infer_backend_cuda::{executor::CudaBackend, registry::CudaKernels};
use infer_core::{Error, ModelId, RequestId, Result};
use infer_ir::{CanonicalRequest, PrecisionPlan, Qos, RequestInput, Sampling, Workload};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use std::{
    collections::BTreeMap,
    thread::yield_now,
    time::{Duration, Instant},
};

/// Wall-clock budget for the bounded runtime acceptance workload.
const ENGINE_TIMEOUT: Duration = Duration::from_secs(30);

pub fn run(
    backend: CudaBackend,
    prompt: &[u32],
    sampling: &Sampling,
    expected: &[u32],
) -> Result<()> {
    let mut registry = KernelRegistry::default();
    registry.register(&CudaKernels)?;
    let model = backend.model().clone();
    let config = RuntimeConfig {
        max_requests: 2,
        max_num_seqs: 2,
        candidate_limit: 2,
        max_request_units: 2,
        max_num_batched_tokens: 128,
        workspace_bytes: 256 * 1024 * 1024,
        ..RuntimeConfig::default()
    };
    let mut engine = Engine::new(backend, model, PrecisionPlan::f32(), &registry, config)?;
    let ids = [RequestId::ONE, RequestId::new(2)?];
    for id in ids {
        engine.submit(CanonicalRequest {
            id,
            model: ModelId::ONE,
            session: None,
            input: RequestInput::Sequence {
                tokens: prompt.to_vec().into(),
                media: vec![],
            },
            workload: Workload::Generate { max_new_tokens: 16 },
            qos: Qos::default(),
            sampling: Sampling {
                temperature: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                ..sampling.clone()
            },
            prefix_boundary: None,
            extensions: BTreeMap::new(),
        })?;
    }
    let start = Instant::now();
    while start.elapsed() < ENGINE_TIMEOUT {
        engine.tick(u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX))?;
        engine.check_invariants()?;
        if engine.is_idle() {
            break;
        }
        yield_now();
    }
    if !engine.is_idle() {
        return Err(Error::invariant("CUDA runtime did not settle"));
    }
    for id in ids {
        let request = engine.request(id)?;
        if request
            .completed
            .as_ref()
            .is_none_or(|done| done.output.is_none())
            || request.generated.as_slice() != expected
        {
            return Err(Error::invariant(format!(
                "CUDA runtime result mismatch: {:?}",
                request.completed
            )));
        }
    }
    Ok(())
}
