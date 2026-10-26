use super::lifecycle;
use infer_backend_cuda::executor::CudaBackend;
use infer_backend_cuda::loading::LoadedModel;
use infer_core::{Error, RequestId, Result, StateId};
use infer_ir::{ExecutionInput, ExecutionProgram, ExecutionTask, OutputReadout, TokenBuffer};
use infer_spi::BackendProvider;
use std::env::args;
use std::time::Instant;

pub fn check(
    backend: &mut CudaBackend,
    program: &ExecutionProgram,
    prompt: &[u32],
    logits: &[f32],
    states: [StateId; 2],
    capacity: usize,
) -> Result<serde_json::Value> {
    // A fresh program's prefill width is the reference a pooled reuse must keep.
    let fresh_width = backend.prefill_width_for(states[0])?;
    for state in states {
        backend.release_state(state)?;
    }
    backend.validate_state_ownership(&[])?;
    let before = backend.pool_inspection();
    if before.cached_sequences != 2 {
        return Err(Error::invariant("released graphs were not retained"));
    }
    let id = StateId::new(3)?;
    // Same physical bucket, different logical capacity: reject the logical overflow.
    backend.reserve_state_for(id, capacity + 1, OutputReadout::Logits)?;
    backend.validate_state_ownership(&[(id, capacity + 1, 0)])?;
    if lifecycle::submit(
        backend,
        program,
        &[ExecutionTask {
            plain_greedy: false,
            request: RequestId::ONE,
            state: id,
            tokens: vec![0; capacity + 2].into(),

            sampling: None,

            media: None,
        }],
        capacity + 2,
    )
    .is_ok()
    {
        return Err(Error::invariant("pooled state lost logical capacity limit"));
    }
    if backend.prefill_width_for(id)? != fresh_width {
        return Err(Error::invariant("pooled state lost its wide prefill path"));
    }
    verify(backend, program, id, prompt, logits)?;
    backend.release_state(id)?;
    let reused = backend.pool_inspection();
    if reused.sequence_allocations != before.sequence_allocations
        || reused.sequence_reuses != before.sequence_reuses + 1
    {
        return Err(Error::invariant("matching pooled graph was not reused"));
    }
    // A third capacity bucket must evict idle state under the two-sequence limit.
    backend.reserve_state_for(id, 128, OutputReadout::Logits)?;
    backend.release_state(id)?;
    let evicted = backend.pool_inspection();
    if evicted.cached_sequences > 2 || evicted.sequence_evictions <= reused.sequence_evictions {
        return Err(Error::invariant(
            "state pool did not enforce its count bound",
        ));
    }
    backend.trim_state_pool()?;
    if backend.pool_inspection().cached_sequences != 0 {
        return Err(Error::invariant("CUDA pool trim"));
    }
    let cold = Instant::now();
    backend.reserve_state_for(id, capacity, OutputReadout::Logits)?;
    let cold_us = cold.elapsed().as_micros();
    verify(backend, program, id, prompt, logits)?;
    backend.release_state(id)?;
    let mut warm_us = Vec::new();
    for _ in 0..3 {
        let started = Instant::now();
        backend.reserve_state_for(id, capacity, OutputReadout::Logits)?;
        warm_us.push(started.elapsed().as_micros());
        verify(backend, program, id, prompt, logits)?;
        backend.release_state(id)?;
    }
    let large_capacity = long_capacity(backend, program, prompt, logits)?;
    backend.validate_state_ownership(&[])?;
    Ok(
        serde_json::json!({"passed":true, "inspection":backend.pool_inspection(),
        "uncached_reservation_us":cold_us, "cached_reservation_us":warm_us,
        "scope":"same-capacity graph/state preparation wall time; not inference throughput",
        "large_capacity": large_capacity}),
    )
}
fn verify(
    backend: &mut CudaBackend,
    program: &ExecutionProgram,
    state: StateId,
    prompt: &[u32],
    logits: &[f32],
) -> Result<()> {
    let tokens: TokenBuffer = prompt.to_vec().into();
    let output = lifecycle::submit(
        backend,
        program,
        &[ExecutionTask {
            plain_greedy: false,
            request: RequestId::ONE,
            state,
            tokens: ExecutionInput::Prefill {
                span: tokens.span(0..tokens.len())?,
                readout: OutputReadout::Logits,
            },

            sampling: None,

            media: None,
        }],
        prompt.len(),
    )?;
    if output[0].output.logits != logits || !output[0].output.hidden.is_empty() {
        return Err(Error::invariant("pooled CUDA graph reset/readout mismatch"));
    }
    Ok(())
}

pub fn backend(loaded: LoadedModel) -> Result<CudaBackend> {
    let pressure = args().any(|arg| arg == "--pool-budget-pressure");
    CudaBackend::new(
        loaded,
        if pressure { 6 } else { 8 } * 1024 * 1024 * 1024,
        if pressure { 3 } else { 2 },
    )
}

fn long_capacity(
    backend: &mut CudaBackend,
    program: &ExecutionProgram,
    prompt: &[u32],
    logits: &[f32],
) -> Result<serde_json::Value> {
    if !args().any(|arg| arg == "--large-capacity") {
        return Ok(serde_json::Value::Null);
    }
    backend.trim_state_pool()?;
    let state = StateId::new(5)?;
    let capacity = 8192 + 128;
    let required = backend.state_reservation_bytes_for(capacity, OutputReadout::Logits)?;
    backend.reserve_state_for(state, capacity, OutputReadout::Logits)?;
    verify(backend, program, state, prompt, logits)?;
    backend.release_state(state)?;
    backend.trim_state_pool()?;
    Ok(
        serde_json::json!({"capacity":capacity,"physical_capacity":capacity.next_power_of_two(),
        "admission_bytes":required,"short_prefix_logits_match":true,
        "scope":"large allocation and short-prefix correctness; not an 8k-token benchmark"}),
    )
}
