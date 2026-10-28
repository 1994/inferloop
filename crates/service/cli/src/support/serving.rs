//! Derive cold-start service limits from the loaded graph and backend contracts.
use infer_core::{ProgramId, Result};
use infer_ir::{ModelIr, PrecisionPlan};
use infer_kernel_api::KernelRegistry;
use infer_runtime::RuntimeConfig;
use infer_spi::BackendProvider;

/// Fallback logical CUDA prefill chunk when the backend does not report graph geometry.
const CUDA_PREFILL_CHUNK_TOKENS: usize = 64;

/// Step GPU-time budget for CUDA serving: wide enough that the scheduler packs every ready
/// decode and a prompt chunk, and leaves `max_num_batched_tokens` to bound a step. The
/// calibrated per-token cost mixes decode and prompt steps, so it prices a 2048-token chunk
/// at tens of seconds; any tighter budget trims or defers prompt work.
const CUDA_STEP_BUDGET_US: u64 = 60_000_000;

/// Acknowledgement bound for a CUDA resource command. The serving pool is captured for actual
/// request lengths inside the reservation that first needs it, so that reservation waits for
/// every pooled graph width plus any tuning the capture plans. With warm tuning caches a 27B
/// hybrid model on one RTX 5090 took 34 s, past the 30 s submission bound, and cold searches
/// add their own finite allowances. The bound still fails a reservation whose owner never
/// answers; it only stops a capture in progress from failing the request it serves.
const CUDA_RESOURCE_TIMEOUT_US: u64 = 600_000_000;

pub fn derive_config(
    config: &mut RuntimeConfig,
    backend: &impl BackendProvider,
    model: &ModelIr,
    registry: &KernelRegistry,
) -> Result<()> {
    config.resolve_concurrency(backend.execution_profile().as_ref())?;
    let capabilities = backend.capabilities();
    let budget = backend
        .free_state_bytes()?
        .unwrap_or(capabilities.memory_bytes);
    let program = infer_compiler::compile(
        ProgramId::ONE,
        infer_compiler::lower(model, backend.execution_graph(model)?, PrecisionPlan::f32())?,
        registry,
        &capabilities,
        budget,
    )?;
    config.workspace_bytes = program.workspace_bytes.max(1);
    if capabilities.backend_kind() == infer_ir::BackendKind::Cuda {
        // Preserve the private program's default chunk width. A smaller pooled worker
        // splits it internally; reporting that smaller width globally would make long
        // private graphs replay half-filled.
        let profile = backend.execution_profile();
        let width = profile
            .as_ref()
            .map_or(CUDA_PREFILL_CHUNK_TOKENS, |profile| profile.prefill_width)
            .max(1);
        let automatic = profile
            .as_ref()
            .is_some_and(|profile| profile.automatic_prefill);
        cuda_prefill_limits(config, width, automatic);
        // One decode step of a large model already exceeds the generic 10 ms step budget, and
        // the cost model prices an unseen batch shape linearly, so a second decode was always
        // deferred and batched shapes were never observed: concurrent requests ran one at a
        // time. Batched decode is bandwidth-bound, so the token budget is the real limit.
        config.gpu_budget_us = config.gpu_budget_us.max(CUDA_STEP_BUDGET_US);
    }
    // Cold graph capture can take as long as a device submission. The generic resource
    // default is intended for already-warm pools and is too short for JIT-backed loading.
    config.resource_timeout_us = config.submission_timeout_us;
    if capabilities.backend_kind() == infer_ir::BackendKind::Cuda {
        config.resource_timeout_us = config.resource_timeout_us.max(CUDA_RESOURCE_TIMEOUT_US);
    }
    if let Some(cache) = backend.kv_cache() {
        config.num_gpu_blocks = cache.total_blocks.max(1);
        // The engine's logical pages mirror the backend's physical blocks one for one; keeping
        // a different page size would silently re-denominate every page-based limit.
        config.block_size = cache.block_size.max(1);
    }
    config.validate()
}

/// Keep one fixed-width prompt replay whole beside the round's ready decode requests.
/// Without this headroom a single decode trims a full chunk by one token and creates
/// another padded prompt replay for the remainder. Explicit token limits are retained.
fn cuda_prefill_limits(config: &mut RuntimeConfig, width: usize, automatic: bool) {
    config.scheduler.prefill_chunk_tokens = config.scheduler.prefill_chunk_tokens.max(width);
    let decode_slots = if automatic {
        config.max_num_seqs.saturating_sub(1)
    } else {
        0
    };
    let complete_round = config
        .scheduler
        .prefill_chunk_tokens
        .saturating_add(decode_slots);
    config.max_num_batched_tokens = config.max_num_batched_tokens.max(complete_round);
}

#[cfg(test)]
#[path = "../../tests/unit/support_serving.rs"]
mod tests;
