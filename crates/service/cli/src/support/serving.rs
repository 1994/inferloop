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
/// every pooled graph width. The serving process reuses cached tuning and never searches, and
/// a 27B hybrid model on one RTX 5090 captured in 34 to 52 s, past the 30 s submission bound;
/// this leaves about twice that. A reservation whose owner never answers still fails.
const CUDA_RESOURCE_TIMEOUT_US: u64 = 120_000_000;

/// Tokens of one sequence in the backend's multi-sequence prompt graph. An automatic round
/// funds one for every sequence, so a burst of short prompts enters together and shares one
/// weight pass; a model whose chunk is already that wide keeps its one-chunk round.
const CUDA_PROMPT_SEGMENT_TOKENS: usize = 64;

/// What a program's workspace is checked against: the device's state capacity. What a pool already
/// captured is not a limit on the program, or a pool funded to its envelope would refuse to start.
fn program_workspace_limit(capacity: Option<u64>, free: Option<u64>, memory: u64) -> u64 {
    capacity.or(free).unwrap_or(memory)
}

pub fn derive_config(
    config: &mut RuntimeConfig,
    backend: &impl BackendProvider,
    model: &ModelIr,
    registry: &KernelRegistry,
) -> Result<()> {
    config.resolve_concurrency(backend.execution_profile().as_ref())?;
    let capabilities = backend.capabilities();
    let budget = program_workspace_limit(
        backend.state_capacity_bytes()?,
        backend.free_state_bytes()?,
        capabilities.memory_bytes,
    );
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
    let minimum = if automatic {
        complete_round.max(
            config
                .max_num_seqs
                .saturating_mul(CUDA_PROMPT_SEGMENT_TOKENS)
                .saturating_add(decode_slots),
        )
    } else {
        complete_round
    };
    config.max_num_batched_tokens = config.max_num_batched_tokens.max(minimum);
}

#[cfg(test)]
#[path = "../../tests/unit/support_serving.rs"]
mod tests;
