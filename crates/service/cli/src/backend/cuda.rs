use super::{SelectedBackend, Selection};
use crate::constants::{GIB_U64, MIB_U64};
use crate::support::memory::utilization_bytes;
use infer_backend_cuda::device::{CudaDevice, DeviceProfile};
use infer_backend_cuda::target::CudaTarget;
use infer_backend_cuda::tuning::TuningReport;
use infer_backend_cuda::{
    executor::CudaBackend,
    loading::{LoadOptions, LoadedModel},
};
use infer_core::{Error, Result};
use std::path::Path;

use crate::support::human_bytes;
use std::time::Instant;

pub fn load(path: &Path, memory_bytes: u64, selection: &Selection) -> Result<SelectedBackend> {
    let mut automatic_fast = false;
    let result = load_once(path, memory_bytes, selection, &mut automatic_fast);
    if automatic_fast
        && let Err(error) = &result
        && (error.code == infer_core::ErrorCode::Capacity || out_of_memory(&error.message))
    {
        tracing::warn!(target: "infer::load", %error,
            "automatic execution profile did not fit; retrying canonical attention without MTP");
        let recovery = CudaDevice::new(0)?;
        recovery.synchronize()?;
        drop(recovery);
        let conservative = Selection {
            num_speculative_tokens: Some(super::SpeculativeTokens::Depth(0)),
            attention_profile: Some(super::CudaAttentionProfile::Canonical),
            ..selection.clone()
        };
        return load_once(path, memory_bytes, &conservative, &mut automatic_fast);
    }
    result
}

fn load_once(
    path: &Path,
    memory_bytes: u64,
    selection: &Selection,
    automatic_fast: &mut bool,
) -> Result<SelectedBackend> {
    if selection.num_gpu_blocks_override.is_some() || selection.upload_staging_mib.is_some() {
        return Err(Error::unsupported(
            "CUDA physical KV blocks/upload staging overrides are not implemented",
        ));
    }
    let started = Instant::now();
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let (_, total) = device.memory_info()?;
    // A retry may reuse freed blocks still reserved in this context's pool.
    let free = device.allocatable_memory_bytes()?;
    // vLLM semantics: the utilization fraction bounds the whole engine, weights included.
    let share = utilization_bytes(total, selection.gpu_memory_utilization)?;
    let requested = share.min(if memory_bytes == 0 {
        total
    } else {
        memory_bytes
    });
    let limit = requested.min(free.saturating_sub(GIB_U64));
    let budget = LoadBudget { limit, free, total };
    let summary = preflight(
        path,
        device.target(),
        &device.profile()?.architecture,
        limit,
        selection,
    )
    .map_err(|error| load_failure(&error, budget, selection))?;
    *automatic_fast = selection.num_speculative_tokens.is_none()
        && selection.attention_profile.is_none()
        && summary.mtp_depth == 2
        && summary.attention_profile == super::CudaAttentionProfile::Bf16;
    report_environment(device.profile()?, budget, selection, memory_bytes, &summary);
    report_model(path, &summary);
    let loaded = LoadedModel::open(
        device,
        path,
        infer_core::ModelId::ONE,
        LoadOptions {
            prefill_width: selection.max_num_batched_tokens.unwrap_or(0),
            mtp_depth: summary.mtp_depth,
            draft_model: selection.draft_model.clone(),
            autotune: selection.autotune,
            attention_profile: summary.attention_profile.into(),
            vision: !selection.text_only,
            ..LoadOptions::default()
        },
    )
    .map_err(|error| load_failure(&error, budget, selection))?;
    report_tuning(loaded.tuning());
    // Prompt scratch is allocated with the first sequence and shared by all of them.
    let used = free
        .saturating_sub(loaded.device().allocatable_memory_bytes()?)
        .saturating_add(loaded.shared_scratch_bytes()?);
    let Some(state_budget) = limit.checked_sub(used).filter(|bytes| *bytes > 0) else {
        let error = Error::new(
            infer_core::ErrorCode::Capacity,
            format!(
                "weights used {} of the {} budget, leaving no room for request state",
                human_bytes(used),
                human_bytes(limit)
            ),
        );
        return Err(load_failure(&error, budget, selection));
    };
    let device_states = loaded.profile().resident_states();
    let maximum_states = selection
        .max_num_seqs
        .unwrap_or(device_states)
        .min(device_states)
        .min(selection.tuned_pool_width.unwrap_or(usize::MAX));
    tracing::info!(
        target: "infer::load",
        resident = %human_bytes(used),
        state_budget = %human_bytes(state_budget),
        max_resident_states = maximum_states,
        prefill_width = loaded.execution_profile().prefill_width,
        elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        "model loaded"
    );
    let backend = resident_backend(
        loaded,
        state_budget,
        maximum_states,
        *automatic_fast,
        selection.max_num_seqs.is_none(),
        selection.pool_capture,
    )
    .map_err(|error| load_failure(&error, budget, selection))?;
    Ok(SelectedBackend::Cuda(Box::new(backend)))
}

/// Automatic fast execution requires the complete batching resources it was selected for.
fn resident_backend(
    loaded: LoadedModel,
    budget: u64,
    maximum: usize,
    require_pool: bool,
    automatic_concurrency: bool,
    capture: super::PoolCapture,
) -> Result<CudaBackend> {
    let backend = if automatic_concurrency {
        CudaBackend::new_automatic(loaded, budget, maximum)?
    } else {
        CudaBackend::new(loaded, budget, maximum)?
    };
    let backend = match capture {
        super::PoolCapture::AtLoad => backend.with_pool_captured()?,
        super::PoolCapture::OnDemand => backend,
    };
    let profile = backend.execution_profile();
    let serial_roofline = profile
        .concurrency
        .as_ref()
        .is_some_and(|limit| limit.roofline_bound == Some(1));
    // Automatic concurrency can have a complete funded quote pending the first
    // request's actual length. No worker width is reported until capture succeeds.
    let funded_pending = profile
        .concurrency
        .as_ref()
        .is_some_and(|limit| limit.selected > 1);
    if require_pool && profile.pooled_prefill_width.is_none() && !serial_roofline && !funded_pending
    {
        return Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "automatic fast execution could not fund a complete pooled serving plan",
        ));
    }
    Ok(backend)
}

/// Device budget a load was admitted against, reported when the load cannot fit.
#[derive(Debug, Clone, Copy)]
struct LoadBudget {
    /// Usable byte budget the load was admitted against.
    limit: u64,
    /// Free device memory when the load started.
    free: u64,
    /// Total device memory.
    total: u64,
}

/// Report an operator hint for a capacity failure and attach the budget to the error, so the
/// final JSON failure says what bound the load instead of only that it ran out.
fn load_failure(error: &Error, budget: LoadBudget, selection: &Selection) -> Error {
    if error.code == infer_core::ErrorCode::Capacity || out_of_memory(&error.message) {
        tracing::warn!(
            target: "infer::load",
            gpu_memory_utilization = selection.gpu_memory_utilization,
            "hint: the engine keeps the whole payload resident plus request state; free device \
             memory or lower --gpu-memory-utilization"
        );
    }
    Error::new(
        error.code,
        format!(
            "{}; budget was {} of {} total, {} free before the load",
            error.message,
            human_bytes(budget.limit),
            human_bytes(budget.total),
            human_bytes(budget.free)
        ),
    )
}

/// Whether a driver message reports an allocation failure.
fn out_of_memory(message: &str) -> bool {
    message.contains("out of memory") || message.contains("OUT_OF_MEMORY")
}

/// Announce the device, the budget derived from it and the operator's effective options, so the
/// log starts with everything that decided the outcome.
fn report_environment(
    profile: &DeviceProfile,
    budget: LoadBudget,
    selection: &Selection,
    memory_bytes: u64,
    summary: &PackageSummary,
) {
    tracing::info!(
        target: "infer::load",
        gpu = %profile.name,
        architecture = %profile.architecture,
        sms = profile.multiprocessors,
        "cuda device 0: {} total, {} free",
        human_bytes(budget.total),
        human_bytes(budget.free)
    );
    tracing::info!(
        target: "infer::load",
        gpu_memory_utilization = selection.gpu_memory_utilization,
        "device budget {} of {} total after {} reservation",
        human_bytes(budget.limit),
        human_bytes(budget.total),
        human_bytes(GIB_U64)
    );
    tracing::info!(
        target: "infer::load",
        mtp_depth = summary.mtp_depth,
        automatic_mtp = (selection.num_speculative_tokens.is_none() && summary.mtp_depth > 0)
            || selection.num_speculative_tokens == Some(super::SpeculativeTokens::Auto),
        attention_profile = summary.attention_profile.name(),
        autotune = selection.autotune,
        max_num_batched_tokens = %optional(selection.max_num_batched_tokens),
        block_size = %optional(selection.block_size),
        host_memory_mib = %optional_mib(memory_bytes),
        "effective options"
    );
}

/// Announce which model was identified and how much payload it declares.
fn report_model(path: &Path, summary: &PackageSummary) {
    tracing::info!(
        target: "infer::load",
        provider = %summary.provider,
        layers = summary.layers,
        hidden = summary.hidden,
        vocab = summary.vocab,
        max_sequence = summary.max_sequence,
        "model {} identified",
        path.display()
    );
    tracing::info!(
        target: "infer::load",
        weights = summary.tensors,
        mtp_tensors = summary.mtp_tensors,
        encoded = %human_bytes(summary.payload_bytes),
        kv = %summary.kv_cache,
        native_fp4 = summary.native_fp4,
        "checkpoint payload"
    );
}

/// Render an optional operator override, so a log line never leaves a parameter unstated.
fn optional(value: Option<usize>) -> String {
    value.map_or_else(|| "auto".to_owned(), |value| value.to_string())
}

/// Render an optional host-memory budget in MiB.
fn optional_mib(bytes: u64) -> String {
    if bytes == 0 {
        "auto".to_owned()
    } else {
        (bytes / MIB_U64).to_string()
    }
}

fn report_tuning(report: &TuningReport) {
    for decision in &report.measured {
        tracing::info!(
            target: "infer::load",
            tile = %decision.key,
            rows = decision.tiling.rows(),
            columns = decision.tiling.columns(),
            speedup = decision.speedup,
            "autotuned projection tile"
        );
    }
    for failure in &report.fallback {
        tracing::warn!(
            target: "infer::load",
            "kept the conservative tile, measurement failed: {failure}"
        );
    }
    if !report.measured.is_empty() {
        tracing::info!(
            target: "infer::load",
            measured = report.measured.len(),
            cached = report.cached,
            "projection tiles measured"
        );
    }
}

/// Checkpoint facts a load reports before binding a byte: what the package claims to be and how
/// much payload it declares. Returned by [`preflight`] so the identity and the budget check come
/// from one read of the metadata.
struct PackageSummary {
    provider: String,
    layers: usize,
    hidden: usize,
    vocab: usize,
    max_sequence: usize,
    tensors: usize,
    mtp_tensors: usize,
    mtp_depth: usize,
    attention_profile: super::CudaAttentionProfile,
    payload_bytes: u64,
    native_fp4: bool,
    kv_cache: &'static str,
}

fn preflight(
    path: &Path,
    target: &CudaTarget,
    architecture: &str,
    budget: u64,
    selection: &Selection,
) -> Result<PackageSummary> {
    use infer_models::{QuantizedPackage, WeightEncoding};
    let package = QuantizedPackage::open(path, infer_core::ModelId::ONE)?;
    // The checkpoint's provider declares the precision preference; the device answers what it can
    // compute. A decoded fallback costs two bytes per element instead of the packed storage.
    let native_fp4 = package
        .imported
        .precision
        .resolve(|dtype| target.supports_compute(dtype))
        .storage
        == infer_ir::DType::Fp4E2M1;
    let mut minimum = 0_u64;
    for weight in package.weights.values() {
        let bytes = if weight.encoding == WeightEncoding::Nvfp4 && !native_fp4 {
            weight
                .shape
                .iter()
                .try_fold(2_u64, |n, d| n.checked_mul(*d as u64))
                .ok_or_else(|| Error::invalid("CUDA weight budget overflow"))?
        } else {
            weight.data.bytes
        };
        minimum = minimum
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("CUDA weight budget overflow"))?;
    }
    if minimum >= budget {
        return Err(Error::new(
            infer_core::ErrorCode::Capacity,
            format!(
                "the checkpoint payload needs at least {} but the device budget is {}",
                human_bytes(minimum),
                human_bytes(budget)
            ),
        ));
    }
    // The automatic BF16 attention profile is admitted only with its MTP depth: under a DFlash2
    // draft it measured 441/512 on GSM8K against 474 for canonical attention.
    let fast_supported = selection.draft_model.is_none()
        && native_fp4
        && package.kv_cache_dtype == Some(infer_models::TensorDtype::F8E4m3)
        && super::speculation::resolve(
            super::SpeculativeTokens::Auto,
            selection.max_num_batched_tokens,
            &package,
        ) == 2
        && infer_backend_cuda::AttentionProfile::supports_bf16_prefill(
            architecture,
            &package.graph,
            selection.max_num_batched_tokens.unwrap_or(0),
        );
    let (requested_depth, attention_profile) = super::defaults::resolve(
        selection.num_speculative_tokens,
        selection.attention_profile,
        fast_supported,
    );
    Ok(PackageSummary {
        provider: package.provider.metadata().name,
        layers: package.imported.model.mixers.len(),
        hidden: package.imported.model.hidden_size,
        vocab: package.imported.model.vocab_size,
        max_sequence: package.imported.model.max_sequence,
        tensors: package.weights.len(),
        mtp_tensors: package.mtp.len(),
        // A separate draft replaces the checkpoint's own MTP head.
        mtp_depth: if selection.draft_model.is_some() {
            0
        } else {
            super::speculation::resolve(requested_depth, selection.max_num_batched_tokens, &package)
        },
        attention_profile,
        payload_bytes: minimum,
        native_fp4,
        kv_cache: kv_cache_label(&package),
    })
}

/// KV storage the checkpoint declares, in operator terms.
const fn kv_cache_label(package: &infer_models::QuantizedPackage) -> &'static str {
    match package.kv_cache_dtype {
        Some(infer_models::TensorDtype::F8E4m3) => "fp8",
        Some(_) => "declared",
        None => "model default",
    }
}
