//! Validated runtime capacity and planning limits.
use crate::constants::MAX_BOUNDED_CAPACITY;
use infer_core::{Error, Result};
use infer_ir::{AdmissionConfig, CostModelConfig, SchedulerConfig};
use infer_scheduler::{CalibratedCosts, ResourceAdmission};
use serde::{Deserialize, Serialize};

/// Default admission slots retained for queued requests.
const DEFAULT_MAX_REQUESTS: usize = 256;
/// Default number of scheduler candidates scanned per admission pass.
const DEFAULT_CANDIDATE_LIMIT: usize = 256;
/// Default request-unit budget for one submission.
const DEFAULT_MAX_REQUEST_UNITS: usize = 256;
/// Unset prompt cap: the resolved model/service context bounds actual requests.
const DEFAULT_MAX_INPUT_TOKENS: usize = usize::MAX;
/// Default number of KV state pages pinned by the runtime.
const DEFAULT_STATE_PAGES: usize = 4096;
/// Default number of KV tokens addressed by one state page.
const DEFAULT_PAGE_TOKENS: usize = 16;
/// Default number of completion cells published in one batch.
const DEFAULT_MAX_BATCH: usize = 16;
/// Default decode token budget charged to one scheduling round.
const DEFAULT_TOKEN_BUDGET: usize = 64;
/// Default GPU compute budget per scheduling round, in microseconds.
const DEFAULT_GPU_BUDGET_US: u64 = 10_000;
/// Default workspace reservation, in bytes (1 MiB).
const DEFAULT_WORKSPACE_BYTES: u64 = 1 << 20;
/// Default number of in-flight program progress records retained.
const DEFAULT_PROGRESS_LIMIT: usize = 3;
/// Default submission timeout, in microseconds.
const DEFAULT_SUBMISSION_TIMEOUT_US: u64 = 30_000_000;
/// Default backend resource-operation timeout, in microseconds.
const DEFAULT_RESOURCE_TIMEOUT_US: u64 = 250_000;
/// Default client output timeout, in microseconds.
const DEFAULT_OUTPUT_TIMEOUT_US: u64 = 5_000_000;
/// Queueing has no implicit deadline; request SLOs and explicit service limits still apply.
/// Submission, resource acknowledgement and output stalls keep their separate timeouts.
const DEFAULT_MAX_QUEUE_WAIT_US: u64 = u64::MAX;
/// Default number of decision/history records retained for replay.
const DEFAULT_HISTORY_CAPACITY: usize = 4096;
/// Default token budget retained for replay history.
const DEFAULT_MAX_HISTORY_TOKENS: usize = 1 << 20;
/// Default byte budget retained for replay history (16 MiB).
const DEFAULT_MAX_HISTORY_BYTES: usize = 16 << 20;
/// Default number of event records retained for observation.
const DEFAULT_EVENT_CAPACITY: usize = 8192;
/// Default host transfer budget per scheduling round, in microseconds.
const DEFAULT_TRANSFER_BUDGET_US: u64 = 10_000;
/// Default backend encoder budget per scheduling round, in microseconds.
const DEFAULT_ENCODER_BUDGET_US: u64 = 10_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub max_requests: usize,
    pub cpu: crate::CpuRuntimeConfig,
    pub candidate_limit: usize,
    pub max_request_units: usize,
    /// Operator-requested single-sequence total context (`--max-model-len`), prompt and output
    /// together. `None` follows the model's supported ceiling; the effective value is resolved by
    /// [`crate::ResolvedLengthLimits`], which also rejects a request the model cannot fulfil.
    #[serde(default)]
    pub max_model_len: Option<usize>,
    /// Operator-requested service cap on generated tokens (`--max-output-tokens`). `None` follows
    /// the effective total context.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    /// Optional service prompt cap. The default follows the resolved model context.
    pub max_input_tokens: usize,
    pub num_gpu_blocks: usize,
    pub block_size: usize,
    pub max_num_seqs: usize,
    /// True when the caller set `max_num_seqs`; automatic configurations may use a backend recommendation.
    #[serde(default)]
    pub max_num_seqs_explicit: bool,
    /// Name clients address this deployment by (`--served-model-name`). `None` uses the model
    /// identity, which is what `/v1/models` and the request `model` field then report.
    #[serde(default)]
    pub served_model_name: Option<String>,
    pub max_num_batched_tokens: usize,
    pub gpu_budget_us: u64,
    pub workspace_bytes: u64,
    pub cost_per_token_us: u64,
    pub progress_limit: usize,
    pub submission_timeout_us: u64,
    pub resource_timeout_us: u64,
    pub output_timeout_us: u64,
    pub max_queue_wait_us: u64,
    pub history_capacity: usize,
    pub max_history_tokens: usize,
    pub max_history_bytes: usize,
    pub event_capacity: usize,
    pub scheduler: SchedulerConfig,
    pub cost_model: CostModelConfig,
    pub admission: AdmissionConfig,
    pub transfer_budget_us: u64,
    pub encoder_budget_us: u64,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_requests: DEFAULT_MAX_REQUESTS,
            cpu: crate::CpuRuntimeConfig::default(),
            candidate_limit: DEFAULT_CANDIDATE_LIMIT,
            max_request_units: DEFAULT_MAX_REQUEST_UNITS,
            max_model_len: None,
            max_output_tokens: None,
            max_input_tokens: DEFAULT_MAX_INPUT_TOKENS,
            num_gpu_blocks: DEFAULT_STATE_PAGES,
            block_size: DEFAULT_PAGE_TOKENS,
            max_num_seqs: DEFAULT_MAX_BATCH,
            max_num_seqs_explicit: false,
            served_model_name: None,
            max_num_batched_tokens: DEFAULT_TOKEN_BUDGET,
            gpu_budget_us: DEFAULT_GPU_BUDGET_US,
            workspace_bytes: DEFAULT_WORKSPACE_BYTES,
            cost_per_token_us: 1,
            progress_limit: DEFAULT_PROGRESS_LIMIT,
            submission_timeout_us: DEFAULT_SUBMISSION_TIMEOUT_US,
            resource_timeout_us: DEFAULT_RESOURCE_TIMEOUT_US,
            output_timeout_us: DEFAULT_OUTPUT_TIMEOUT_US,
            max_queue_wait_us: DEFAULT_MAX_QUEUE_WAIT_US,
            history_capacity: DEFAULT_HISTORY_CAPACITY,
            max_history_tokens: DEFAULT_MAX_HISTORY_TOKENS,
            max_history_bytes: DEFAULT_MAX_HISTORY_BYTES,
            event_capacity: DEFAULT_EVENT_CAPACITY,
            scheduler: SchedulerConfig::default(),
            cost_model: CostModelConfig::default(),
            admission: AdmissionConfig::default(),
            transfer_budget_us: DEFAULT_TRANSFER_BUDGET_US,
            encoder_budget_us: DEFAULT_ENCODER_BUDGET_US,
        }
    }
}
impl RuntimeConfig {
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.event_capacity > MAX_BOUNDED_CAPACITY
            || self.history_capacity > MAX_BOUNDED_CAPACITY
        {
            return Err(Error::invalid(
                "event/history capacities must not exceed 1048576 records",
            ));
        }
        if [
            self.max_requests,
            self.candidate_limit,
            self.max_request_units,
            self.max_input_tokens,
            self.num_gpu_blocks,
            self.block_size,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.progress_limit,
            self.history_capacity,
            self.max_history_tokens,
            self.max_history_bytes,
            self.event_capacity,
        ]
        .contains(&0)
            || self.gpu_budget_us == 0
            || self.cost_per_token_us == 0
            || self.max_queue_wait_us == 0
            || self.submission_timeout_us == 0
            || self.resource_timeout_us == 0
            || self.output_timeout_us == 0
        {
            return Err(Error::invalid(
                "runtime capacities/timeouts must be positive",
            ));
        }
        if self.max_model_len == Some(0) || self.max_output_tokens == Some(0) {
            return Err(Error::invalid(
                "max-model-len and max-output-tokens must be positive when set",
            ));
        }
        if self
            .served_model_name
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(Error::invalid(
                "served-model-name must not be empty when set",
            ));
        }
        if self.max_num_seqs > infer_gpu_api::MAX_SUBMISSION_BATCH {
            return Err(Error::invalid("runtime batch exceeds 64 completion cells"));
        }
        if self.cost_per_token_us > self.scheduler.max_singleton_gpu_us {
            return Err(Error::invalid("atomic dispatch limit cannot fit one token"));
        }
        self.cpu.fixed_bytes(self)?;
        infer_scheduler::validate_scheduler(&self.scheduler)?;
        ResourceAdmission::new(self.admission.clone())?;
        CalibratedCosts::new("validation".into(), self.cost_model.clone())?;
        Ok(())
    }
}
