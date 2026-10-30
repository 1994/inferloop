//! Arguments executable responsibilities.
use super::{backend, commands};
use clap::error::ErrorKind;
use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "infer",
    version,
    subcommand_negates_reqs = true,
    subcommand_precedence_over_arg = true,
    arg_required_else_help = true,
    about = "Rust-native inference runtime with Safetensors-backed incremental execution across native backends."
)]
pub struct Cli {
    /// Local model package to serve. Backend and runtime configuration are automatic.
    #[arg(
        value_name = "MODEL_PATH",
        required_unless_present = "package",
        conflicts_with = "package"
    )]
    pub(super) model_path: Option<PathBuf>,
    #[command(flatten)]
    pub(super) serve: commands::ServeOptions,
    #[arg(long, global = true, value_enum, default_value = "auto")]
    pub(super) backend: backend::BackendChoice,
    /// Number of GPU blocks to use, overriding the profiled count; automatic when omitted.
    #[arg(long, global = true)]
    pub(super) num_gpu_blocks_override: Option<usize>,
    /// Token block size for contiguous chunks of tokens.
    #[arg(long, global = true)]
    pub(super) block_size: Option<usize>,
    /// Maximum number of tokens batched together per step.
    #[arg(long, global = true)]
    pub(super) max_num_batched_tokens: Option<usize>,
    /// Total context per sequence, prompt and output together; defaults to the model limit.
    #[arg(long, global = true, value_name = "TOKENS")]
    pub(super) max_model_len: Option<usize>,
    /// Service cap on generated tokens per request; defaults to the total context.
    #[arg(long, global = true, value_name = "TOKENS")]
    pub(super) max_output_tokens: Option<usize>,
    /// Cap on sequences the scheduler runs at once; defaults to the runtime batch limit.
    #[arg(long, global = true, value_name = "SEQUENCES")]
    pub(super) max_num_seqs: Option<usize>,
    /// Name clients address this deployment by; defaults to the model identity.
    #[arg(long, global = true, value_name = "NAME")]
    pub(super) served_model_name: Option<String>,
    /// Bounded temporary weight upload memory, independent of device residency.
    #[arg(long, global = true)]
    pub(super) upload_staging_mib: Option<usize>,
    /// Disable new tuning measurements; reuse validated cached choices or conservative defaults.
    #[arg(long, global = true)]
    pub(super) no_autotune: bool,
    /// Draft tokens per step; 0 disables MTP, 'auto' selects a complete supported head.
    /// With both execution options omitted, CUDA may select a validated fast profile.
    #[arg(long, global = true, value_name = "COUNT|auto")]
    pub(super) num_speculative_tokens: Option<backend::SpeculativeTokens>,
    /// Separate `DFlash2` draft package to speculate with (CUDA); replaces the MTP head.
    #[arg(long, global = true, value_name = "PATH")]
    pub(super) draft_model: Option<PathBuf>,
    /// Target prefill attention profile. Omit both execution options for automatic selection;
    /// an explicit override keeps the other option at canonical/depth0.
    #[arg(long, global = true, value_enum)]
    pub(super) cuda_attention_profile: Option<backend::CudaAttentionProfile>,
    /// Per-prompt multimodal item limits as JSON, as in vLLM (for example
    /// `{"image":1,"video":1}`). All-zero limits serve text only and skip loading the vision
    /// tower; media inputs are then rejected.
    #[arg(long, global = true, value_name = "JSON", value_parser = parse_media_limits)]
    pub(super) limit_mm_per_prompt: Option<infer_models::MediaLimits>,
    /// Fraction of device memory the engine may use, as in vLLM (0 disables the cap).
    #[arg(long, global = true, default_value_t = crate::constants::DEFAULT_GPU_MEMORY_UTILIZATION)]
    pub(super) gpu_memory_utilization: f64,
    #[command(subcommand)]
    pub(super) command: Option<Command>,
}
#[derive(Subcommand)]
pub enum Command {
    /// Serve native token HTTP/SSE and package tokenizer text requests.
    Serve(commands::ServeOptions),
    /// Run canonical requests with a Safetensors package or reference fixture.
    Run(commands::RunOptions),
    /// Exercise Generate, Embed, Rerank and Decision together.
    /// Compare numeric arrays, or verify batch/chunk invariance of the fixture.
    Verify(commands::VerifyOptions),
    /// Replay journal actions, or resume a quiescent execution checkpoint.
    Replay(commands::ReplayOptions),
    /// Export a CPU/control timeline in Chrome Trace / Perfetto JSON format.
    Profile(commands::ProfileOptions),
    /// Measure B3 CPU reference engine throughput and request latency.
    Benchmark(commands::BenchmarkOptions),
    /// Gate a candidate benchmark with correctness and latency constraints.
    Compare(commands::CompareOptions),
    /// Import an HF Qwen configuration and optionally check a weight index budget.
    InspectModel(commands::InspectModelOptions),
    /// Inspect supported execution paths and host environment.
    Doctor,
    /// Local JSON-RPC over stdin/stdout for inspect, explain, replay and control.
    Agent(commands::AgentOptions),
    /// Read shard headers and validate every text-backbone weight binding.
    InspectPackage(commands::InspectPackageOptions),
    /// Encode plain text or package chat template with native Rust assets.
    Tokenize(commands::TokenizeOptions),
}

impl Cli {
    pub fn into_command(self) -> Result<Command, clap::Error> {
        if self.command.is_some()
            && (self.model_path.is_some()
                || self.serve.package.is_some()
                || self.serve.config.is_some()
                || self.serve.image_decoder != commands::ImageDecoder::Auto
                || self.serve.host_memory_mib != 0
                || self.serve.listen != commands::ServeOptions::default_listen())
        {
            return Err(clap::Error::raw(
                ErrorKind::ArgumentConflict,
                "model launch arguments cannot be combined with a subcommand; put subcommand options after its name",
            ));
        }
        Ok(self.command.unwrap_or_else(|| {
            let mut options = self.serve;
            options.package = self.model_path.or(options.package);
            Command::Serve(options)
        }))
    }
}

#[cfg(test)]
#[path = "../tests/unit/arguments.rs"]
mod tests;

/// Parse the vLLM-style per-prompt media limits.
fn parse_media_limits(value: &str) -> Result<infer_models::MediaLimits, String> {
    let limits: BTreeMap<String, u64> =
        serde_json::from_str(value).map_err(|error| format!("modality limits: {error}"))?;
    let mut parsed = infer_models::MediaLimits::default();
    let mut seen = false;
    for (modality, limit) in limits {
        let limit = usize::try_from(limit)
            .map_err(|_| format!("modality limit for `{modality}` is too large"))?;
        match modality.as_str() {
            "image" => parsed.images = limit,
            "video" => parsed.videos = limit,
            _ => return Err(format!("unsupported modality limit `{modality}`")),
        }
        seen = true;
    }
    if !seen {
        return Err("modality limits must not be empty".into());
    }
    parsed.validate().map_err(|error| error.to_string())?;
    Ok(parsed)
}
