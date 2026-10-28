//! Dispatch executable responsibilities.
#[cfg(native_backend)]
use crate::support::agent;
use crate::support::flush_logs;
use crate::support::init_logging;
use crate::{arguments::Cli, arguments::Command, backend, commands, support::print};
use clap::Parser;
use infer_core::Result;
use std::env::consts::ARCH;
use std::env::consts::OS;
use std::io;
use std::process::exit;

pub fn run() {
    init_logging();
    let cli = Cli::parse();
    let selection = backend::Selection {
        kind: cli.backend,
        num_gpu_blocks_override: cli.num_gpu_blocks_override,
        block_size: cli.block_size,
        max_num_batched_tokens: cli.max_num_batched_tokens,
        max_model_len: cli.max_model_len,
        max_output_tokens: cli.max_output_tokens,
        max_num_seqs: cli.max_num_seqs,
        served_model_name: cli.served_model_name.clone(),
        upload_staging_mib: cli.upload_staging_mib,
        num_speculative_tokens: cli.num_speculative_tokens,
        draft_model: cli.draft_model.clone(),
        #[cfg(cuda_backend)]
        attention_profile: cli.cuda_attention_profile,
        gpu_memory_utilization: cli.gpu_memory_utilization,
        autotune: !cli.no_autotune,
        text_only: cli.limit_mm_per_prompt.unwrap_or_default().text_only(),
        media_limits: cli.limit_mm_per_prompt.unwrap_or_default(),
    };
    if let Err(error) = execute(
        cli.into_command().unwrap_or_else(|error| error.exit()),
        selection,
    ) {
        let _ = serde_json::to_writer(io::stderr().lock(), &error);
        eprintln!();
        flush_logs();
        exit(1);
    }
}

// Host-only builds route the selection into subcommands that take it by value but never
// consume it, which the ownership lint flags there and only there.
#[cfg_attr(
    not(native_backend),
    expect(
        clippy::needless_pass_by_value,
        reason = "The dispatcher owns the single request-local selection in host-only builds"
    )
)]
pub fn execute(command: Command, backend_choice: backend::Selection) -> Result<()> {
    match command {
        #[cfg(not(native_backend))]
        Command::Run(options) if options.package.is_none() => Err(infer_core::Error::invalid(
            "a device backend requires --package",
        )),
        #[cfg(native_backend)]
        Command::Serve(options) => commands::serve(options, backend_choice),
        #[cfg(native_backend)]
        Command::Run(options) => commands::run(options, backend_choice),
        Command::Verify(options) => commands::verify(options, &backend_choice),
        #[cfg(native_backend)]
        Command::Replay(options) => commands::replay(options, backend_choice),
        #[cfg(native_backend)]
        Command::Profile(options) => commands::profile(options, backend_choice),
        #[cfg(native_backend)]
        Command::Benchmark(options) => commands::benchmark(options, backend_choice),
        Command::Compare(options) => commands::compare(options),
        Command::InspectModel(options) => commands::inspect_model(options),
        Command::InspectPackage(options) => commands::inspect_package(options),
        Command::Tokenize(options) => commands::tokenize(options),
        Command::Doctor => print(
            &serde_json::json!({"host_arch":ARCH,"host_os":OS,"backends":backend::catalog(),"qwen38_config_import":true,"safetensors_weight_binding":true,"native_tokenizer":true,"incremental_hybrid_state":true,"cuda_execution":"resident CUDA BackendProvider; requires Linux --features cuda","qwen38_27b_execution_validated":false,"native_workloads":["generate","embed","rerank","decision"]}),
        ),
        #[cfg(native_backend)]
        Command::Agent(options) => agent(
            options.package.as_deref(),
            options.host_memory_mib,
            options.probe_memory_mib,
            backend_choice,
        ),
        #[cfg(not(native_backend))]
        _ => Err(infer_core::Error::unsupported(match backend_choice.kind {
            backend::BackendChoice::Cuda => {
                "CUDA execution requires Linux and a build with --features cuda"
            }
            backend::BackendChoice::Metal => {
                "Metal backend requires macOS and a supported Metal device"
            }
            backend::BackendChoice::Auto => {
                "no supported GPU backend available (CUDA or Metal); CPU execution is test-only"
            }
        })),
    }
}
