//! Options commands.
use crate::constants::DEFAULT_LISTEN_PORT;
use infer_frontdoor::ImageProcessingOptions;
use infer_frontdoor::cpu::CpuConfig;
use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::path::PathBuf;

/// Bytes in one mebibyte.
const BYTES_PER_MIB: usize = 1 << 20;
/// Microseconds in one millisecond.
const MICROS_PER_MILLI: u64 = 1_000;

#[derive(clap::Args)]
pub struct ServeOptions {
    #[arg(long, default_value_t = Self::default_listen())]
    pub listen: SocketAddr,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, default_value_t = 0, hide = true)]
    pub host_memory_mib: u64,
    /// CPU preparation admission budget in MiB; does not limit process memory.
    #[arg(long = "cpu-staging-mib", value_name = "MIB", value_parser = parse_cpu_staging_mib)]
    pub cpu_staging_bytes: Option<usize>,
    /// Default image resize options as a JSON object; requests may override individual keys.
    #[arg(long = "mm-processor-kwargs", value_name = "JSON", value_parser = parse_image_options)]
    pub image_options: Option<ImageProcessingOptions>,
}
impl ServeOptions {
    pub const fn default_listen() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), DEFAULT_LISTEN_PORT)
    }

    pub(crate) fn cpu_config(staging_bytes: Option<usize>, resource_timeout_us: u64) -> CpuConfig {
        let mut config = CpuConfig {
            device_control_timeout_ms: resource_timeout_us.div_ceil(MICROS_PER_MILLI),
            ..Default::default()
        };
        if let Some(bytes) = staging_bytes {
            config.max_bytes = bytes;
        }
        config
    }
}

fn parse_cpu_staging_mib(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|mib| *mib > 0)
        .and_then(|mib| mib.checked_mul(BYTES_PER_MIB))
        .ok_or_else(|| "CPU staging MiB must be positive and fit in bytes".to_owned())
}

fn parse_image_options(value: &str) -> Result<ImageProcessingOptions, String> {
    ImageProcessingOptions::parse(value).map_err(|error| error.to_string())
}
#[derive(clap::Args)]
pub struct RunOptions {
    #[arg(long)]
    pub requests: PathBuf,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, conflicts_with = "model")]
    pub package: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub host_memory_mib: u64,
    #[arg(long)]
    pub op_trace: Option<PathBuf>,
    #[arg(long)]
    pub journal: Option<PathBuf>,
    #[arg(long)]
    pub snapshot: Option<PathBuf>,
    #[arg(long)]
    pub events: Option<PathBuf>,
}
#[derive(clap::Args)]
pub struct VerifyOptions {
    #[arg(long, requires = "candidate")]
    pub reference: Option<PathBuf>,
    #[arg(long, requires = "reference")]
    pub candidate: Option<PathBuf>,
    #[arg(long, default_value_t = crate::constants::DEFAULT_ATOL)]
    pub atol: f64,
    #[arg(long, default_value_t = crate::constants::DEFAULT_RTOL)]
    pub rtol: f64,
    #[arg(long, conflicts_with = "reference", requires = "golden")]
    pub package: Option<PathBuf>,
    #[arg(long, requires = "package")]
    pub golden: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub host_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct ReplayOptions {
    #[arg(
        long,
        conflicts_with = "snapshot",
        required_unless_present = "snapshot"
    )]
    pub journal: Option<PathBuf>,
    #[arg(long, conflicts_with = "journal")]
    pub snapshot: Option<PathBuf>,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, conflicts_with = "model")]
    pub package: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub host_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct ProfileOptions {
    #[arg(long)]
    pub journal: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub host_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct BenchmarkOptions {
    #[arg(long, default_value_t = crate::constants::DEFAULT_BENCHMARK_REQUESTS)]
    pub requests: usize,
    #[arg(long, default_value_t = crate::constants::DEFAULT_BENCHMARK_INPUT_TOKENS)]
    pub input_tokens: usize,
    #[arg(long, default_value_t = crate::constants::DEFAULT_BENCHMARK_OUTPUT_TOKENS)]
    pub output_tokens: usize,
    #[arg(long, default_value_t = crate::constants::DEFAULT_TTFT_SLO_US)]
    pub ttft_slo_us: u64,
    #[arg(long, default_value_t = crate::constants::DEFAULT_TPOT_SLO_US)]
    pub tpot_slo_us: u64,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub host_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct CompareOptions {
    #[arg(long)]
    pub baseline: PathBuf,
    #[arg(long)]
    pub candidate: PathBuf,
    #[arg(long)]
    pub correctness_passed: bool,
    #[arg(long, default_value_t = crate::constants::DEFAULT_MAX_P99_REGRESSION)]
    pub max_p99_regression: f64,
}
#[derive(clap::Args)]
pub struct InspectModelOptions {
    #[arg(long)]
    pub config: PathBuf,
    #[arg(long)]
    pub index: Option<PathBuf>,
    #[arg(long, default_value_t = crate::constants::DEFAULT_CONTEXT_TOKENS)]
    pub context_tokens: usize,
    #[arg(long, default_value_t = 1)]
    pub sequences: usize,
    #[arg(long, default_value_t = crate::constants::DEFAULT_DEVICE_MEMORY_GIB)]
    pub device_memory_gib: u64,
}
#[derive(clap::Args)]
pub struct AgentOptions {
    #[arg(long)]
    pub package: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub host_memory_mib: u64,
    #[arg(long, default_value_t = 0)]
    pub probe_memory_mib: u64,
}
#[derive(clap::Args)]
pub struct InspectPackageOptions {
    #[arg(long)]
    pub package: PathBuf,
    /// Fail inspection when resident weights exceed this optional budget.
    #[arg(long)]
    pub host_memory_mib: Option<u64>,
    #[arg(long, default_value_t = crate::constants::DEFAULT_STAGING_MIB)]
    pub staging_memory_mib: usize,
    /// Inspect expanded F32 storage instead of the source storage format.
    #[arg(long)]
    pub weights_f32: bool,
}
#[derive(clap::Args)]
pub struct TokenizeOptions {
    #[arg(long)]
    pub package: PathBuf,
    #[arg(
        long,
        conflicts_with = "messages",
        required_unless_present = "messages"
    )]
    pub text: Option<String>,
    #[arg(long, conflicts_with = "text")]
    pub messages: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub enable_thinking: bool,
}
