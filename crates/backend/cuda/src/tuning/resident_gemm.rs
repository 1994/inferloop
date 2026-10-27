//! Actual-weight W4A4 graph tuning. Every candidate preserves quantization and K order.
use super::projection_budget::{
    GEMM_RANK, ProjectionKind, ProjectionShape, SearchBudget, SearchScope,
};
use super::{MAX_REPS, MIN_CONFIRM_SPEEDUP, MIN_REPS, REP_MS, TilingTable, WARMUP_MS};
use crate::constants::NVFP4_GROUP_SIZE;
use crate::constants::NVFP4_QUANT_CODES_TILE;
use crate::constants::NVFP4_QUANT_SCALES_TILE;
use crate::constants::PROMPT_GEMM_TILE_ROWS;
use crate::constants::QUANT_GEMM_TILE;
use crate::constants::WIDE_PROMPT_GEMM_TILE_ROWS;
use crate::device::vendor_fp8::nvfp4::ModelCache;
use crate::resident::nvfp4_gemm::kernels;
use crate::{
    device::{CudaDevice, DeviceProfile, device_error},
    mlp::ProjectionWeight,
};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::env::var_os;
use std::fs::read_to_string;
use std::path::Path;
use std::result::Result as StdResult;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, OnceLock, Weak},
    time::{Duration, Instant},
};

/// Bump when quantization, arithmetic or the paired measurement protocol changes.
const KERNEL_SCHEMA: &str = "cutile0.4-nvfp4-v3:search-ab:confirm-ba:block16-e4m3:quant512:mma-k256:f32:full-output:alpha1/(wg*xs):tile-m128";
const ACTIVATION_MODE: &str = "nvfp4-w4a4-block16";
/// Linear interpolation weight of both central samples for an even-sized median.
const MEDIAN_INTERPOLATION: f32 = 0.5;

/// Rows processed by one prompt activation quantization block.
pub(crate) const QUANTIZATION_ROWS: usize = 16;

/// Default row tile before a hardware-scoped measurement selects another configuration.
pub(crate) const fn prompt_tile_rows(rows: usize) -> usize {
    if rows >= WIDE_PROMPT_GEMM_TILE_ROWS {
        WIDE_PROMPT_GEMM_TILE_ROWS
    } else if rows > QUANT_GEMM_TILE[0] {
        PROMPT_GEMM_TILE_ROWS
    } else {
        QUANT_GEMM_TILE[0]
    }
}

/// Native FP8/block projection baseline; W4A4 configurations use [`GemmConfig::baseline`].
pub(crate) const fn native_tile(rows: usize) -> [usize; 2] {
    [prompt_tile_rows(rows), QUANT_GEMM_TILE[1]]
}

/// Prompt rows from which the baseline takes [`WIDE_PROMPT_COLUMNS`].
const WIDE_PROMPT_ROWS: usize = 512;
/// Output-column tile of the wide prompt baseline.
const WIDE_PROMPT_COLUMNS: usize = 128;
/// Supported row-tile variants of the existing native scaled MMA kernels.
const TILE_M_SEARCH: [usize; 3] = [
    QUANT_GEMM_TILE[0],
    PROMPT_GEMM_TILE_ROWS,
    WIDE_PROMPT_GEMM_TILE_ROWS,
];
/// Supported output-column tile variants; the first is the conservative baseline.
const TILE_N_SEARCH: [usize; 2] = [QUANT_GEMM_TILE[1], QUANT_GEMM_TILE[1] * 2];
/// The existing native K loop has plain and eight-stage compiler-prefetch variants.
const LOOP_SEARCH: [MainLoop; 2] = [MainLoop::Plain, MainLoop::Prefetch8];
/// Deterministic signed activation pattern used with the real loaded checkpoint weights.
const SAMPLE_PERIOD: usize = 131;
/// Coprime stride visits every sampled activation magnitude.
const SAMPLE_STRIDE: usize = 17;
/// Center of the deterministic activation sample.
const SAMPLE_CENTER: f32 = 65.0;
/// Scaling avoids saturating every activation group.
const SAMPLE_DIVISOR: f32 = 37.0;
/// CUDA JIT optimization level required for representative timing.
const JIT_OPT_LEVEL: u8 = 3;

/// Existing independent allowances for the non-projection tuning families.
const AUXILIARY_SEARCH_BUDGET: Duration = Duration::from_secs(30);
/// One reverse confirmation and an optional native-baseline confirmation after rejection.
const CONFIRMATION_TRIALS: usize = 2;

/// Native candidate trials plus the independent confirmation paths.
pub(crate) const fn search_trials() -> usize {
    TILE_M_SEARCH.len() * TILE_N_SEARCH.len() * LOOP_SEARCH.len() - 1 + CONFIRMATION_TRIALS
}
#[cfg(test)]
const HARDWARE_VALIDATION_BUDGET: Duration = Duration::from_secs(600);

/// Shared by target/draft weights and every program created from one loaded model.
///
/// The allowance starts at the first new resident search, after weight binding.
/// Cache identities and mandatory real-weight proofs remain available after it expires.
pub struct ResidentTuningSession {
    /// Immutable-weight Lt proofs shared across this model's serving states.
    pub(crate) vendor_fp4: Arc<ModelCache>,
    projection_budget: Mutex<SearchBudget>,
    greedy_started: OnceLock<Instant>,
    norm_started: OnceLock<Instant>,
    recurrent_started: OnceLock<Instant>,
    attention_started: OnceLock<Instant>,
    budget: Duration,
}

impl Default for ResidentTuningSession {
    fn default() -> Self {
        Self {
            vendor_fp4: Arc::new(ModelCache::default()),
            projection_budget: Mutex::new(SearchBudget::default()),
            greedy_started: OnceLock::new(),
            norm_started: OnceLock::new(),
            recurrent_started: OnceLock::new(),
            attention_started: OnceLock::new(),
            budget: AUXILIARY_SEARCH_BUDGET,
        }
    }
}

impl ResidentTuningSession {
    #[cfg(test)]
    pub(crate) fn for_hardware_validation() -> Self {
        Self {
            budget: HARDWARE_VALIDATION_BUDGET,
            projection_budget: Mutex::new(SearchBudget::with_limit(HARDWARE_VALIDATION_BUDGET)),
            ..Self::default()
        }
    }
    #[cfg(test)]
    pub(crate) fn without_search_budget() -> Self {
        Self {
            budget: Duration::ZERO,
            projection_budget: Mutex::new(SearchBudget::with_limit(Duration::ZERO)),
            ..Self::default()
        }
    }

    /// A separate finite allowance lets greedy tuning run after GEMM searches finish.
    pub(crate) fn permits_greedy_search(&self) -> bool {
        self.greedy_started.get_or_init(Instant::now).elapsed() < self.budget
    }

    /// Recurrent searches share one finite allowance across all programs in this model.
    pub(crate) fn permits_recurrent_search(&self) -> bool {
        self.recurrent_started.get_or_init(Instant::now).elapsed() < self.budget
    }

    /// Prefill attention has its own finite allowance across all model programs.
    pub(crate) fn permits_attention_search(&self) -> bool {
        self.attention_started.get_or_init(Instant::now).elapsed() < self.budget
    }

    /// Normalization uses its own finite allowance across this model.
    pub(crate) fn permits_norm_search(&self) -> bool {
        self.norm_started.get_or_init(Instant::now).elapsed() < self.budget
    }

    pub(crate) fn plan_projection(
        &self,
        kind: ProjectionKind,
        shape: ProjectionShape,
        trials: usize,
    ) -> Result<()> {
        let mut budget = self
            .projection_budget
            .lock()
            .map_err(|_| Error::invariant("projection search budget poisoned"))?;
        let snapshot = budget.plan(kind, shape, trials)?.then(|| budget.snapshot());
        drop(budget);
        if let Some(snapshot) = snapshot {
            tracing::info!(target: "infer::tuning", ?kind, ?shape,
                planned_shapes=snapshot.planned, budget_ms=snapshot.allowance.as_millis(),
                "resident projection search workload planned");
        }
        Ok(())
    }

    pub(crate) fn complete_projection(
        &self,
        kind: ProjectionKind,
        shape: ProjectionShape,
    ) -> Result<()> {
        self.projection_budget
            .lock()
            .map_err(|_| Error::invariant("projection search budget poisoned"))?
            .complete(kind, shape);
        Ok(())
    }

    pub(crate) fn report_projection_work(&self) -> Result<()> {
        let snapshot = self
            .projection_budget
            .lock()
            .map_err(|_| Error::invariant("projection search budget poisoned"))?
            .snapshot();
        tracing::info!(target: "infer::tuning", planned_shapes=snapshot.planned,
            budget_ms=snapshot.allowance.as_millis(), search_elapsed_ms=snapshot.elapsed.as_millis(),
            pending_shapes=?snapshot.pending, "resident projection preparation coverage");
        Ok(())
    }

    pub(crate) fn projection_search(&self) -> Result<SearchScope<'_>> {
        SearchScope::begin(&self.projection_budget)
    }

    pub(crate) fn permits_search(&self) -> bool {
        let Ok(mut budget) = self.projection_budget.lock() else {
            tracing::error!(target: "infer::tuning", "projection search budget poisoned; search disabled");
            return false;
        };
        let elapsed = budget.elapsed();
        let permitted = budget.permits(elapsed);
        let snapshot = budget.report_expiration(elapsed).then(|| budget.snapshot());
        drop(budget);
        if let Some(snapshot) = snapshot {
            tracing::info!(target: "infer::tuning", elapsed_ms=elapsed.as_millis(),
                budget_ms=snapshot.allowance.as_millis(), pending_shapes=?snapshot.pending,
                "resident projection search budget exhausted: missing shapes");
        }
        permitted
    }

    fn exhausted(&self, key: &ResidentGemmKey) -> bool {
        if self.permits_search() {
            return false;
        }
        tracing::info!(target: "infer::tuning", ?key,
            "resident GEMM model search budget exhausted: incomplete evidence discarded");
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResidentGemmKey {
    pub hardware: String,
    pub driver_version: u32,
    /// Missing driver build forbids persistent decisions; process-local reuse remains safe.
    #[serde(default)]
    pub driver_build: Option<String>,
    pub kernel_schema: String,
    /// Canonical quantizer and candidate Tile IR identities, including the assembler build.
    #[serde(default)]
    pub kernel_fingerprint: String,
    pub activation_mode: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl ResidentGemmKey {
    fn new(
        profile: &DeviceProfile,
        driver_version: u32,
        shape: [usize; GEMM_RANK],
        driver_build: Option<String>,
        kernel_fingerprint: String,
    ) -> Result<Self> {
        let [m, n, k] = shape;
        let compile = CompileOptions::default();
        Ok(Self {
            hardware: serde_json::to_string(profile).map_err(device_error)?,
            driver_version,
            driver_build,
            kernel_fingerprint,
            kernel_schema: format!(
                "{KERNEL_SCHEMA}:jit-opt{:?}:debug{}",
                compile.opt_level, compile.device_debug
            ),
            activation_mode: ACTIVATION_MODE.into(),
            m,
            n,
            k,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MainLoop {
    Plain,
    Prefetch8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GemmConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub main_loop: MainLoop,
}

impl GemmConfig {
    pub(crate) const fn baseline(rows: usize) -> Self {
        // Wide prompt graphs take a 128-column tile with the prefetching main loop: at 2048 rows
        // it cut a 32K prefill by 0.14 s, while decode and short prompts keep the narrow tile.
        if rows >= WIDE_PROMPT_ROWS {
            return Self {
                tile_m: prompt_tile_rows(rows),
                tile_n: WIDE_PROMPT_COLUMNS,
                main_loop: MainLoop::Prefetch8,
            };
        }
        Self {
            tile_m: prompt_tile_rows(rows),
            tile_n: QUANT_GEMM_TILE[1],
            main_loop: MainLoop::Plain,
        }
    }
    fn valid(self) -> bool {
        TILE_M_SEARCH.contains(&self.tile_m) && TILE_N_SEARCH.contains(&self.tile_n)
    }
    pub(crate) const fn tile(self) -> [usize; 2] {
        [self.tile_m, self.tile_n]
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Captured W4A4 operands and scale contract"
    )]
    pub(crate) fn record(
        self,
        scope: &Scope,
        weight: &ProjectionWeight,
        scale: f32,
        q: &Tensor<f4e2m1fnx2>,
        qs: &Tensor<f8e4m3fn>,
        out: &mut Tensor<f32>,
        columns: usize,
    ) -> StdResult<(), DeviceError> {
        let ProjectionWeight::Fp4(w, s, global) = weight else {
            return Err(DeviceError::Launch(
                "resident tuning requires FP4 weights".into(),
            ));
        };
        let generics = vec![
            columns.to_string(),
            self.tile_m.to_string(),
            self.tile_n.to_string(),
        ];
        match self.main_loop {
            MainLoop::Plain => scope.record(
                kernels::packed(
                    out.partition(self.tile()),
                    q,
                    qs,
                    w,
                    s,
                    1.0 / (global * scale),
                )
                .generics(generics),
            )?,
            MainLoop::Prefetch8 => scope.record(
                kernels::packed_prefill(
                    out.partition(self.tile()),
                    q,
                    qs,
                    w,
                    s,
                    1.0 / (global * scale),
                )
                .generics(generics),
            )?,
        };
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedTimes {
    pub baseline_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
    pub baseline_median_ms: f32,
    pub candidate_median_ms: f32,
    pub speedup: f32,
}

/// First measured arm of every pair, retained so cached evidence is auditable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementOrder {
    /// Legacy entries lack the arm order and cannot satisfy the current protocol.
    #[default]
    Unspecified,
    BaselineFirst,
    CandidateFirst,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmTrial {
    pub config: GemmConfig,
    #[serde(default)]
    pub order: MeasurementOrder,
    pub cold: PairedTimes,
    pub warm: PairedTimes,
}

impl GemmTrial {
    fn valid(&self) -> bool {
        self.config.valid() && times_valid(&self.cold) && times_valid(&self.warm)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidentGemmEntry {
    pub key: ResidentGemmKey,
    pub selected: GemmConfig,
    pub bitwise_verified: bool,
    pub trials: Vec<GemmTrial>,
    pub confirmation: GemmTrial,
}

impl ResidentGemmEntry {
    fn valid(&self) -> bool {
        self.bitwise_verified
            && !self.key.kernel_fingerprint.is_empty()
            && self
                .key
                .driver_build
                .as_ref()
                .is_none_or(|build| !build.is_empty())
            && self.selected.valid()
            && self.confirmation.config == self.selected
            && self.confirmation.order == MeasurementOrder::CandidateFirst
            && self.confirmation.valid()
            && self
                .trials
                .iter()
                .all(|trial| trial.order == MeasurementOrder::BaselineFirst && trial.valid())
            && (self.selected == GemmConfig::baseline(self.key.m)
                || (confirmed(&self.confirmation)
                    && self
                        .trials
                        .iter()
                        .any(|trial| trial.config == self.selected && confirmed(trial))))
    }
}

fn sample_median(samples: &[f32]) -> Option<f32> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f32::total_cmp);
    let lo = sorted.len().saturating_sub(1) / 2;
    let hi = sorted.len() / 2;
    // Preserve cutile Measurement::quantile_ms(0.5)'s separate f32 multiplies and sum.
    let median = if lo == hi {
        sorted[lo]
    } else {
        let lower = sorted[lo] * MEDIAN_INTERPOLATION;
        let upper = sorted[hi] * MEDIAN_INTERPOLATION;
        lower + upper
    };
    Some(median)
}

pub(crate) fn times_valid(times: &PairedTimes) -> bool {
    !times.baseline_ms.is_empty()
        && times.baseline_ms.len() == times.candidate_ms.len()
        && sample_median(&times.baseline_ms)
            .is_some_and(|median| median.to_bits() == times.baseline_median_ms.to_bits())
        && sample_median(&times.candidate_ms)
            .is_some_and(|median| median.to_bits() == times.candidate_median_ms.to_bits())
        && times
            .baseline_ms
            .iter()
            .chain(&times.candidate_ms)
            .all(|t| t.is_finite() && *t > 0.0)
        && times.baseline_median_ms.is_finite()
        && times.baseline_median_ms > 0.0
        && times.candidate_median_ms.is_finite()
        && times.candidate_median_ms > 0.0
        && times.speedup.is_finite()
        && times.speedup > 0.0
        && times.speedup.to_bits()
            == (times.baseline_median_ms / times.candidate_median_ms).to_bits()
}

fn confirmed(trial: &GemmTrial) -> bool {
    trial.valid()
        && trial.cold.speedup >= MIN_CONFIRM_SPEEDUP
        && trial.warm.speedup >= MIN_CONFIRM_SPEEDUP
}

#[derive(Debug, PartialEq, Eq)]
enum Decision {
    Cached(GemmConfig),
    Baseline,
    Measure,
}

fn decision(
    table: Option<&TilingTable>,
    key: &ResidentGemmKey,
    autotune: bool,
) -> Result<Decision> {
    if let Some(entry) = table.and_then(|t| t.resident_gemm.iter().find(|entry| entry.key == *key))
    {
        if !entry.valid() {
            return Err(Error::invalid("invalid resident GEMM tuning evidence"));
        }
        return Ok(Decision::Cached(entry.selected));
    }
    Ok(if autotune {
        Decision::Measure
    } else {
        Decision::Baseline
    })
}

/// Process-local identity of immutable weight codes, scales and scalar scale values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct WeightKey {
    codes: usize,
    scales: usize,
    global: u32,
    activation: u32,
}

impl WeightKey {
    pub(crate) fn new(weight: &ProjectionWeight, scale: f32) -> Result<Self> {
        let ProjectionWeight::Fp4(codes, scales, global) = weight else {
            return Err(Error::invalid("resident GEMM requires FP4 weights"));
        };
        Ok(Self {
            codes: Arc::as_ptr(codes) as usize,
            scales: Arc::as_ptr(scales) as usize,
            global: global.to_bits(),
            activation: scale.to_bits(),
        })
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct VerificationKey {
    geometry: ResidentGemmKey,
    weight: WeightKey,
    config: GemmConfig,
}

struct Verification {
    codes: Weak<Tensor<f4e2m1fnx2>>,
    scales: Weak<Tensor<f8e4m3fn>>,
    accepted: bool,
}

type Verifications = Mutex<BTreeMap<VerificationKey, Verification>>;
fn verifications() -> &'static Verifications {
    static VERIFIED: OnceLock<Verifications> = OnceLock::new();
    VERIFIED.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Read the installed driver release/build rather than its supported CUDA API version.
fn driver_build() -> Option<String> {
    let version = read_to_string("/proc/driver/nvidia/version").ok()?;
    version
        .lines()
        .find(|line| line.starts_with("NVRM version:"))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
}

/// Frontend-only identities: no cubin compilation or kernel execution is requested here.
fn kernel_fingerprint(device: &CudaDevice, shape: [usize; GEMM_RANK]) -> Result<String> {
    use cutile::cutile_compiler::cuda_tile_runtime_utils as compiler;
    type Fingerprints = Mutex<BTreeMap<(String, [usize; GEMM_RANK]), String>>;
    static FINGERPRINTS: OnceLock<Fingerprints> = OnceLock::new();
    let compile = CompileOptions::default();
    let environment = format!(
        "{}:{}:{:?}:{:?}:{:?}:{:?}",
        device.profile()?.architecture,
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        compile,
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS"),
    );
    let memo = FINGERPRINTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let cached = memo
        .lock()
        .map_err(|_| Error::invariant("kernel identity cache poisoned"))?
        .get(&(environment.clone(), shape))
        .cloned();
    if let Some(value) = cached {
        return Ok(value);
    }
    let [m, n, k] = shape;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut packed = api::meta::<f4e2m1fnx2>(&[m, k / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut scales = api::meta::<f8e4m3fn>(&[m, k / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight = api::meta::<f4e2m1fnx2>(&[n, k / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight_scales = api::meta::<f8e4m3fn>(&[n, k / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::meta::<f32>(&[m, n])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut identities = vec![
        kernels::quantize(
            (&mut packed).partition(NVFP4_QUANT_CODES_TILE),
            (&mut scales).partition(NVFP4_QUANT_SCALES_TILE),
            &input,
            1.0,
        )
        .generics(vec![k.to_string()])
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?,
    ];
    for tile_m in TILE_M_SEARCH {
        for tile_n in TILE_N_SEARCH {
            for main_loop in LOOP_SEARCH {
                let tile = [tile_m, tile_n];
                let generics = vec![k.to_string(), tile_m.to_string(), tile_n.to_string()];
                let identity = match main_loop {
                    MainLoop::Plain => kernels::packed(
                        (&mut output).partition(tile),
                        &packed,
                        &scales,
                        &weight,
                        &weight_scales,
                        1.0,
                    )
                    .generics(generics)
                    .l2_cache_key_on(&device.stream),
                    MainLoop::Prefetch8 => kernels::packed_prefill(
                        (&mut output).partition(tile),
                        &packed,
                        &scales,
                        &weight,
                        &weight_scales,
                        1.0,
                    )
                    .generics(generics)
                    .l2_cache_key_on(&device.stream),
                }
                .map_err(device_error)?;
                identities.push(identity);
            }
        }
    }
    let fingerprint = serde_json::to_string(&identities).map_err(device_error)?;
    memo.lock()
        .map_err(|_| Error::invariant("kernel identity cache poisoned"))?
        .insert((environment, shape), fingerprint.clone());
    Ok(fingerprint)
}

/// A performance cache never substitutes for this real weight and scale's numerical gate.
fn verified_config(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    scale: f32,
    key: &ResidentGemmKey,
    selected: GemmConfig,
) -> Result<GemmConfig> {
    let baseline = GemmConfig::baseline(key.m);
    if selected == baseline {
        return Ok(baseline);
    }
    let ProjectionWeight::Fp4(codes, scales, _) = weight else {
        return Err(Error::invalid("resident GEMM requires FP4 weights"));
    };
    let verification_key = VerificationKey {
        geometry: key.clone(),
        weight: WeightKey::new(weight, scale)?,
        config: selected,
    };
    let cache = verifications();
    if let Some(previous) = cache
        .lock()
        .map_err(|_| Error::invariant("weight verification cache poisoned"))?
        .get(&verification_key)
        && previous
            .codes
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, codes))
        && previous
            .scales
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, scales))
    {
        return Ok(if previous.accepted {
            selected
        } else {
            baseline
        });
    }
    // A prior search cannot evict an otherwise usable cached choice. Every new weight still
    // completes its numerical proof; only searches for new performance evidence are time-bounded.
    let started = Instant::now();
    let validate = || -> Result<()> {
        let shape = [key.m, key.n, key.k];
        let base = Captured::new(device, weight, scale, shape, baseline)?;
        let candidate = Captured::new(device, weight, scale, shape, selected)?;
        base.run(device)?;
        candidate.run(device)?;
        verify_bitwise(
            &device.read(&candidate.output)?,
            &device.read(&base.output)?,
        )
    };
    let result = validate();
    if let Err(error) = &result {
        tracing::warn!(target: "infer::tuning", ?key, ?selected, %error,
            "resident GEMM weight verification failed: this weight uses baseline");
    }
    let accepted = result.is_ok();
    remember_verification(weight, verification_key, accepted)?;
    tracing::info!(target: "infer::tuning", shape=?[key.m, key.n, key.k], ?selected,
        accepted, elapsed_ms=started.elapsed().as_millis(), "resident GEMM cached choice weight proof");
    Ok(if accepted { selected } else { baseline })
}

/// Store a real-weight proof without retaining a checkpoint after its final owner is dropped.
fn remember_verification(
    weight: &ProjectionWeight,
    verification_key: VerificationKey,
    accepted: bool,
) -> Result<()> {
    let ProjectionWeight::Fp4(codes, scales, _) = weight else {
        return Err(Error::invalid("resident GEMM requires FP4 weights"));
    };
    let mut verified = verifications()
        .lock()
        .map_err(|_| Error::invariant("weight verification cache poisoned"))?;
    verified.retain(|_, entry| entry.codes.strong_count() > 0 && entry.scales.strong_count() > 0);
    verified.insert(
        verification_key,
        Verification {
            codes: Arc::downgrade(codes),
            scales: Arc::downgrade(scales),
            accepted,
        },
    );
    drop(verified);
    Ok(())
}

type Memo = Mutex<BTreeMap<ResidentGemmKey, GemmConfig>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Resolve outside graph capture. No diagnostic override contributes tuning evidence.
pub(crate) fn select(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    scale: f32,
    shape: [usize; GEMM_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> GemmConfig {
    let baseline = GemmConfig::baseline(shape[0]);
    match resolve(device, weight, scale, shape, autotune, session) {
        Ok(config) => config,
        Err(error) => {
            tracing::warn!(target: "infer::tuning", ?shape, ?baseline, %error, "resident GEMM conservative fallback");
            baseline
        }
    }
}

/// Current native comparison identity, shared by optional FP4 library qualifications.
pub(crate) fn cache_key(device: &CudaDevice, shape: [usize; GEMM_RANK]) -> Result<ResidentGemmKey> {
    ResidentGemmKey::new(
        device.profile()?,
        CudaDevice::driver_version()?,
        shape,
        driver_build(),
        kernel_fingerprint(device, shape)?,
    )
}

fn resolve(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    scale: f32,
    shape: [usize; GEMM_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<GemmConfig> {
    let baseline = GemmConfig::baseline(shape[0]);
    let fingerprint = kernel_fingerprint(device, shape)?;
    let key = ResidentGemmKey::new(
        device.profile()?,
        CudaDevice::driver_version()?,
        shape,
        driver_build(),
        fingerprint,
    )?;
    let cached = memo()
        .lock()
        .map_err(|_| Error::invariant("resident tuning cache poisoned"))?
        .get(&key)
        .copied();
    if let Some(config) = cached {
        tracing::info!(target: "infer::tuning", ?key, ?config, source="process-cache", "resident GEMM selected");
        let config = verified_config(device, weight, scale, &key, config)?;
        session.complete_projection(ProjectionKind::Fp4, shape)?;
        return Ok(config);
    }
    let path = super::table_path();
    let table = if key.driver_build.is_some() {
        super::resolve_table(path.as_deref())?
    } else {
        tracing::warn!(target: "infer::tuning", "driver build unavailable: persistent resident GEMM decisions disabled");
        None
    };
    let Some((config, source)) = selected_config(
        device,
        weight,
        scale,
        &key,
        table.as_ref(),
        session,
        (autotune, path.as_deref()),
    )?
    else {
        return Ok(baseline);
    };
    memo()
        .lock()
        .map_err(|_| Error::invariant("resident tuning cache poisoned"))?
        .insert(key.clone(), config);
    tracing::info!(target: "infer::tuning", ?key, ?config, source, "resident GEMM selected");
    let config = verified_config(device, weight, scale, &key, config)?;
    session.complete_projection(ProjectionKind::Fp4, shape)?;
    Ok(config)
}

fn selected_config(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    scale: f32,
    key: &ResidentGemmKey,
    table: Option<&TilingTable>,
    session: &ResidentTuningSession,
    (autotune, path): (bool, Option<&Path>),
) -> Result<Option<(GemmConfig, &'static str)>> {
    let selected = match decision(table, key, autotune)? {
        Decision::Cached(config) => {
            if let Some(entry) =
                table.and_then(|table| table.resident_gemm.iter().find(|entry| entry.key == *key))
            {
                tracing::info!(target: "infer::tuning", ?key, ?config, source="table-cache",
                    cold_baseline_ms=entry.confirmation.cold.baseline_median_ms,
                    cold_candidate_ms=entry.confirmation.cold.candidate_median_ms,
                    warm_baseline_ms=entry.confirmation.warm.baseline_median_ms,
                    warm_candidate_ms=entry.confirmation.warm.candidate_median_ms,
                    "resident GEMM cached timings");
            }
            (config, "table-cache")
        }
        Decision::Baseline => {
            let baseline = GemmConfig::baseline(key.m);
            tracing::info!(target: "infer::tuning", ?key, ?baseline, source="autotune-disabled", "resident GEMM selected");
            return Ok(None);
        }
        Decision::Measure => {
            let Some(config) = measured_config(device, weight, scale, key, session, path)? else {
                return Ok(None);
            };
            (config, "measured")
        }
    };
    Ok(Some(selected))
}

fn measured_config(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    scale: f32,
    key: &ResidentGemmKey,
    session: &ResidentTuningSession,
    path: Option<&Path>,
) -> Result<Option<GemmConfig>> {
    let baseline = GemmConfig::baseline(key.m);
    let entry = match tune(device, weight, scale, key.clone(), session) {
        Ok(Some(entry)) => entry,
        Ok(None) => return Ok(None),
        Err(error) => {
            tracing::warn!(target: "infer::tuning", ?key, ?baseline, %error,
                source="measurement-fallback", "resident GEMM conservative fallback");
            memo()
                .lock()
                .map_err(|_| Error::invariant("resident tuning cache poisoned"))?
                .insert(key.clone(), baseline);
            return Ok(None);
        }
    };
    let config = entry.selected;
    // Search and independent confirmation already proved this exact weight and scale.
    if config != baseline {
        remember_verification(
            weight,
            VerificationKey {
                geometry: key.clone(),
                weight: WeightKey::new(weight, scale)?,
                config,
            },
            true,
        )?;
    }
    let cached = key.driver_build.is_some()
        && super::update_table(path, |table| {
            table.resident_gemm.retain(|old| old.key != entry.key);
            table.resident_gemm.push(entry.clone());
        });
    tracing::info!(target: "infer::tuning", ?key, ?config, cached,
        cold_speedup=entry.confirmation.cold.speedup, warm_speedup=entry.confirmation.warm.speedup,
        cold_baseline_ms=entry.confirmation.cold.baseline_median_ms,
        cold_candidate_ms=entry.confirmation.cold.candidate_median_ms,
        warm_baseline_ms=entry.confirmation.warm.baseline_median_ms,
        warm_candidate_ms=entry.confirmation.warm.candidate_median_ms,
        source="measured", "resident GEMM selected");
    tracing::debug!(target: "infer::tuning", evidence=%serde_json::to_string(&entry).map_err(device_error)?, "resident GEMM paired timings");
    Ok(Some(config))
}

struct Captured {
    graph: CudaGraph<()>,
    output: Arc<Tensor<f32>>,
    _input: Arc<Tensor<f32>>,
    _packed: Tensor<f4e2m1fnx2>,
    _scales: Tensor<f8e4m3fn>,
}

impl Captured {
    fn new(
        device: &CudaDevice,
        weight: &ProjectionWeight,
        scale: f32,
        shape: [usize; GEMM_RANK],
        config: GemmConfig,
    ) -> Result<Self> {
        let [m, n, k] = shape;
        weight.validate(n, k)?;
        if m == 0 || !scale.is_finite() || scale <= 0.0 || !config.valid() {
            return Err(Error::invalid("resident GEMM tuning input"));
        }
        let length = m
            .checked_mul(k)
            .ok_or_else(|| Error::invalid("resident tuning input overflow"))?;
        let values = (0..length)
            .map(|i| {
                (f32::from(
                    u16::try_from((i % SAMPLE_PERIOD) * SAMPLE_STRIDE % SAMPLE_PERIOD)
                        .unwrap_or_default(),
                ) - SAMPLE_CENTER)
                    / SAMPLE_DIVISOR
            })
            .collect();
        let input = device.upload(values, &[m, k])?;
        let mut packed = api::zeros::<f4e2m1fnx2>(&[m, k / 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut scales = api::zeros::<f8e4m3fn>(&[m, k / NVFP4_GROUP_SIZE])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut output = api::zeros::<f32>(&[m, n])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::quantize(
                    (&mut packed).partition(NVFP4_QUANT_CODES_TILE),
                    (&mut scales).partition(NVFP4_QUANT_SCALES_TILE),
                    &input,
                    scale,
                )
                .generics(vec![k.to_string()]),
            )?;
            config.record(scope, weight, scale, &packed, &scales, &mut output, k)
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            output: Arc::new(output),
            _input: input,
            _packed: packed,
            _scales: scales,
        })
    }
    fn run(&self, device: &CudaDevice) -> Result<()> {
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        Ok(())
    }
}

fn verify_bitwise(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.is_empty()
        || actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(Error::invariant("resident GEMM tuning changed output bits"));
    }
    Ok(())
}

fn ordered_times(first: &[f32], second: &[f32], order: MeasurementOrder) -> Result<PairedTimes> {
    let (baseline, candidate) = match order {
        MeasurementOrder::BaselineFirst => (first, second),
        MeasurementOrder::CandidateFirst => (second, first),
        MeasurementOrder::Unspecified => {
            return Err(Error::invalid(
                "resident GEMM measurement arm order missing",
            ));
        }
    };
    let baseline_median_ms = sample_median(baseline)
        .ok_or_else(|| Error::invalid("resident GEMM baseline timing samples"))?;
    let candidate_median_ms = sample_median(candidate)
        .ok_or_else(|| Error::invalid("resident GEMM candidate timing samples"))?;
    let times = PairedTimes {
        baseline_ms: baseline.to_vec(),
        candidate_ms: candidate.to_vec(),
        baseline_median_ms,
        candidate_median_ms,
        speedup: baseline_median_ms / candidate_median_ms,
    };
    if !times_valid(&times) {
        return Err(Error::invalid("resident GEMM paired timing samples"));
    }
    Ok(times)
}

fn measure(
    device: &CudaDevice,
    baseline: &Captured,
    candidate: &Captured,
    clear_l2: bool,
    order: MeasurementOrder,
) -> Result<PairedTimes> {
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2,
    };
    let run = |graph: &Captured| {
        graph
            .graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (first, second) = match order {
        MeasurementOrder::BaselineFirst => (baseline, candidate),
        MeasurementOrder::CandidateFirst => (candidate, baseline),
        MeasurementOrder::Unspecified => {
            return Err(Error::invalid(
                "resident GEMM measurement arm order missing",
            ));
        }
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(first), |_| run(second))
        .map_err(device_error)?;
    ordered_times(a.times_ms(), b.times_ms(), order)
}

fn trial(
    device: &CudaDevice,
    base: &Captured,
    candidate: &Captured,
    config: GemmConfig,
    order: MeasurementOrder,
) -> Result<GemmTrial> {
    Ok(GemmTrial {
        config,
        order,
        cold: measure(device, base, candidate, true, order)?,
        warm: measure(device, base, candidate, false, order)?,
    })
}

/// One known-good baseline and immutable real-weight inputs for candidate trials.
struct CandidateSearch<'a> {
    device: &'a CudaDevice,
    weight: &'a ProjectionWeight,
    scale: f32,
    shape: [usize; GEMM_RANK],
    base: &'a Captured,
    expected: &'a [f32],
}

impl CandidateSearch<'_> {
    fn trial(&self, config: GemmConfig, order: MeasurementOrder) -> Result<GemmTrial> {
        let candidate = Captured::new(self.device, self.weight, self.scale, self.shape, config)?;
        candidate.run(self.device)?;
        verify_bitwise(&self.device.read(&candidate.output)?, self.expected)?;
        let measured = trial(self.device, self.base, &candidate, config, order)?;
        verify_bitwise(&self.device.read(&candidate.output)?, self.expected)?;
        Ok(measured)
    }

    fn verify_baseline(&self) -> Result<()> {
        self.base.run(self.device)?;
        verify_bitwise(&self.device.read(&self.base.output)?, self.expected)
    }
}

fn tune(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    scale: f32,
    key: ResidentGemmKey,
    session: &ResidentTuningSession,
) -> Result<Option<ResidentGemmEntry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile.opt_level.is_some_and(|opt| opt != JIT_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "resident tuning requires release and CUDA JIT O3",
        ));
    }
    let shape = [key.m, key.n, key.k];
    session.plan_projection(ProjectionKind::Fp4, shape, search_trials())?;
    let _search = session.projection_search()?;
    if session.exhausted(&key) {
        return Ok(None);
    }
    let baseline = GemmConfig::baseline(key.m);
    let base = Captured::new(device, weight, scale, shape, baseline)?;
    base.run(device)?;
    let expected = device.read(&base.output)?;
    verify_bitwise(&expected, &expected)?;
    let search = CandidateSearch {
        device,
        weight,
        scale,
        shape,
        base: &base,
        expected: &expected,
    };
    let mut selected = baseline;
    let mut best = 1.0_f32;
    let mut trials = Vec::new();
    for tile_m in TILE_M_SEARCH {
        for tile_n in TILE_N_SEARCH {
            for main_loop in LOOP_SEARCH {
                let config = GemmConfig {
                    tile_m,
                    tile_n,
                    main_loop,
                };
                if config == baseline {
                    continue;
                }
                if session.exhausted(&key) {
                    return Ok(None);
                }
                match search.trial(config, MeasurementOrder::BaselineFirst) {
                    Ok(measured) => {
                        let score = measured.cold.speedup.min(measured.warm.speedup);
                        if confirmed(&measured) && score > best {
                            best = score;
                            selected = config;
                        }
                        trials.push(measured);
                    }
                    Err(error) => {
                        tracing::warn!(target: "infer::tuning", ?key, ?config, %error,
                            "resident GEMM candidate rejected; no evidence retained");
                        // A baseline/device failure still aborts the search; no recovery is assumed.
                        search.verify_baseline()?;
                    }
                }
            }
        }
    }
    // A new capture measures candidate first to confirm both cache states in reverse order.
    if session.exhausted(&key) {
        return Ok(None);
    }
    let mut confirmation = search.trial(selected, MeasurementOrder::CandidateFirst)?;
    if selected != baseline && !confirmed(&confirmation) {
        if session.exhausted(&key) {
            return Ok(None);
        }
        selected = baseline;
        confirmation = search.trial(baseline, MeasurementOrder::CandidateFirst)?;
    }
    let entry = ResidentGemmEntry {
        key,
        selected,
        bitwise_verified: true,
        trials,
        confirmation,
    };
    if !entry.valid() {
        return Err(Error::invalid("invalid resident GEMM confirmation timings"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../tests/unit/resident_gemm_tuning.rs"]
mod tests;
