//! Hardware-scoped FP8 channel GEMM tuning on immutable loaded weights.
//! Every candidate uses the original quantizer and passes full-output finite/bitwise gates.

#[cfg(test)]
use super::TilingTable;
use super::resident_gemm::{PairedTimes, ResidentTuningSession};
use super::{MAX_REPS, MIN_CONFIRM_SPEEDUP, MIN_REPS, REP_MS, WARMUP_MS};
use crate::constants::AUX_KERNEL_TILE;
use crate::constants::PROMPT_GEMM_TILE_ROWS;
use crate::constants::QUANT_GEMM_TILE;
use crate::constants::WIDE_PREFILL_LANES;
use crate::constants::WIDE_PROMPT_GEMM_TILE_ROWS;
pub use crate::device::vendor_fp8::{AlgorithmConfig, LibraryIdentity};
use crate::tuning::resident_gemm::prompt_tile_rows;
use crate::{
    device::{CudaDevice, device_error, vendor_fp8},
    mlp::ProjectionWeight,
    resident::fp8_gemm::kernels,
};
use cuda_core::f8e4m3fn;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::env::var;
use std::env::var_os;
use std::fs::read_to_string;
use std::iter::once;
use std::result::Result as StdResult;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock, Weak},
    time::{Duration, Instant},
};

pub(crate) mod probe_inputs;

const CONTRACT: &str = "fp8-channel-exact-v2:row-pad-pow2:maxabs/448:floor1e-12:clip448:e4m3-rne:mma-k128-f32:lt-scalar32f-scales1:fastaccum0:col-tn:raw*xs*ws:rn-ftz0:align256:workspace0:four-fixtures:exact-native-or-verified-lt-optin:search-ab-confirm-ba:tile-m128";
const ROW_TILES: [usize; 3] = [
    QUANT_GEMM_TILE[0],
    PROMPT_GEMM_TILE_ROWS,
    WIDE_PROMPT_GEMM_TILE_ROWS,
];
const COLUMN_TILES: [usize; 2] = [QUANT_GEMM_TILE[1], QUANT_GEMM_TILE[1] * 2];
const PERIOD: usize = 131;
const STRIDE: usize = 17;
const CENTER: f32 = 65.0;
const DIVISOR: f32 = 37.0;
const FIXTURES: usize = 4;
const FIXTURE_RANDOM: usize = 3;
const RANDOM_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
const RANDOM_SEED: u64 = 0xd1b5_4a32_d192_ed03;
const RANDOM_SHIFT: u32 = 29;
const GEMM_RANK: usize = 3;
const EXPONENT_PERIOD: usize = 17;
const EXPONENT_OFFSET: i32 = 12;
const MEDIAN_INTERPOLATION: f32 = 0.5;
const JIT_OPT_LEVEL: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NativeConfig {
    pub tile_m: usize,
    pub tile_n: usize,
}
impl NativeConfig {
    pub(crate) const fn baseline(rows: usize) -> Self {
        Self {
            tile_m: prompt_tile_rows(rows),
            tile_n: QUANT_GEMM_TILE[1],
        }
    }
    fn valid(self) -> bool {
        ROW_TILES.contains(&self.tile_m) && COLUMN_TILES.contains(&self.tile_n)
    }
    const fn tile(self) -> [usize; 2] {
        [self.tile_m, self.tile_n]
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Choice {
    Native(NativeConfig),
    Lt(AlgorithmConfig),
}
impl Choice {
    pub(crate) const fn baseline(m: usize) -> Self {
        Self::Native(NativeConfig::baseline(m))
    }
    fn valid(self) -> bool {
        match self {
            Self::Native(config) => config.valid(),
            Self::Lt(config) => config.id >= 0 && config.split_k >= 0,
        }
    }
    /// Capture only consumes a verified choice and pre-warmed Lt descriptors.
    #[expect(
        clippy::too_many_arguments,
        reason = "The capture boundary explicitly names every GEMM operand, its scales and its immutable choice"
    )]
    pub(crate) fn record(
        self,
        scope: &Scope,
        support: &vendor_fp8::Support,
        q: &Tensor<f8e4m3fn>,
        weight: &Tensor<f8e4m3fn>,
        output: &mut Tensor<f32>,
        xs: &Tensor<f32>,
        ws: &Tensor<f32>,
        columns: usize,
    ) -> StdResult<(), DeviceError> {
        let native = match self {
            Self::Native(config) => config,
            Self::Lt(config) => {
                if support.record_config(scope, config, q, weight, output, xs, ws)? {
                    return Ok(());
                }
                // An arena alias with a different alignment must fall back before any Lt node.
                NativeConfig::baseline(
                    usize::try_from(output.shape()[0])
                        .map_err(|error| DeviceError::Internal(error.to_string()))?,
                )
            }
        };
        scope.record(
            kernels::matmul(output.partition(native.tile()), q, weight, xs, ws).generics(vec![
                columns.to_string(),
                native.tile_m.to_string(),
                native.tile_n.to_string(),
            ]),
        )?;
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key {
    pub hardware: String,
    pub driver_api: u32,
    pub driver_build: Option<String>,
    pub kernel_fingerprint: String,
    pub library: Option<LibraryIdentity>,
    pub contract: String,
    /// Explicit numerical-profile opt-in; Lt does not promise the native K accumulation order.
    #[serde(default)]
    pub vendor_candidates: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}
impl Key {
    fn persistent(&self) -> bool {
        self.driver_build
            .as_ref()
            .is_some_and(|build| !build.is_empty())
            && !self.kernel_fingerprint.is_empty()
            && (!self.vendor_candidates || self.library.is_some())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct WeightKey {
    codes: usize,
    scales: usize,
}
impl WeightKey {
    pub(crate) fn new(weight: &ProjectionWeight) -> Result<Self> {
        let ProjectionWeight::Fp8(codes, scales) = weight else {
            return Err(Error::invalid("FP8 channel tuning weight"));
        };
        Ok(Self {
            codes: Arc::as_ptr(codes) as usize,
            scales: Arc::as_ptr(scales) as usize,
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// False records search baseline/candidate; true records independent candidate/baseline.
    pub reverse: bool,
    pub choice: Choice,
    pub cold: PairedTimes,
    pub warm: PairedTimes,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub selected: Choice,
    pub finite_bitwise: bool,
    pub fixtures: usize,
    pub trials: Vec<Trial>,
    pub confirmation: Trial,
}
fn median(values: &[f32]) -> Option<f32> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f32::total_cmp);
    let lo = ordered.len().saturating_sub(1) / 2;
    let hi = ordered.len() / 2;
    Some(if lo == hi {
        ordered[lo]
    } else {
        // Keep the benchmark library's separately rounded multiply/multiply/add median bits.
        let lower = ordered[lo] * MEDIAN_INTERPOLATION;
        let upper = ordered[hi] * MEDIAN_INTERPOLATION;
        lower + upper
    })
}
fn times_valid(times: &PairedTimes) -> bool {
    !times.baseline_ms.is_empty()
        && times.baseline_ms.len() == times.candidate_ms.len()
        && median(&times.baseline_ms)
            .is_some_and(|value| value.to_bits() == times.baseline_median_ms.to_bits())
        && median(&times.candidate_ms)
            .is_some_and(|value| value.to_bits() == times.candidate_median_ms.to_bits())
        && times.speedup.is_finite()
        && times.speedup > 0.0
        && times.speedup.to_bits()
            == (times.baseline_median_ms / times.candidate_median_ms).to_bits()
}
fn confirmed(trial: &Trial) -> bool {
    times_valid(&trial.cold)
        && times_valid(&trial.warm)
        && trial.cold.speedup >= MIN_CONFIRM_SPEEDUP
        && trial.warm.speedup >= MIN_CONFIRM_SPEEDUP
}
impl Entry {
    fn valid(&self) -> bool {
        self.finite_bitwise
            && self.fixtures == FIXTURES
            && self.selected.valid()
            && (!matches!(self.selected, Choice::Lt(_)) || self.key.vendor_candidates)
            && !self.key.kernel_fingerprint.is_empty()
            && self.key.contract == CONTRACT
            && self.confirmation.reverse
            && self.confirmation.choice == self.selected
            && times_valid(&self.confirmation.cold)
            && times_valid(&self.confirmation.warm)
            && self.trials.iter().all(|trial| {
                !trial.reverse
                    && trial.choice.valid()
                    && times_valid(&trial.cold)
                    && times_valid(&trial.warm)
            })
            && (self.selected == Choice::baseline(self.key.m)
                || (confirmed(&self.confirmation)
                    && self
                        .trials
                        .iter()
                        .any(|trial| trial.choice == self.selected && confirmed(trial))))
    }
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ProofKey {
    geometry: Key,
    weight: WeightKey,
    choice: Choice,
}
struct Proof {
    codes: Weak<Tensor<f8e4m3fn>>,
    scales: Weak<Tensor<f32>>,
    accepted: bool,
}
type Proofs = Mutex<BTreeMap<ProofKey, Proof>>;
fn proofs() -> &'static Proofs {
    static PROOFS: OnceLock<Proofs> = OnceLock::new();
    PROOFS.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn remembered(weight: &ProjectionWeight, key: &Key, choice: Choice) -> Result<Option<bool>> {
    let ProjectionWeight::Fp8(codes, scales) = weight else {
        return Err(Error::invalid("FP8 channel proof weight"));
    };
    let proof_key = ProofKey {
        geometry: key.clone(),
        weight: WeightKey::new(weight)?,
        choice,
    };
    let cache = proofs()
        .lock()
        .map_err(|_| Error::invariant("FP8 proof cache poisoned"))?;
    Ok(cache
        .get(&proof_key)
        .filter(|proof| {
            proof
                .codes
                .upgrade()
                .is_some_and(|owner| Arc::ptr_eq(&owner, codes))
                && proof
                    .scales
                    .upgrade()
                    .is_some_and(|owner| Arc::ptr_eq(&owner, scales))
        })
        .map(|proof| proof.accepted))
}
fn remember(weight: &ProjectionWeight, key: &Key, choice: Choice, accepted: bool) -> Result<()> {
    let ProjectionWeight::Fp8(codes, scales) = weight else {
        return Err(Error::invalid("FP8 channel proof weight"));
    };
    let mut cache = proofs()
        .lock()
        .map_err(|_| Error::invariant("FP8 proof cache poisoned"))?;
    cache.retain(|_, proof| proof.codes.strong_count() > 0 && proof.scales.strong_count() > 0);
    cache.insert(
        ProofKey {
            geometry: key.clone(),
            weight: WeightKey::new(weight)?,
            choice,
        },
        Proof {
            codes: Arc::downgrade(codes),
            scales: Arc::downgrade(scales),
            accepted,
        },
    );
    drop(cache);
    Ok(())
}
fn permits(session: &ResidentTuningSession, shape: [usize; GEMM_RANK]) -> bool {
    if session.permits_search() {
        true
    } else {
        tracing::info!(target:"infer::tuning", ?shape, "FP8 model tuning budget exhausted: unmeasured choice stays native");
        false
    }
}
fn driver_build() -> Option<String> {
    read_to_string("/proc/driver/nvidia/version")
        .ok()?
        .lines()
        .find(|line| line.starts_with("NVRM version:"))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
}
type Identities = Mutex<BTreeMap<(String, [usize; GEMM_RANK]), String>>;
fn fingerprint(device: &CudaDevice, shape: [usize; GEMM_RANK]) -> Result<String> {
    use cutile::cutile_compiler::cuda_tile_runtime_utils as compiler;
    static IDENTITIES: OnceLock<Identities> = OnceLock::new();
    let environment = format!(
        "{}:{}:{:?}:{:?}:{:?}:{:?}",
        device.profile()?.architecture,
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        CompileOptions::default(),
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS")
    );
    let cache = IDENTITIES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let remembered = cache
        .lock()
        .map_err(|_| Error::invariant("FP8 identity cache poisoned"))?
        .get(&(environment.clone(), shape))
        .cloned();
    if let Some(value) = remembered {
        return Ok(value);
    }
    let [rows, outputs, columns] = shape;
    let input = api::meta::<f32>(&[rows, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut codes = api::meta::<f8e4m3fn>(&[rows, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut xs = api::meta::<f32>(&[rows, 1])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight_codes = api::meta::<f8e4m3fn>(&[outputs, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let ws = api::meta::<f32>(&[outputs])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut out = api::meta::<f32>(&[rows, outputs])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut ids = vec![
        kernels::quantize(
            (&mut codes).partition([1, columns.next_power_of_two()]),
            (&mut xs).partition([1, 1]),
            &input,
        )
        .generics(vec![
            columns.to_string(),
            columns.next_power_of_two().to_string(),
        ])
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?,
    ];
    for tile_m in ROW_TILES {
        for tile_n in COLUMN_TILES {
            ids.push(
                kernels::matmul(
                    (&mut out).partition([tile_m, tile_n]),
                    &codes,
                    &weight_codes,
                    &xs,
                    &ws,
                )
                .generics(vec![
                    columns.to_string(),
                    tile_m.to_string(),
                    tile_n.to_string(),
                ])
                .l2_cache_key_on(&device.stream)
                .map_err(device_error)?,
            );
        }
    }
    ids.push(
        vendor_fp8::kernels::scale((&mut out).partition([1, AUX_KERNEL_TILE]), &xs, &ws)
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?,
    );
    let value = serde_json::to_string(&ids).map_err(device_error)?;
    cache
        .lock()
        .map_err(|_| Error::invariant("FP8 identity cache poisoned"))?
        .insert((environment, shape), value.clone());
    Ok(value)
}
type Memo = Mutex<BTreeMap<Key, Choice>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}

struct Captured {
    graph: CudaGraph<()>,
    output: Tensor<f32>,
    _vendor: Option<vendor_fp8::CaptureOwner>,
    _input: Arc<Tensor<f32>>,
    codes: Tensor<f8e4m3fn>,
    _xs: Tensor<f32>,
    _weight: ProjectionWeight,
}
fn inputs(m: usize, k: usize, fixture: usize) -> Result<Vec<f32>> {
    let length = m
        .checked_mul(k)
        .ok_or_else(|| Error::invalid("FP8 tuning input overflow"))?;
    if fixture >= FIXTURES {
        return Err(Error::invalid("FP8 tuning fixture index"));
    }
    Ok((0..length)
        .map(|i| {
            if m > 1 && i < k {
                return 0.0;
            }
            let base =
                (f32::from(u16::try_from((i % PERIOD) * STRIDE % PERIOD).unwrap_or_default())
                    - CENTER)
                    / DIVISOR;
            let exponent = i32::try_from(i % EXPONENT_PERIOD).unwrap_or_default() - EXPONENT_OFFSET;
            match fixture {
                0 => base,
                1 => {
                    if i % STRIDE == 0 {
                        base
                    } else {
                        0.0
                    }
                }
                2 => {
                    if i % 2 == 0 {
                        base.abs() * 2.0_f32.powi(exponent)
                    } else {
                        -base.abs() * 2.0_f32.powi(exponent)
                    }
                }
                FIXTURE_RANDOM => {
                    let random = u64::try_from(i)
                        .unwrap_or_default()
                        .wrapping_mul(RANDOM_MULTIPLIER)
                        .wrapping_add(RANDOM_SEED);
                    let random = random ^ (random >> RANDOM_SHIFT);
                    let magnitude = (f32::from(
                        u16::try_from(random % u64::try_from(PERIOD).unwrap_or_default())
                            .unwrap_or_default(),
                    ) - CENTER)
                        / DIVISOR;
                    let exponent =
                        i32::try_from(random % u64::try_from(EXPONENT_PERIOD).unwrap_or_default())
                            .unwrap_or_default()
                            - EXPONENT_OFFSET;
                    magnitude * 2.0_f32.powi(exponent)
                }
                _ => 0.0,
            }
        })
        .collect())
}

impl Captured {
    fn new(
        device: &CudaDevice,
        support: &mut vendor_fp8::Support,
        weight: &ProjectionWeight,
        shape: [usize; GEMM_RANK],
        choice: Choice,
        fixture: usize,
    ) -> Result<Self> {
        let [rows, outputs, columns] = shape;
        weight.validate(outputs, columns)?;
        if rows == 0 || !choice.valid() {
            return Err(Error::invalid("FP8 tuning shape/choice"));
        }
        let ProjectionWeight::Fp8(weight_codes, ws) = weight else {
            return Err(Error::invalid("FP8 tuning channel weight"));
        };
        let input = probe_inputs::upload(device, [rows, columns], fixture, || {
            inputs(rows, columns, fixture)
        })?;
        let mut codes = api::zeros::<f8e4m3fn>(&[rows, columns])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut xs = api::zeros::<f32>(&[rows, 1])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut output = api::zeros::<f32>(&[rows, outputs])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        support.retain_channels(weight_codes, ws);
        let owner = if let Choice::Lt(config) = choice {
            if !support
                .prepare_config(&codes, weight_codes, &output, config)
                .map_err(device_error)?
            {
                return Err(Error::invalid("FP8 Lt algorithm unavailable"));
            }
            Some(
                support
                    .capture_owner(shape, config)
                    .ok_or_else(|| Error::invariant("FP8 Lt capture owner missing"))?,
            )
        } else {
            None
        };
        let graph = CudaGraph::scope(&device.stream, |scope| {
            scope.record(
                kernels::quantize(
                    (&mut codes).partition([1, columns.next_power_of_two()]),
                    (&mut xs).partition([1, 1]),
                    &input,
                )
                .generics(vec![
                    columns.to_string(),
                    columns.next_power_of_two().to_string(),
                ]),
            )?;
            if let Choice::Lt(config) = choice {
                if !support.record_config(
                    scope,
                    config,
                    &codes,
                    weight_codes,
                    &mut output,
                    &xs,
                    ws,
                )? {
                    return Err(DeviceError::Internal(
                        "FP8 Lt candidate did not record its algorithm".into(),
                    ));
                }
                Ok(())
            } else {
                choice.record(
                    scope,
                    support,
                    &codes,
                    weight_codes,
                    &mut output,
                    &xs,
                    ws,
                    columns,
                )
            }
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            output,
            _vendor: owner,
            _input: input,
            codes,
            _xs: xs,
            _weight: weight.clone(),
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
fn bitwise(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.is_empty()
        || actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(Error::invariant(
            "FP8 candidate changed full-output bits or finiteness",
        ));
    }
    Ok(())
}
fn verify(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    shape: [usize; GEMM_RANK],
    choice: Choice,
    search: Option<&ResidentTuningSession>,
) -> Result<Option<bool>> {
    for fixture in 0..FIXTURES {
        // Cached choices always finish their current-weight proof. New candidates retain the
        // search allowance, including the numerical checks before their first measurement.
        if search.is_some_and(|session| !permits(session, shape)) {
            return Ok(None);
        }
        let base = Captured::new(
            device,
            support,
            weight,
            shape,
            Choice::baseline(shape[0]),
            fixture,
        )?;
        let candidate = Captured::new(device, support, weight, shape, choice, fixture)?;
        base.run(device)?;
        candidate.run(device)?;
        if let Err(error) = bitwise(
            &device.read_borrowed(&candidate.output)?,
            &device.read_borrowed(&base.output)?,
        ) {
            tracing::info!(target:"infer::tuning", ?shape, ?choice, fixture, %error, "FP8 candidate rejected by exact numerical gate");
            return Ok(Some(false));
        }
    }
    Ok(Some(true))
}
fn measure(
    device: &CudaDevice,
    baseline: &Captured,
    candidate: &Captured,
    clear_l2: bool,
    reverse: bool,
) -> Result<PairedTimes> {
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2,
    };
    let run = |capture: &Captured| {
        capture
            .graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    // Confirmation flips both warm and cold execution order, then maps samples back to their roles.
    let (first, second) = if reverse {
        (candidate, baseline)
    } else {
        (baseline, candidate)
    };
    let (first, second) =
        do_bench_paired(&device.stream, &options, |_| run(first), |_| run(second))
            .map_err(device_error)?;
    let (a, b) = if reverse {
        (second, first)
    } else {
        (first, second)
    };

    Ok(PairedTimes {
        baseline_ms: a.times_ms().to_vec(),
        candidate_ms: b.times_ms().to_vec(),
        baseline_median_ms: a.median_ms(),
        candidate_median_ms: b.median_ms(),
        speedup: a.median_ms() / b.median_ms(),
    })
}
fn trial(
    device: &CudaDevice,
    baseline: &Captured,
    candidate: &Captured,
    choice: Choice,
    reverse: bool,
) -> Result<Trial> {
    Ok(Trial {
        reverse,
        choice,
        cold: measure(device, baseline, candidate, true, reverse)?,
        warm: measure(device, baseline, candidate, false, reverse)?,
    })
}
fn cached_gate(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    key: &Key,
    choice: Choice,
) -> Result<Choice> {
    let baseline = Choice::baseline(key.m);
    if choice == baseline {
        return Ok(baseline);
    }
    // A new Support still has to restore descriptors before any capture, even with a live proof.
    let shape = [key.m, key.n, key.k];
    if let Some(accepted) = remembered(weight, key, choice)? {
        if !accepted {
            return Ok(baseline);
        }
        if let Choice::Lt(config) = choice {
            let ProjectionWeight::Fp8(w, _) = weight else {
                return Err(Error::invalid("FP8 cached channel weight"));
            };
            let q = api::zeros::<f8e4m3fn>(&[key.m, key.k])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            let out = api::zeros::<f32>(&[key.m, key.n])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            if !support
                .prepare_config(&q, w, &out, config)
                .map_err(device_error)?
            {
                return Ok(baseline);
            }
        }
        return Ok(choice);
    }
    let started = Instant::now();
    let accepted = match verify(device, support, weight, shape, choice, None) {
        Ok(Some(accepted)) => accepted,
        Ok(None) => return Ok(baseline),
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?key,?choice,%error,"FP8 new weight proof failed: native fallback");
            false
        }
    };
    remember(weight, key, choice, accepted)?;
    tracing::info!(target:"infer::tuning", ?shape, ?choice, accepted,
        elapsed_ms=started.elapsed().as_millis(), "FP8 cached choice weight proof");
    Ok(if accepted { choice } else { baseline })
}
fn finite_baseline_fixtures(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    shape: [usize; GEMM_RANK],
    session: &ResidentTuningSession,
) -> Result<bool> {
    // Both numerical fixtures must be finite even when every non-baseline candidate is rejected.
    for fixture in 1..FIXTURES {
        if !permits(session, shape) {
            return Ok(false);
        }
        let finite = Captured::new(
            device,
            support,
            weight,
            shape,
            Choice::baseline(shape[0]),
            fixture,
        )?;
        finite.run(device)?;
        let values = device.read_borrowed(&finite.output)?;
        bitwise(&values, &values)?;
    }
    Ok(true)
}
fn candidates(
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    shape: [usize; GEMM_RANK],
    base: &Captured,
    session: &ResidentTuningSession,
    vendor_candidates: bool,
) -> Option<Vec<Choice>> {
    let mut choices = Vec::new();
    for tile_m in ROW_TILES {
        for tile_n in COLUMN_TILES {
            choices.push(Choice::Native(NativeConfig { tile_m, tile_n }));
        }
    }
    if vendor_candidates && let ProjectionWeight::Fp8(w, _) = weight {
        if permits(session, shape) {
            match support.prepare_candidates(&base.codes, w, &base.output) {
                Ok(configs) => choices.extend(configs.into_iter().map(Choice::Lt)),
                Err(error) => {
                    tracing::warn!(target:"infer::tuning",?shape,%error,"FP8 Lt candidates unavailable: native candidates remain");
                }
            }
        } else {
            return None;
        }
    }
    Some(choices)
}

fn confirm_best(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    key: &Key,
    session: &ResidentTuningSession,
    base: &Captured,
    trials: &[Trial],
) -> Result<Option<Trial>> {
    let shape = [key.m, key.n, key.k];
    let mut ranked: Vec<&Trial> = trials.iter().filter(|trial| confirmed(trial)).collect();
    ranked.sort_by(|a, b| {
        b.cold
            .speedup
            .min(b.warm.speedup)
            .total_cmp(&a.cold.speedup.min(a.warm.speedup))
    });
    let baseline = Choice::baseline(key.m);
    for choice in ranked
        .into_iter()
        .map(|trial| trial.choice)
        .chain(once(baseline))
    {
        if !permits(session, shape) {
            return Ok(None);
        }
        let measured = (|| {
            let candidate = Captured::new(device, support, weight, shape, choice, 0)?;
            candidate.run(device)?;
            bitwise(
                &device.read_borrowed(&candidate.output)?,
                &device.read_borrowed(&base.output)?,
            )?;
            let measured = trial(device, base, &candidate, choice, true)?;
            bitwise(
                &device.read_borrowed(&candidate.output)?,
                &device.read_borrowed(&base.output)?,
            )?;
            Ok::<Trial, Error>(measured)
        })();
        match measured {
            Ok(measured) if choice == baseline || confirmed(&measured) => {
                return Ok(Some(measured));
            }
            Ok(_) => {
                tracing::info!(target:"infer::tuning",?shape,?choice,"FP8 reverse-order confirmation did not meet cold/warm threshold");
            }
            Err(error) => {
                remember(weight, key, choice, false)?;
                tracing::warn!(target:"infer::tuning",?shape,?choice,%error,"FP8 reverse-order confirmation rejected; next ranked candidate continues");
            }
        }
    }
    Ok(None)
}

enum SearchTrial {
    BudgetExhausted,
    Rejected,
    Measured(Trial),
}

fn search_trial(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    key: &Key,
    choice: Choice,
    session: &ResidentTuningSession,
    (base, expected): (&Captured, &[f32]),
) -> Result<SearchTrial> {
    let shape = [key.m, key.n, key.k];
    let valid = match verify(device, support, weight, shape, choice, Some(session)) {
        Ok(Some(valid)) => valid,
        Ok(None) => return Ok(SearchTrial::BudgetExhausted),
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,?choice,%error,"FP8 candidate rejected before measurement");
            false
        }
    };
    if !valid {
        return Ok(SearchTrial::Rejected);
    }
    remember(weight, key, choice, true)?;
    measure_verified_trial(device, support, weight, key, choice, base, expected)
}

fn measure_verified_trial(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    key: &Key,
    choice: Choice,
    base: &Captured,
    expected: &[f32],
) -> Result<SearchTrial> {
    let shape = [key.m, key.n, key.k];
    let candidate = match Captured::new(device, support, weight, shape, choice, 0) {
        Ok(candidate) => candidate,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,?choice,%error,"FP8 candidate capture rejected");
            return Ok(SearchTrial::Rejected);
        }
    };
    let measured = match trial(device, base, &candidate, choice, false) {
        Ok(measured) => measured,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,?choice,%error,"FP8 candidate measurement rejected; remaining candidates continue");
            return Ok(SearchTrial::Rejected);
        }
    };
    if let Err(error) = bitwise(&device.read_borrowed(&candidate.output)?, expected) {
        remember(weight, key, choice, false)?;
        tracing::warn!(target:"infer::tuning",?shape,?choice,%error,"FP8 candidate changed bits after measurement; remaining candidates continue");
        return Ok(SearchTrial::Rejected);
    }
    Ok(
        if times_valid(&measured.cold) && times_valid(&measured.warm) {
            SearchTrial::Measured(measured)
        } else {
            SearchTrial::Rejected
        },
    )
}

fn tune(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    key: Key,
    session: &ResidentTuningSession,
) -> Result<Option<Entry>> {
    let shape = [key.m, key.n, key.k];
    if !permits(session, shape) {
        return Ok(None);
    }
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile
            .opt_level
            .is_some_and(|level| level != JIT_OPT_LEVEL)
    {
        return Err(Error::invalid("FP8 tuning requires release/CUDA JIT O3"));
    }
    let baseline = Choice::baseline(key.m);
    let base = Captured::new(device, support, weight, shape, baseline, 0)?;
    base.run(device)?;
    let expected = device.read_borrowed(&base.output)?;
    bitwise(&expected, &expected)?;
    if !finite_baseline_fixtures(device, support, weight, shape, session)? {
        return Ok(None);
    }
    let Some(choices) = candidates(
        support,
        weight,
        shape,
        &base,
        session,
        key.vendor_candidates,
    ) else {
        return Ok(None);
    };
    let mut trials = Vec::new();
    for choice in choices {
        if choice == baseline {
            continue;
        }
        if !permits(session, shape) {
            return Ok(None);
        }
        match search_trial(
            device,
            support,
            weight,
            &key,
            choice,
            session,
            (&base, &expected),
        )? {
            SearchTrial::BudgetExhausted => return Ok(None),
            SearchTrial::Rejected => {}
            SearchTrial::Measured(measured) => trials.push(measured),
        }
    }
    let Some(confirmation) = confirm_best(device, support, weight, &key, session, &base, &trials)?
    else {
        return Ok(None);
    };
    let selected = confirmation.choice;
    let entry = Entry {
        key,
        selected,
        finite_bitwise: true,
        fixtures: FIXTURES,
        trials,
        confirmation,
    };
    if !entry.valid() {
        return Err(Error::invalid("FP8 invalid confirmation evidence"));
    }
    remember(weight, &entry.key, selected, true)?;
    Ok(Some(entry))
}
fn resolve(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    shape: [usize; GEMM_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    let baseline = Choice::baseline(shape[0]);
    let kernel_fingerprint = fingerprint(device, shape)?;
    let key = Key {
        hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
        driver_api: CudaDevice::driver_version()?,
        driver_build: driver_build(),
        kernel_fingerprint,
        library: support.library_identity(),
        contract: CONTRACT.into(),
        vendor_candidates: vendor_autotune_enabled(),
        m: shape[0],
        n: shape[1],
        k: shape[2],
    };
    let remembered = memo()
        .lock()
        .map_err(|_| Error::invariant("FP8 tuning cache poisoned"))?
        .get(&key)
        .copied();
    if let Some(choice) = remembered {
        return cached_gate(device, support, weight, &key, choice);
    }
    let path = super::table_path();
    let table = if key.persistent() {
        super::resolve_table(path.as_deref())?
    } else {
        None
    };
    let cached = table.as_ref().and_then(|table| {
        table
            .resident_fp8_gemm
            .iter()
            .find(|entry| entry.key == key)
    });
    let choice = if let Some(entry) = cached {
        if !entry.valid() {
            return Err(Error::invalid("FP8 invalid persistent evidence"));
        }
        tracing::info!(target:"infer::tuning",?key,?entry.selected,source="table-cache","FP8 GEMM choice");
        entry.selected
    } else {
        if !autotune {
            return Ok(baseline);
        }
        let Some(entry) = tune(device, support, weight, key.clone(), session)? else {
            return Ok(baseline);
        };
        let cached = key.persistent()
            && super::update_table(path.as_deref(), |table| {
                table.resident_fp8_gemm.retain(|old| old.key != entry.key);
                table.resident_fp8_gemm.push(entry.clone());
            });
        tracing::info!(target:"infer::tuning",?key,?entry.selected,cached,cold_speedup=entry.confirmation.cold.speedup,
            warm_speedup=entry.confirmation.warm.speedup,source="measured","FP8 quantization-inclusive paired A/B GEMM choice");
        tracing::debug!(target:"infer::tuning",evidence=%serde_json::to_string(&entry).map_err(device_error)?,"FP8 GEMM tuning evidence");
        entry.selected
    };
    memo()
        .lock()
        .map_err(|_| Error::invariant("FP8 tuning cache poisoned"))?
        .insert(key.clone(), choice);
    cached_gate(device, support, weight, &key, choice)
}
/// Lt's reduction order lacks a full model qualification. Only internal tests may search it;
/// production automatically tunes the qualified native geometry candidates.
#[cfg(not(test))]
pub(crate) const fn vendor_autotune_enabled() -> bool {
    false
}
#[cfg(test)]
pub(crate) fn vendor_autotune_enabled() -> bool {
    var("INFER_CUDA_FP8_LT_AUTOTUNE").is_ok_and(|value| value == "1" || value == "true")
}

/// Default prompt projection algorithm queried from the installed library for this device.
/// The vendor heuristic warms zero-workspace candidates before graph capture.
pub(crate) fn prefill_choice(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    input: &Tensor<f8e4m3fn>,
    weight: &Arc<Tensor<f8e4m3fn>>,
    shape: [usize; GEMM_RANK],
) -> Result<Option<Choice>> {
    let [rows, outputs, _] = shape;
    if rows < WIDE_PREFILL_LANES {
        return Ok(None);
    }
    let out = api::zeros::<f32>(&[rows, outputs])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let configs = support
        .prepare_candidates(input, weight, &out)
        .map_err(device_error)?;
    let choice = configs.first().copied().map(Choice::Lt);
    tracing::info!(target: "infer::tuning", ?shape, ?choice, "FP8 prefill cuBLASLt default");
    Ok(choice)
}

/// Resolve every loaded channel weight before its graph capture; failures affect only this weight.
pub(crate) fn select(
    device: &CudaDevice,
    support: &mut vendor_fp8::Support,
    weight: &ProjectionWeight,
    shape: [usize; GEMM_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> Choice {
    if let ProjectionWeight::Fp8(codes, scales) = weight {
        support.retain_channels(codes, scales);
    }
    match resolve(device, support, weight, shape, autotune, session) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"FP8 conservative baseline fallback");
            Choice::baseline(shape[0])
        }
    }
}
#[cfg(test)]
#[path = "../../tests/unit/fp8_gemm_tuning.rs"]
mod tests;
