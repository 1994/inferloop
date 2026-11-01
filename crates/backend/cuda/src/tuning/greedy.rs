//! Exact integer greedy reductions: per-device geometry search with independent confirmation.
use super::resident_gemm::ResidentTuningSession;
use super::{MAX_REPS, MIN_CONFIRM_SPEEDUP, MIN_REPS, REP_MS, WARMUP_MS};
use crate::tuning::preparation::driver_build;
use crate::{
    device::{CudaDevice, device_error},
    resident::greedy::{Greedy, TILE_SEARCH},
};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use serde::{Deserialize, Serialize};
use std::iter::once;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};

const BASELINE_TILE: usize = TILE_SEARCH[0];
const SCHEMA: &str = "f32-total-cmp:lowest-index:all-finite:two-stage-u32-bounded-offset-v3";
const SAMPLE_PERIOD: usize = 257;
const SAMPLE_CENTER: i16 = 128;
const MEDIAN_WEIGHT: f32 = 0.5;
const JIT_OPT_LEVEL: u8 = 3;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key {
    pub hardware: String,
    pub driver_version: u32,
    pub driver_build: Option<String>,
    pub kernel_fingerprint: String,
    pub rows: usize,
    pub vocabulary: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Times {
    pub baseline_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
}
impl Times {
    fn valid(&self) -> bool {
        self.baseline_ms.len() >= MIN_REPS
            && self.baseline_ms.len() == self.candidate_ms.len()
            && self
                .baseline_ms
                .iter()
                .chain(&self.candidate_ms)
                .all(|v| v.is_finite() && *v > 0.0)
    }
    fn speedup(&self) -> f32 {
        median(&self.baseline_ms) / median(&self.candidate_ms)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub tile: usize,
    pub candidate_first: bool,
    pub cold: Times,
    pub warm: Times,
}
impl Trial {
    fn valid(&self) -> bool {
        TILE_SEARCH.contains(&self.tile) && self.cold.valid() && self.warm.valid()
    }
    fn confirmed(&self) -> bool {
        self.valid()
            && self.cold.speedup() >= MIN_CONFIRM_SPEEDUP
            && self.warm.speedup() >= MIN_CONFIRM_SPEEDUP
    }
    fn score(&self) -> f32 {
        self.cold.speedup().min(self.warm.speedup())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub selected: usize,
    pub search: Vec<Trial>,
    pub confirmation: Trial,
}
impl Entry {
    fn valid(&self) -> bool {
        self.confirmation.valid()
            && self.confirmation.candidate_first
            && self.selected == self.confirmation.tile
            && (self.selected == BASELINE_TILE
                || self.confirmation.confirmed()
                    && self
                        .search
                        .iter()
                        .any(|t| !t.candidate_first && t.tile == self.selected && t.confirmed()))
    }
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn median(values: &[f32]) -> f32 {
    let mut values = values.to_vec();
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.is_empty() {
        return f32::NAN;
    }
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * MEDIAN_WEIGHT
    } else {
        values[middle]
    }
}
/// Kernel code, compiler options and driver identity are immutable in a live context.
/// Keep the full identity; only its repeated construction is memoized, not validation.
fn key(device: &CudaDevice, rows: usize, vocabulary: usize) -> Result<Key> {
    type Identity = (usize, usize, usize, usize);
    type Identities = Mutex<BTreeMap<Identity, Key>>;
    static KEYS: OnceLock<Identities> = OnceLock::new();
    let keys = KEYS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let identity = (
        device.ordinal(),
        device.stream.cu_stream() as usize,
        rows,
        vocabulary,
    );
    let cached = keys
        .lock()
        .map_err(|_| Error::invariant("greedy identity cache poisoned"))?
        .get(&identity)
        .cloned();
    if let Some(key) = cached {
        return Ok(key);
    }
    let key = build_key(device, rows, vocabulary)?;
    keys.lock()
        .map_err(|_| Error::invariant("greedy identity cache poisoned"))?
        .insert(identity, key.clone());
    Ok(key)
}

fn build_key(device: &CudaDevice, rows: usize, vocabulary: usize) -> Result<Key> {
    let input = api::meta::<f32>(&[rows, vocabulary])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut identities = Vec::new();
    for tile in TILE_SEARCH {
        let mut greedy = Greedy::new(device, rows, vocabulary, tile)?;
        identities.push(greedy.fingerprint(device, &input)?);
    }
    Ok(Key {
        hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
        driver_version: CudaDevice::driver_version()?,
        driver_build: driver_build(),
        kernel_fingerprint: format!("{SCHEMA}:{}", identities.join(":")),
        rows,
        vocabulary,
    })
}
/// Select before capture; incomplete or rejected measurements retain a portable baseline.
pub(crate) fn select(
    device: &CudaDevice,
    rows: usize,
    vocabulary: usize,
    autotune: bool,
    session: &ResidentTuningSession,
) -> usize {
    match resolve(device, rows, vocabulary, autotune, session) {
        Ok(tile) => tile,
        Err(error) => {
            tracing::warn!(target:"infer::tuning", rows, vocabulary, %error, "greedy reduction baseline fallback");
            BASELINE_TILE
        }
    }
}
fn resolve(
    device: &CudaDevice,
    rows: usize,
    vocabulary: usize,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<usize> {
    let key = key(device, rows, vocabulary)?;
    let cached = memo()
        .lock()
        .map_err(|_| Error::invariant("greedy tuning cache poisoned"))?
        .get(&key)
        .filter(|entry| entry.valid())
        .cloned();
    if let Some(entry) = cached {
        return Ok(entry.selected);
    }
    let path = super::table_path();
    let table = if key.driver_build.is_some() {
        super::resolve_table(path.as_deref())?
    } else {
        None
    };
    if let Some(entry) = table.as_ref().and_then(|t| cached_entry(&t.greedy, &key)) {
        memo()
            .lock()
            .map_err(|_| Error::invariant("greedy tuning cache poisoned"))?
            .insert(key, entry.clone());
        return Ok(entry.selected);
    }
    if !autotune || !session.permits_greedy_search() {
        return Ok(BASELINE_TILE);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(BASELINE_TILE);
    };
    let cached = key.driver_build.is_some()
        && super::update_table(path.as_deref(), |table| {
            table.greedy.retain(|old| old.key != key);
            table.greedy.push(entry.clone());
        });
    tracing::info!(target:"infer::tuning", ?key, tile=entry.selected, cached,
        cold_speedup=entry.confirmation.cold.speedup(), warm_speedup=entry.confirmation.warm.speedup(),
        "greedy reduction selected");
    memo()
        .lock()
        .map_err(|_| Error::invariant("greedy tuning cache poisoned"))?
        .insert(key, entry.clone());
    Ok(entry.selected)
}

struct Captured {
    graph: Graph,
    greedy: Greedy,
}
impl Captured {
    fn new(
        device: &CudaDevice,
        input: &Tensor<f32>,
        shape: [usize; 2],
        tile: usize,
    ) -> Result<Self> {
        let mut greedy = Greedy::new(device, shape[0], shape[1], tile)?;
        let graph = Graph::scope(&device.stream, |scope| greedy.record(scope, input))
            .map_err(device_error)?;
        Ok(Self { graph, greedy })
    }
    fn verify(&self, device: &CudaDevice, expected: &[u32]) -> Result<()> {
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        if self.greedy.read(device)? != expected {
            return Err(Error::invalid("greedy candidate changed token decisions"));
        }
        Ok(())
    }
}
fn measure(
    device: &CudaDevice,
    base: &Captured,
    candidate: &Captured,
    clear_l2: bool,
    candidate_first: bool,
) -> Result<Times> {
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
    let (first, second) = if candidate_first {
        (candidate, base)
    } else {
        (base, candidate)
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(first), |_| run(second))
        .map_err(device_error)?;
    let times = if candidate_first {
        Times {
            baseline_ms: b.times_ms().to_vec(),
            candidate_ms: a.times_ms().to_vec(),
        }
    } else {
        Times {
            baseline_ms: a.times_ms().to_vec(),
            candidate_ms: b.times_ms().to_vec(),
        }
    };
    if !times.valid() {
        return Err(Error::invalid("greedy paired timing samples"));
    }
    Ok(times)
}
struct Search<'a> {
    device: &'a CudaDevice,
    input: &'a Tensor<f32>,
    shape: [usize; 2],
    base: &'a Captured,
    expected: &'a [u32],
}
impl Search<'_> {
    fn trial(&self, tile: usize, candidate_first: bool) -> Result<Trial> {
        let candidate = Captured::new(self.device, self.input, self.shape, tile)?;
        candidate.verify(self.device, self.expected)?;
        let trial = Trial {
            tile,
            candidate_first,
            cold: measure(self.device, self.base, &candidate, true, candidate_first)?,
            warm: measure(self.device, self.base, &candidate, false, candidate_first)?,
        };
        candidate.verify(self.device, self.expected)?;
        Ok(trial)
    }
}
fn fixture(rows: usize, vocabulary: usize) -> Result<(Vec<f32>, Vec<u32>)> {
    let count = rows
        .checked_mul(vocabulary)
        .ok_or_else(|| Error::invalid("greedy fixture dimensions"))?;
    let values: Vec<f32> = (0..count)
        .map(|i| f32::from(i16::try_from(i % SAMPLE_PERIOD).unwrap_or_default() - SAMPLE_CENTER))
        .collect();
    let expected = values
        .chunks_exact(vocabulary)
        .map(|row| infer_workloads::sample(row, &infer_ir::Sampling::default(), 0, 0))
        .collect::<Result<Vec<_>>>()?;
    Ok((values, expected))
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile
            .opt_level
            .is_some_and(|level| level != JIT_OPT_LEVEL)
    {
        return Ok(None);
    }
    let (values, expected) = fixture(key.rows, key.vocabulary)?;
    let shape = [key.rows, key.vocabulary];
    let input = device.upload(values, &shape)?;
    let base = Captured::new(device, &input, shape, BASELINE_TILE)?;
    base.verify(device, &expected)?;
    let search = Search {
        device,
        input: &input,
        shape,
        base: &base,
        expected: &expected,
    };
    let mut trials = Vec::new();
    for tile in TILE_SEARCH {
        if !session.permits_greedy_search() {
            return Ok(None);
        }
        match search.trial(tile, false) {
            Ok(trial) => trials.push(trial),
            Err(error) => {
                tracing::warn!(target:"infer::tuning", tile, %error, "greedy candidate rejected");
            }
        }
    }
    let mut ranked: Vec<_> = trials
        .iter()
        .filter(|t| t.tile != BASELINE_TILE && t.confirmed())
        .collect();
    ranked.sort_by(|a, b| b.score().total_cmp(&a.score()));
    for tile in ranked
        .into_iter()
        .map(|t| t.tile)
        .chain(once(BASELINE_TILE))
    {
        if !session.permits_greedy_search() {
            return Ok(None);
        }
        match search.trial(tile, true) {
            Ok(confirmation) if tile == BASELINE_TILE || confirmation.confirmed() => {
                return Ok(Some(Entry {
                    key,
                    selected: tile,
                    search: trials,
                    confirmation,
                }));
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(target:"infer::tuning", tile, %error, "greedy confirmation rejected");
            }
        }
    }
    Ok(None)
}

fn cached_entry<'a>(entries: &'a [Entry], key: &Key) -> Option<&'a Entry> {
    entries
        .iter()
        .find(|entry| entry.key == *key && entry.valid())
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_greedy.rs"]
mod tests;
