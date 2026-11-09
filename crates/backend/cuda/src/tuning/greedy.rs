//! Exact integer greedy reductions: per-device geometry search with independent confirmation.
use super::resident_gemm::{MeasurementOrder, PairedTimes, ResidentTuningSession, paired_times};
use super::search::{self, Evidence, FormerTrial, Protocol, Record};
use super::{MAX_REPS, MIN_REPS, REP_MS, WARMUP_MS};
use crate::tuning::preparation::driver_build;
use crate::{
    device::{CudaDevice, device_error},
    resident::greedy::{Greedy, TILE_SEARCH},
};
use infer_core::{Error, Result};
use infer_cuda_kernels::jit::JitOptions;
use infer_cuda_kernels::runtime::Timing;
use infer_cuda_kernels::runtime::{Allocation, Graph};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};

const BASELINE_TILE: usize = TILE_SEARCH[0];
const SCHEMA: &str = "f32-total-cmp:lowest-index:all-finite:two-stage-u32-bounded-offset-v3";
const SAMPLE_PERIOD: usize = 257;
const SAMPLE_CENTER: i16 = 128;
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
/// The family's records in the shared table section.
pub const FAMILY: &str = "greedy";
/// A reduction tile measured against the baseline tile.
pub type Trial = search::Trial<usize>;
pub type Entry = Record<Key, usize>;
/// Every searched tile against the first.
fn protocol() -> Protocol<usize> {
    Protocol {
        baseline: BASELINE_TILE,
        candidates: TILE_SEARCH[1..].to_vec(),
    }
}
impl Entry {
    fn valid(&self) -> bool {
        self.evidence.sampled(MIN_REPS)
            && self
                .evidence
                .valid(&protocol(), |tile| TILE_SEARCH.contains(&tile))
    }
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[derive(Deserialize)]
struct LegacyTrial {
    tile: usize,
    #[serde(flatten)]
    trial: FormerTrial,
}
#[derive(Deserialize)]
struct LegacyEntry {
    key: Key,
    selected: usize,
    search: Vec<LegacyTrial>,
    confirmation: LegacyTrial,
}

/// A record from the family's former table section in the shared form; one that does not
/// convert stays as stored. Every former trial matched the reference tokens, and its search
/// kept no rejections.
pub(super) fn legacy(stored: Value) -> Value {
    convert(&stored).unwrap_or(stored)
}

fn convert(stored: &Value) -> Option<Value> {
    let legacy = LegacyEntry::deserialize(stored).ok()?;
    let trial = |legacy: LegacyTrial| legacy.trial.into_trial(legacy.tile);
    let entry = Entry {
        key: legacy.key,
        evidence: Evidence {
            selected: legacy.selected,
            exact_candidates_verified: true,
            trials: legacy
                .search
                .into_iter()
                .map(trial)
                .collect::<Option<_>>()?,
            rejected: None,
            confirmation: trial(legacy.confirmation)?,
            max_error: None,
        },
    };
    serde_json::to_value(entry).ok()
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
        device.tile().stream_id(),
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
    let mut identities = Vec::new();
    for tile in TILE_SEARCH {
        let greedy = Greedy::new(device, rows, vocabulary, tile)?;
        identities.push(greedy.fingerprint(device)?);
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
        .map(|entry| entry.evidence.selected);
    if let Some(selected) = cached {
        return Ok(selected);
    }
    let path = super::table_path();
    let table = if key.driver_build.is_some() {
        super::resolve_table(path.as_deref())?
    } else {
        None
    };
    if let Some(entry) = table
        .as_ref()
        .and_then(|table| search::lookup(table, FAMILY, &key, Entry::valid))
    {
        let selected = entry.evidence.selected;
        memo()
            .lock()
            .map_err(|_| Error::invariant("greedy tuning cache poisoned"))?
            .insert(key, entry);
        return Ok(selected);
    }
    if !autotune || !session.permits_greedy_search() {
        return Ok(BASELINE_TILE);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(BASELINE_TILE);
    };
    let cached = key.driver_build.is_some() && search::store(path.as_deref(), FAMILY, &entry);
    let evidence = &entry.evidence;
    tracing::info!(target:"infer::tuning", ?key, tile=evidence.selected, cached,
        cold_speedup=evidence.confirmation.cold.speedup, warm_speedup=evidence.confirmation.warm.speedup,
        "greedy reduction selected");
    let selected = evidence.selected;
    memo()
        .lock()
        .map_err(|_| Error::invariant("greedy tuning cache poisoned"))?
        .insert(key, entry);
    Ok(selected)
}

struct Captured {
    graph: Graph,
    greedy: Greedy,
}
impl Captured {
    fn new(
        device: &CudaDevice,
        input: &Allocation<f32>,
        shape: [usize; 2],
        tile: usize,
    ) -> Result<Self> {
        let greedy = Greedy::new(device, shape[0], shape[1], tile)?;
        let graph = device.tile().capture(|scope| greedy.record(scope, input))?;
        Ok(Self { graph, greedy })
    }
    fn verify(&self, device: &CudaDevice, expected: &[u32]) -> Result<()> {
        device.tile().sync(self.graph.launch())?;
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
    order: MeasurementOrder,
) -> Result<PairedTimes> {
    let options = Timing {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2,
    };
    let run = |graph: &Captured| device.tile().sync(graph.graph.launch());
    let candidate_first = order == MeasurementOrder::CandidateFirst;
    let (first, second) = if candidate_first {
        (candidate, base)
    } else {
        (base, candidate)
    };
    let (a, b) = device
        .tile()
        .time_paired(&options, || run(first), || run(second))
        .map_err(device_error)?;
    let (baseline, candidate) = if candidate_first { (b, a) } else { (a, b) };
    paired_times(baseline.times_ms(), candidate.times_ms())
        .filter(|times| times.baseline_ms.len() >= MIN_REPS)
        .ok_or_else(|| Error::invalid("greedy paired timing samples"))
}
struct Search<'a> {
    device: &'a CudaDevice,
    input: &'a Allocation<f32>,
    shape: [usize; 2],
    base: &'a Captured,
    expected: &'a [u32],
}
impl Search<'_> {
    fn trial(&self, tile: usize, order: MeasurementOrder) -> Result<Trial> {
        let candidate = Captured::new(self.device, self.input, self.shape, tile)?;
        candidate.verify(self.device, self.expected)?;
        let trial = Trial {
            choice: tile,
            order,
            cold: measure(self.device, self.base, &candidate, true, order)?,
            warm: measure(self.device, self.base, &candidate, false, order)?,
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
    let compile = JitOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug()
        || compile
            .opt_level()
            .is_some_and(|level| level != JIT_OPT_LEVEL)
    {
        return Ok(None);
    }
    let (values, expected) = fixture(key.rows, key.vocabulary)?;
    let shape = [key.rows, key.vocabulary];
    let input = device.runtime().upload(values)?;
    let base = Captured::new(device, &input, shape, BASELINE_TILE)?;
    base.verify(device, &expected)?;
    let search = Search {
        device,
        input: &input,
        shape,
        base: &base,
        expected: &expected,
    };
    let evidence = search::run(
        &protocol(),
        || session.permits_greedy_search(),
        |tile, order| search.trial(tile, order),
        |tile, error| {
            tracing::warn!(target:"infer::tuning", tile, %error, "greedy candidate rejected");
            Ok(())
        },
    )?;
    Ok(evidence.map(|evidence| Entry { key, evidence }))
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_greedy.rs"]
mod tests;
