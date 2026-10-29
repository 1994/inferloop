//! Separate F32-cache input contract; reuses the same BF16 AOT architecture cores.
mod capture;
mod reference;
use super::{
    Choice, Key, Shape, Times,
    mma::{self, Case, Trial},
};
use crate::device::platform::driver_build;
use crate::tuning::MAX_REPS;
use crate::tuning::MIN_REPS;
use crate::tuning::REP_MS;
use crate::tuning::WARMUP_MS;
use crate::tuning::resolve_table;
use crate::tuning::table_path;
use crate::tuning::update_table;
use crate::{
    device::{CudaDevice, device_error},
    resident::attention_mma::catalog::{self, Variant},
    tuning::resident_gemm::ResidentTuningSession,
};
use capture::{Captured, Proof};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use reference::Fixture;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};
const SCHEMA: &str = "fa4-bf16-from-f32-kv:full-f64-normalized1e-2-l2-2e-2:permuted-pages:long-rows:original-append:cold-warm-ab-ba-v1";
const PROFILE: &str = "f32-cache-bf16-qkv:f32-acc:bf16-output:gpu-seqlen-v1";
const SHA256_HEX_LEN: usize = 64;
const HISTORY_CASES: usize = 3;
const HISTORY_ROWS: usize = 3;
const HISTORY_HEADS: usize = 2;
const ORDERINGS: usize = 2;
const VERIFY_STAGES: usize = 2;
const CUDA_OPT_LEVEL: u8 = 3;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub architecture: String,
    pub profile: String,
    pub selected: Choice,
    pub trials: Vec<Trial>,
    pub max_relative_l2: f64,
    pub history_f64_rows: usize,
}
impl Entry {
    pub(crate) fn valid(&self) -> bool {
        let s = self.key.shape;
        let variants = Variant::candidates(&self.architecture);
        self.key.schema == SCHEMA
            && self.profile == PROFILE
            && catalog::supported(
                &self.architecture,
                s.heads,
                s.kv_heads,
                s.dim,
                s.lanes,
                s.window,
            )
            && serde_json::from_str::<serde_json::Value>(&self.key.hardware)
                .ok()
                .and_then(|v| {
                    v.get("architecture")
                        .and_then(|a| a.as_str())
                        .map(str::to_owned)
                })
                .is_some_and(|a| a == self.architecture)
            && self.key.fingerprint.len() == SHA256_HEX_LEN
            && self.key.fingerprint.bytes().all(|b| b.is_ascii_hexdigit())
            && self.key.driver_build.as_ref().is_none_or(|v| !v.is_empty())
            && self.trials.len() == variants.len()
            && self
                .trials
                .iter()
                .zip(variants)
                .all(|(t, v)| t.variant == *v && t.valid(s))
            && self.history_f64_rows
                == variants.len()
                    * HISTORY_CASES
                    * HISTORY_ROWS
                    * HISTORY_HEADS
                    * ORDERINGS
                    * VERIFY_STAGES
            && self.max_relative_l2.is_finite()
            && (0.0..reference::MAX_RELATIVE_L2).contains(&self.max_relative_l2)
            && (self.selected == Choice::Simt
                || self
                    .trials
                    .iter()
                    .any(|t| self.selected == Choice::Mma { variant: t.variant } && t.accepted()))
    }
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
pub(crate) fn select(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Choice {
    match resolve(device, shape, autotune, session) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"F32 KV MMA unavailable: F32 SIMT fallback");
            Choice::Simt
        }
    }
}
fn key(device: &CudaDevice, shape: Shape) -> Result<Key> {
    let driver_build = driver_build();
    Ok(Key {
        hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
        driver_api: CudaDevice::driver_version()?,
        driver_build,
        fingerprint: capture::fingerprint(device, shape)?,
        schema: SCHEMA.into(),
        shape,
    })
}
fn resolve(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    let architecture = &device.profile()?.architecture;
    if !catalog::supported(
        architecture,
        shape.heads,
        shape.kv_heads,
        shape.dim,
        shape.lanes,
        shape.window,
    ) {
        return Ok(Choice::Simt);
    }
    let key = key(device, shape)?;
    if let Some(e) = memo()
        .lock()
        .map_err(|_| Error::invariant("F32 MMA memo poisoned"))?
        .get(&key)
    {
        return Ok(e.selected);
    }
    let path = table_path();
    let table = if key.driver_build.is_some() {
        resolve_table(path.as_deref())?
    } else {
        None
    };
    if let Some(e) = table.and_then(|t| {
        t.resident_mma_f32_attention
            .into_iter()
            .find(|e| e.key == key)
    }) {
        if !e.valid() {
            return Err(Error::invalid(
                "incomplete F32 KV numerical/cluster evidence",
            ));
        }
        let choice = e.selected;
        memo()
            .lock()
            .map_err(|_| Error::invariant("F32 MMA memo poisoned"))?
            .insert(key, e);
        tracing::info!(target:"infer::tuning",?shape,?choice,profile=PROFILE,source="table-cache","F32 KV MMA selected");
        return Ok(choice);
    }
    // An unmeasured shape takes the architecture's first FA4 variant: the scalar path is several
    // times slower, and a source-fingerprint change must not silently fall back to it.
    let untuned = || {
        let choice = Variant::candidates(architecture)
            .first()
            .map_or(Choice::Simt, |&variant| Choice::Mma { variant });
        tracing::info!(target:"infer::tuning",?shape,?choice,profile=PROFILE,source="default","F32 KV MMA selected");
        choice
    };
    if !autotune || !session.permits_attention_search() {
        return Ok(untuned());
    }
    let Some(e) = tune(device, key.clone(), session)? else {
        return Ok(untuned());
    };
    let cached = key.driver_build.is_some()
        && update_table(path.as_deref(), |table| {
            table.resident_mma_f32_attention.retain(|e| e.key != key);
            table.resident_mma_f32_attention.push(e.clone());
        });
    let choice = e.selected;
    tracing::info!(target:"infer::tuning",?shape,?choice,cached,profile=PROFILE,max_relative_l2=e.max_relative_l2,history_f64_rows=e.history_f64_rows,"F32 KV MMA hardware choice");
    memo()
        .lock()
        .map_err(|_| Error::invariant("F32 MMA memo poisoned"))?
        .insert(key, e);
    Ok(choice)
}
fn measure(
    device: &CudaDevice,
    base: &Captured,
    candidate: &Captured,
    cold: bool,
    reverse: bool,
) -> Result<Times> {
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: cold,
    };
    let (left, right) = if reverse {
        (candidate, base)
    } else {
        (base, candidate)
    };
    let run = |c: &Captured| {
        c.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(left), |_| run(right))
        .map_err(device_error)?;
    let (a, b) = if reverse { (b, a) } else { (a, b) };
    let t = Times {
        baseline_ms: a.times_ms().to_vec(),
        candidate_ms: b.times_ms().to_vec(),
    };
    if !t.valid() {
        return Err(Error::invalid("F32 MMA paired samples"));
    }
    Ok(t)
}
const fn merge(into: &mut Proof, p: &Proof) {
    into.max_error = into.max_error.max(p.max_error);
    into.relative_l2 = into.relative_l2.max(p.relative_l2);
    into.rows += p.rows;
    into.elements += p.elements;
}
fn cases(
    device: &CudaDevice,
    shape: Shape,
    variant: Variant,
    reverse: bool,
    session: &ResidentTuningSession,
) -> Result<Option<(Vec<Case>, Proof)>> {
    let mut result = Vec::new();
    let mut proof = Proof::default();
    let capacity = mma::capacity(shape);
    for history in mma::histories(shape, capacity) {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let fixture = Fixture::history(shape, history, capacity)?;
        let base = Captured::new(device, &fixture, Choice::Simt)?;
        let candidate = Captured::new(device, &fixture, Choice::Mma { variant })?;
        merge(&mut proof, &candidate.verify(device, &fixture, false)?);
        result.push(Case {
            history,
            cold: measure(device, &base, &candidate, true, reverse)?,
            warm: measure(device, &base, &candidate, false, reverse)?,
        });
        merge(&mut proof, &candidate.verify(device, &fixture, false)?);
    }
    Ok(Some((result, proof)))
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let options = CompileOptions::default();
    if cfg!(debug_assertions)
        || options.device_debug
        || options.opt_level.is_some_and(|l| l != CUDA_OPT_LEVEL)
    {
        return Err(Error::invalid("F32 KV MMA tuning requires release CUDA O3"));
    }
    let architecture = device.profile()?.architecture.clone();
    let mut trials = Vec::new();
    let mut selected = Choice::Simt;
    let mut best = 1.0;
    let mut overall = Proof::default();
    let mut history_rows = 0;
    for &variant in Variant::candidates(&architecture) {
        let mut proof = Proof::default();
        for i in 0..reference::FIXTURES {
            if !session.permits_attention_search() {
                return Ok(None);
            }
            let fixture = Fixture::short(key.shape, i)?;
            let candidate = Captured::new(device, &fixture, Choice::Mma { variant })?;
            merge(&mut proof, &candidate.verify(device, &fixture, true)?);
        }
        let elements = proof.elements;
        let Some((forward, fp)) = cases(device, key.shape, variant, false, session)? else {
            return Ok(None);
        };
        let Some((reverse, rp)) = cases(device, key.shape, variant, true, session)? else {
            return Ok(None);
        };
        history_rows += fp.rows + rp.rows;
        merge(&mut proof, &fp);
        merge(&mut proof, &rp);
        let trial = Trial {
            variant,
            fixtures: reference::FIXTURES,
            elements,
            max_f64_error: proof.max_error,
            forward,
            reverse,
        };
        if trial.accepted() && trial.score() > best {
            best = trial.score();
            selected = Choice::Mma { variant };
        }
        trials.push(trial);
        merge(&mut overall, &proof);
    }
    let entry = Entry {
        key,
        architecture,
        profile: PROFILE.into(),
        selected,
        trials,
        max_relative_l2: overall.relative_l2,
        history_f64_rows: history_rows,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete F32 KV MMA evidence"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_mma_f32.rs"]
mod tests;
