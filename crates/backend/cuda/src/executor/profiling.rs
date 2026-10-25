//! Opt-in execution wall-time evidence; async phases include suspension and device waits.
//! No token contents are recorded. These durations are not thread CPU time.
use super::CudaExecutor;
use crate::device::CudaDevice;
use infer_core::StateId;
use infer_ir::{ExecutionInput, ExecutionTask};
use std::collections::BTreeMap;
use std::env::var_os;
use std::fs::OpenOptions;
use std::future::Future;
use std::mem::take;
use std::sync::{Arc, Mutex};
use std::{
    cell::RefCell,
    io::Write,
    path::PathBuf,
    sync::OnceLock,
    time::{Duration, Instant},
};

/// Set before process startup to append task grouping and slot occupancy as JSONL.
const PROFILE_ENV: &str = "INFER_CUDA_EXECUTION_PROFILE";
static OUTPUT: OnceLock<Option<PathBuf>> = OnceLock::new();

pub(super) type PhaseRecords = Arc<Mutex<Vec<(&'static str, Duration)>>>;
thread_local! {
    static PHASES: RefCell<Option<PhaseRecords>> = const { RefCell::new(None) };
}
/// A poll-local binding; no thread-local borrow or lock survives a suspension.
pub(super) struct PhaseScope {
    previous: Option<PhaseRecords>,
}
impl PhaseScope {
    pub(super) fn records() -> PhaseRecords {
        Arc::default()
    }
    pub(super) fn bind(records: Option<PhaseRecords>) -> Self {
        Self {
            previous: PHASES.with(|slot| slot.replace(records)),
        }
    }
    pub(super) fn take_records(records: Option<PhaseRecords>) -> Vec<(&'static str, Duration)> {
        let Some(records) = records else {
            return Vec::new();
        };
        match records.lock() {
            Ok(mut rows) => take(&mut *rows),
            Err(error) => {
                tracing::error!(target: "infer::profile", %error, "execution phase recorder poisoned");
                Vec::new()
            }
        }
    }
}
impl Drop for PhaseScope {
    fn drop(&mut self) {
        PHASES.with(|slot| slot.replace(self.previous.take()));
    }
}
fn record(records: Option<PhaseRecords>, name: &'static str, started: Option<Instant>) {
    if let (Some(records), Some(started)) = (records, started) {
        match records.lock() {
            Ok(mut rows) => rows.push((name, started.elapsed())),
            Err(error) => {
                tracing::error!(target: "infer::profile", %error, "execution phase recorder poisoned");
            }
        }
    }
}
pub(super) fn phase<T>(name: &'static str, body: impl FnOnce() -> T) -> T {
    let records = PHASES.with(|slot| slot.borrow().clone());
    let started = records.as_ref().map(|_| Instant::now());
    let result = body();
    record(records, name, started);
    result
}
pub(super) async fn phase_async<T>(name: &'static str, body: impl Future<Output = T>) -> T {
    let records = PHASES.with(|slot| slot.borrow().clone());
    let started = records.as_ref().map(|_| Instant::now());
    let result = body.await;
    record(records, name, started);
    result
}

pub(super) struct StepProfile {
    path: &'static PathBuf,
    started: Instant,
    inputs: Vec<serde_json::Value>,
    states: Vec<StateId>,
}

impl StepProfile {
    pub(super) fn start(backend: &CudaExecutor, tasks: &[ExecutionTask]) -> Option<Self> {
        let path = output_path()?;
        let inputs = tasks
            .iter()
            .map(|task| {
                let state = backend.states.get(&task.state);
                let position = state.map_or(0, |state| state.history.len());
                let kind = match &task.tokens {
                    ExecutionInput::Decode { .. } => "decode",
                    ExecutionInput::Prefill { .. } => "prefill",
                    ExecutionInput::Full(_) => "full",
                };
                serde_json::json!({
                    "request": task.request,
                    "state": task.state,
                    "kind": kind,
                    "position": position,
                    "input_tokens": task.tokens.delta(position).map_or(0, <[u32]>::len),
                    "capacity": state.map(|state| state.capacity),
                    "slot_before": state.and_then(|state| state.slot.as_ref().map(|lease| lease.slot)),
                })
            })
            .collect();
        Some(Self {
            path,
            started: Instant::now(),
            inputs,
            states: tasks.iter().map(|task| task.state).collect(),
        })
    }

    pub(super) fn finish(
        self,
        backend: &CudaExecutor,
        outputs: Option<&[infer_ir::TaskOutput]>,
        phases: Vec<(&'static str, Duration)>,
    ) {
        let elapsed_us = self.started.elapsed().as_micros();
        let mut by_phase: BTreeMap<&'static str, u64> = BTreeMap::new();
        for (name, elapsed) in phases {
            *by_phase.entry(name).or_default() += u64::try_from(elapsed.as_micros()).unwrap_or(0);
        }
        let slots: Vec<_> = self
            .states
            .iter()
            .map(|state| {
                backend
                    .states
                    .get(state)
                    .and_then(|state| state.slot.as_ref().map(|lease| lease.slot))
            })
            .collect();
        let line = serde_json::json!({
            "execution_wall_us": elapsed_us,
            "phase_wall_us": by_phase,
            "tasks": self.inputs,
            "slots_after": slots,
            "pool_width": backend.slots.as_ref().map(super::SlotPool::width),
            "worker_prompt_width": backend.worker.as_ref().map(|worker| worker.program.prefill_width()),
            "pool_disabled": backend.slots_disabled,
            "draft_pool_width": backend.draft_slots.as_ref().map(super::SlotPool::width),
            "successful": outputs.is_some(),
            "accepted_tokens": outputs.map(|rows| rows.iter().map(|row| row.output.tokens.len()).collect::<Vec<_>>()),
        });
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path)
            .and_then(|mut file| writeln!(file, "{line}"));
        if let Err(error) = result {
            eprintln!("CUDA execution profile write failed: {error}");
        }
    }
}

/// Keep fallback causes alongside grouping evidence instead of silently losing batching.
pub(super) fn pool_failure(kind: &str, width: usize, capacity: usize, error: &infer_core::Error) {
    let Some(path) = output_path() else {
        return;
    };
    let line = serde_json::json!({"pool_kind": kind, "pool_creation_failed": error.to_string(), "width": width, "capacity": capacity});
    if let Err(error) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| writeln!(file, "{line}"))
    {
        eprintln!("CUDA execution profile write failed: {error}");
    }
}

fn output_path() -> Option<&'static PathBuf> {
    OUTPUT
        .get_or_init(|| {
            var_os(PROFILE_ENV)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .as_ref()
}

/// Startup allocation evidence shares the opt-in host profile and never synchronizes the GPU.
pub(super) fn memory(stage: &str, width: usize, device: &CudaDevice) {
    let Some(path) = output_path() else {
        return;
    };
    let (Ok((free, total)), Ok(reusable)) = (device.memory_info(), device.pool_reclaimable_bytes())
    else {
        return;
    };
    let line = serde_json::json!({
        "memory_stage": stage,
        "width": width,
        "driver_free_bytes": free,
        "pool_unused_bytes": reusable,
        "allocatable_bytes": free.saturating_add(reusable).min(total),
    });
    if let Err(error) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| writeln!(file, "{line}"))
    {
        eprintln!("CUDA memory profile write failed: {error}");
    }
}
