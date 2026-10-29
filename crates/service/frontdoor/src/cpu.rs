//! Fixed CPU workers with admission bytes, queue bounds and cancellable asynchronous replies.
use crate::ingress::Pending;
use infer_core::placement::ThreadPlacement;
use infer_core::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::thread::available_parallelism;
use std::{
    sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::AtomicUsize,
    sync::atomic::Ordering, sync::mpsc, time::Duration, time::Instant,
};
use tokio::sync::oneshot;
use tokio::time::timeout as time_timeout;

/// CPU threads reserved for non-inference work when sizing the default worker pool.
const DEFAULT_WORKER_RESERVE: usize = 7;
/// Upper bound on the auto-sized default worker pool.
const MAX_DEFAULT_WORKERS: usize = 32;
/// Default staging byte budget for one CPU pool.
const DEFAULT_MAX_STAGING_BYTES: usize = 64 << 20;
/// Default per-lane CPU job timeout, in milliseconds.
const DEFAULT_JOB_TIMEOUT_MS: u64 = 30_000;
/// Default control-command acknowledgement timeout, in milliseconds.
const DEFAULT_COMMAND_TIMEOUT_MS: u64 = 5_000;
/// Default device control timeout, in milliseconds.
const DEFAULT_DEVICE_CONTROL_TIMEOUT_MS: u64 = 250;
/// Maximum number of CPU workers accepted by configuration validation.
const MAX_WORKERS: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CpuConfig {
    pub placement: ThreadPlacement,
    pub workers: usize,
    pub max_jobs: usize,
    pub max_bytes: usize,
    pub job_timeout_ms: u64,
    pub command_timeout_ms: u64,
    pub device_control_timeout_ms: u64,
}
impl Default for CpuConfig {
    fn default() -> Self {
        let workers = available_parallelism()
            .map_or(2, NonZeroUsize::get)
            .saturating_sub(DEFAULT_WORKER_RESERVE)
            .clamp(1, MAX_DEFAULT_WORKERS);
        Self {
            placement: ThreadPlacement::default(),
            workers,
            max_jobs: workers * 2,
            max_bytes: DEFAULT_MAX_STAGING_BYTES,
            job_timeout_ms: DEFAULT_JOB_TIMEOUT_MS,
            command_timeout_ms: DEFAULT_COMMAND_TIMEOUT_MS,
            device_control_timeout_ms: DEFAULT_DEVICE_CONTROL_TIMEOUT_MS,
        }
    }
}
impl CpuConfig {
    /// # Errors
    /// Rejects zero limits or unbounded/oversubscribed worker configuration.
    pub fn validate(&self) -> Result<()> {
        if self.workers == 0
            || self.workers > MAX_WORKERS
            || self.max_jobs < self.workers
            || self.max_bytes == 0
            || self.job_timeout_ms == 0
            || self.command_timeout_ms == 0
            || self.device_control_timeout_ms == 0
        {
            return Err(Error::invalid("invalid CPU worker configuration"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct CpuInspection {
    pub workers: usize,
    pub retained_jobs: usize,
    pub retained_bytes: usize,
    pub active_workers: usize,
    pub peak_workers: usize,
}
#[derive(Default)]
struct Usage {
    jobs: usize,
    bytes: usize,
}
struct Budget {
    usage: Mutex<Usage>,
    active: AtomicUsize,
    peak: AtomicUsize,
    config: CpuConfig,
}
pub(super) struct Credit {
    budget: Arc<Budget>,
    bytes: usize,
}
impl Credit {
    fn increase(&mut self, extra: usize) -> Result<()> {
        let mut usage = self
            .budget
            .usage
            .lock()
            .map_err(|_| Error::invariant("CPU budget poisoned"))?;
        let total = usage
            .bytes
            .checked_add(extra)
            .ok_or_else(|| Error::invalid("CPU staging size overflow"))?;
        let bytes = self
            .bytes
            .checked_add(extra)
            .ok_or_else(|| Error::invalid("CPU request staging size overflow"))?;
        if total > self.budget.config.max_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "CPU preparation queue/byte budget exhausted",
            ));
        }
        usage.bytes = total;
        self.bytes = bytes;
        drop(usage);
        Ok(())
    }

    fn rollback(&mut self, extra: usize) -> Result<()> {
        let mut usage = self
            .budget
            .usage
            .lock()
            .map_err(|_| Error::invariant("CPU budget poisoned"))?;
        let total = usage
            .bytes
            .checked_sub(extra)
            .ok_or_else(|| Error::invariant("CPU staging rollback underflow"))?;
        let bytes = self
            .bytes
            .checked_sub(extra)
            .ok_or_else(|| Error::invariant("CPU request rollback underflow"))?;
        usage.bytes = total;
        self.bytes = bytes;
        drop(usage);
        Ok(())
    }
}
impl Drop for Credit {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.budget.usage.lock()
            && let Some(jobs) = usage.jobs.checked_sub(1)
            && let Some(bytes) = usage.bytes.checked_sub(self.bytes)
        {
            usage.jobs = jobs;
            usage.bytes = bytes;
        }
    }
}
struct Active(Arc<Budget>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
    }
}
#[derive(Clone)]
pub struct CpuContext {
    state: Arc<JobState>,
    deadline: Instant,
}
struct JobState {
    cancelled: AtomicBool,
    accounting: Mutex<JobAccounting>,
    initial_bytes: usize,
}
struct JobAccounting {
    credit: Option<Credit>,
    ingress: Option<Arc<Pending>>,
}
impl CpuContext {
    /// # Errors
    /// Rejects work abandoned by its caller or past its deadline.
    pub fn check(&self) -> Result<()> {
        if self.state.cancelled.load(Ordering::Acquire) || Instant::now() >= self.deadline {
            return Err(Error::new(
                ErrorCode::Capacity,
                "CPU preparation cancelled or timed out",
            ));
        }
        Ok(())
    }

    /// Fund a media working-set peak before allocating it. Existing non-media credit stays owned.
    pub(super) fn ensure_media_bytes(&self, needed: usize, baseline: usize) -> Result<()> {
        self.check()?;
        let covered = self.state.initial_bytes.min(baseline);
        let target = (self.state.initial_bytes - covered)
            .checked_add(needed.max(covered))
            .ok_or_else(|| Error::invalid("media staging size overflow"))?;
        let mut accounting = self
            .state
            .accounting
            .lock()
            .map_err(|_| Error::invariant("CPU job credit poisoned"))?;
        let JobAccounting { credit, ingress } = &mut *accounting;
        let credit = credit
            .as_mut()
            .ok_or_else(|| Error::new(ErrorCode::Capacity, "CPU preparation already finished"))?;
        if let Some(pending) = ingress.as_ref() {
            pending.check()?;
        }
        let extra = target.saturating_sub(credit.bytes);
        if extra == 0 {
            return Ok(());
        }
        credit.increase(extra)?;
        let result = ingress
            .as_ref()
            .map_or(Ok(()), |pending| pending.increase(extra));
        if let Err(error) = result {
            credit.rollback(extra)?;
            return Err(error);
        }
        drop(accounting);
        Ok(())
    }

    fn release_credit(&self) -> Result<Option<Arc<Pending>>> {
        let mut accounting = self
            .state
            .accounting
            .lock()
            .map_err(|_| Error::invariant("CPU job credit poisoned"))?;
        let credit = accounting.credit.take();
        let ingress = accounting.ingress.take();
        drop(accounting);
        drop(credit);
        Ok(ingress)
    }
}
#[cfg(test)]
pub(crate) fn cancelled_context() -> CpuContext {
    CpuContext {
        state: Arc::new(JobState {
            cancelled: AtomicBool::new(true),
            accounting: Mutex::new(JobAccounting {
                credit: None,
                ingress: None,
            }),
            initial_bytes: 0,
        }),
        deadline: Instant::now() + Duration::from_secs(1),
    }
}

struct CancelOnDrop(Arc<JobState>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancelled.store(true, Ordering::Release);
    }
}
/// An unclaimed result drops before its ingress credit, including when its receiver disappears.
struct CpuReply<T> {
    result: Result<T>,
    ingress: Option<Arc<Pending>>,
}
type Job = Box<dyn FnOnce() + Send>;
#[derive(Clone)]
pub struct CpuPool {
    jobs: mpsc::SyncSender<Job>,
    budget: Arc<Budget>,
}
impl CpuPool {
    /// # Errors
    /// Rejects invalid limits or inability to start a fixed worker.
    pub fn new(config: CpuConfig) -> Result<Self> {
        Self::named(config, "prepare")
    }
    pub(super) fn named(config: CpuConfig, lane: &str) -> Result<Self> {
        let placement = config.placement.clone();
        placement.scope(|| Self::initialize(config, lane))
    }
    fn initialize(config: CpuConfig, lane: &str) -> Result<Self> {
        config.validate()?;
        let (jobs, receiver) = mpsc::sync_channel::<Job>(config.max_jobs);
        let receiver = Arc::new(Mutex::new(receiver));
        let budget = Arc::new(Budget {
            usage: Mutex::new(Usage::default()),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            config,
        });
        for index in 0..budget.config.workers {
            let receiver = receiver.clone();
            let budget = budget.clone();
            let placement = budget.config.placement.clone();
            placement.spawn(format!("infer-cpu-{lane}-{index}"), move |_| {
                Self::work(&receiver, &budget);
            })?;
        }
        Ok(Self { jobs, budget })
    }
    fn work(receiver: &Mutex<mpsc::Receiver<Job>>, budget: &Arc<Budget>) {
        loop {
            let job = match receiver.lock() {
                Ok(receiver) => receiver.recv(),
                Err(_) => break,
            };
            let Ok(job) = job else {
                break;
            };
            let active = budget.active.fetch_add(1, Ordering::Relaxed) + 1;
            budget.peak.fetch_max(active, Ordering::Relaxed);
            let _active = Active(budget.clone());
            // A bad CPU provider fails its reply without destroying the fixed worker pool.
            let _ = catch_unwind(AssertUnwindSafe(job));
        }
    }
    pub(super) fn reserve(&self, bytes: usize) -> Result<Credit> {
        let mut usage = self
            .budget
            .usage
            .lock()
            .map_err(|_| Error::invariant("CPU budget poisoned"))?;
        let total = usage
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("CPU staging size overflow"))?;
        if usage.jobs >= self.budget.config.max_jobs || total > self.budget.config.max_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "CPU preparation queue/byte budget exhausted",
            ));
        }
        usage.jobs += 1;
        usage.bytes = total;
        drop(usage);
        Ok(Credit {
            budget: self.budget.clone(),
            bytes,
        })
    }
    /// # Errors
    /// Returns capacity, cancellation, timeout, provider or stopped-worker errors.
    pub async fn run<T: Send + 'static>(
        &self,
        bytes: usize,
        work: impl FnOnce(&CpuContext) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_until(bytes, None, work).await
    }
    /// Bound queued CPU work by both its lane timeout and an absolute request deadline.
    /// # Errors
    /// Returns queue, deadline, cancellation or provider errors.
    pub async fn run_until<T: Send + 'static>(
        &self,
        bytes: usize,
        deadline: Option<Instant>,
        work: impl FnOnce(&CpuContext) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let credit = self.reserve(bytes)?;
        self.run_reserved(credit, deadline, work).await
    }
    pub(super) async fn run_reserved<T: Send + 'static>(
        &self,
        credit: Credit,
        deadline: Option<Instant>,
        work: impl FnOnce(&CpuContext) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_with_ingress(credit, deadline, None, work).await
    }

    pub(super) async fn run_preparing<T: Send + 'static>(
        &self,
        bytes: usize,
        deadline: Option<Instant>,
        pending: &Arc<Pending>,
        work: impl FnOnce(&CpuContext) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let credit = self.reserve(bytes)?;
        self.run_with_ingress(credit, deadline, Some(pending.clone()), work)
            .await
    }

    async fn run_with_ingress<T: Send + 'static>(
        &self,
        credit: Credit,
        deadline: Option<Instant>,
        ingress: Option<Arc<Pending>>,
        work: impl FnOnce(&CpuContext) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let now = Instant::now();
        let timeout = Duration::from_millis(self.budget.config.job_timeout_ms).min(
            deadline.map_or(Duration::MAX, |deadline| {
                deadline.saturating_duration_since(now)
            }),
        );
        let context = CpuContext {
            state: Arc::new(JobState {
                cancelled: AtomicBool::new(false),
                initial_bytes: credit.bytes,
                accounting: Mutex::new(JobAccounting {
                    credit: Some(credit),
                    ingress,
                }),
            }),
            deadline: now + timeout,
        };
        let _cancel = CancelOnDrop(context.state.clone());
        let (reply, response) = oneshot::channel();
        self.jobs
            .try_send(Box::new(move || {
                let result = context
                    .check()
                    .and_then(|()| work(&context))
                    .and_then(|value| context.check().map(|()| value));
                // Completion must make admission capacity reusable before waking the caller.
                match context.release_credit() {
                    Ok(ingress) => {
                        if let Err(undelivered) = reply.send(CpuReply { result, ingress }) {
                            drop(undelivered);
                        }
                    }
                    Err(error) => {
                        drop(result);
                        if let Err(undelivered) = reply.send(CpuReply {
                            result: Err(error),
                            ingress: None,
                        }) {
                            drop(undelivered);
                        }
                    }
                }
            }))
            .map_err(|_| Error::new(ErrorCode::Capacity, "CPU queue unavailable"))?;
        let reply = time_timeout(timeout, response)
            .await
            .map_err(|_| Error::new(ErrorCode::Capacity, "CPU preparation deadline exceeded"))?
            .map_err(|_| Error::new(ErrorCode::Backend, "CPU preparation worker failed"))?;
        let CpuReply { result, ingress } = reply;
        drop(ingress);
        result
    }
    #[must_use]
    pub fn inspect(&self) -> CpuInspection {
        let usage = self.budget.usage.lock().map_or_else(
            |_| Usage::default(),
            |u| Usage {
                jobs: u.jobs,
                bytes: u.bytes,
            },
        );
        CpuInspection {
            workers: self.budget.config.workers,
            retained_jobs: usage.jobs,
            retained_bytes: usage.bytes,
            active_workers: self.budget.active.load(Ordering::Relaxed),
            peak_workers: self.budget.peak.load(Ordering::Relaxed),
        }
    }
}
