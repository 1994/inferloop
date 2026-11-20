//! Persistent completion cells and reader-counted leases bound compute, CPU work and acknowledgements.
use super::{output::OutputAcknowledgement, output::OutputJob};
use infer_core::placement::ThreadPlacement;
use infer_core::{Error, ErrorCode, Result};
use infer_spi::WorkloadProvider;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::time::Instant;
use std::{
    sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::AtomicU8,
    sync::atomic::AtomicUsize, sync::atomic::Ordering, sync::mpsc,
};

#[cfg(test)]
mod tests;
type Wake = Arc<dyn Fn() + Send + Sync>;
struct Shared {
    used: AtomicUsize,
    wake: Mutex<Option<Wake>>,
}
impl Shared {
    fn notify(&self) {
        // Release the lock before invoking the opaque waker; see `runner::Shared::notify`.
        let wake = self.wake.lock().ok().and_then(|wake| wake.clone());
        if let Some(wake) = wake {
            wake();
        }
    }
}
struct Reply {
    state: AtomicU8,
    value: Mutex<Option<OutputAcknowledgement>>,
}
struct Lease {
    shared: Arc<Shared>,
    leased: AtomicBool,
    readers: AtomicUsize,
    replies: Vec<Reply>,
}
impl Lease {
    fn new(shared: Arc<Shared>, batch: usize) -> Result<Arc<Self>> {
        let mut replies = Vec::new();
        replies
            .try_reserve_exact(batch)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        replies.resize_with(batch, || Reply {
            state: AtomicU8::new(0),
            value: Mutex::new(None),
        });
        for reply in &replies {
            drop(
                reply
                    .value
                    .lock()
                    .map_err(|_| Error::invariant("new output cell poisoned"))?,
            );
        }
        Ok(Arc::new(Self {
            shared,
            leased: AtomicBool::new(false),
            readers: AtomicUsize::new(0),
            replies,
        }))
    }
    fn publish(&self, index: usize, reply: OutputAcknowledgement) {
        if let Some(cell) = self.replies.get(index)
            && let Ok(mut value) = cell.value.lock()
        {
            *value = Some(reply);
            drop(value);
            cell.state.store(1, Ordering::Release);
        }
        self.shared.notify();
    }
}
pub struct OutputCredit {
    lease: Arc<Lease>,
}
impl Clone for OutputCredit {
    fn clone(&self) -> Self {
        self.lease.readers.fetch_add(1, Ordering::Relaxed);
        Self {
            lease: self.lease.clone(),
        }
    }
}
impl Drop for OutputCredit {
    fn drop(&mut self) {
        if self.lease.readers.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        // The final reader also reclaims abandoned results. Jobs retain a reader until processing ends.
        for reply in &self.lease.replies {
            if let Ok(mut value) = reply.value.lock() {
                *value = None;
            }
            reply.state.store(0, Ordering::Relaxed);
        }
        self.lease.shared.used.fetch_sub(1, Ordering::AcqRel);
        self.lease.leased.store(false, Ordering::Release);
        self.lease.shared.notify();
    }
}
pub struct OutputTicket {
    pending: u64,
    credit: OutputCredit,
}
impl OutputTicket {
    pub fn poll(&mut self, index: usize) -> Option<OutputAcknowledgement> {
        if !self.pending(index) {
            return None;
        }
        let reply = self.credit.lease.replies.get(index)?;
        if reply.state.load(Ordering::Acquire) != 1 {
            return None;
        }
        let result = reply.value.lock().ok()?.take()?;
        self.pending &= !(1_u64 << index);
        Some(result)
    }
    pub const fn pending(&self, index: usize) -> bool {
        index < infer_gpu_api::MAX_SUBMISSION_BATCH && self.pending & (1_u64 << index) != 0
    }
    pub const fn done(&self) -> bool {
        self.pending == 0
    }
}
struct Job {
    input: OutputJob,
    credit: OutputCredit,
    index: usize,
}
pub struct OutputWorker {
    jobs: mpsc::SyncSender<Job>,
    shared: Arc<Shared>,
    leases: [Arc<Lease>; 2],
}
impl OutputWorker {
    #[cfg(test)]
    pub fn new(workloads: &Arc<dyn WorkloadProvider + Send + Sync>, batch: usize) -> Result<Self> {
        Self::placed(workloads, batch, 0, &ThreadPlacement::default())
    }
    pub fn placed(
        workloads: &Arc<dyn WorkloadProvider + Send + Sync>,
        batch: usize,
        vocabulary: usize,
        placement: &ThreadPlacement,
    ) -> Result<Self> {
        placement.scope(|| Self::initialize(workloads, batch, vocabulary, placement))
    }
    fn initialize(
        workloads: &Arc<dyn WorkloadProvider + Send + Sync>,
        batch: usize,
        vocabulary: usize,
        placement: &ThreadPlacement,
    ) -> Result<Self> {
        if batch == 0 || batch > infer_gpu_api::MAX_SUBMISSION_BATCH {
            return Err(Error::invalid("output batch must fit completion bitset"));
        }
        let shared = Arc::new(Shared {
            used: AtomicUsize::new(0),
            wake: Mutex::new(None),
        });
        let leases = [
            Lease::new(shared.clone(), batch)?,
            Lease::new(shared.clone(), batch)?,
        ];
        let (jobs, receiver) = mpsc::sync_channel::<Job>(batch * 2);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..2 {
            let (receiver, workloads) = (receiver.clone(), workloads.clone());
            let scratch = infer_workloads::SamplingWorkspace::with_capacity(vocabulary)?;
            placement.spawn(format!("infer-output-{index}"), move |_| {
                Self::work(&receiver, workloads.as_ref(), scratch);
            })?;
        }
        Ok(Self {
            jobs,
            shared,
            leases,
        })
    }
    fn work(
        receiver: &Mutex<mpsc::Receiver<Job>>,
        workloads: &dyn WorkloadProvider,
        mut scratch: infer_workloads::SamplingWorkspace,
    ) {
        loop {
            let job = receiver
                .lock()
                .ok()
                .and_then(|receiver| receiver.recv().ok());
            let Some(Job {
                input,
                credit,
                index,
            }) = job
            else {
                break;
            };
            let started = Instant::now();
            let result = catch_unwind(AssertUnwindSafe(|| input.process(workloads, &mut scratch)))
                .unwrap_or_else(|_| {
                    Err(Error::new(ErrorCode::Backend, "output provider panicked"))
                });
            credit.lease.publish(
                index,
                OutputAcknowledgement {
                    result,
                    elapsed_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
                },
            );
        }
    }
    pub fn set_waker(&self, wake: Wake) {
        if let Ok(mut slot) = self.shared.wake.lock() {
            *slot = Some(wake);
        }
    }
    pub fn available(&self) -> bool {
        self.shared.used.load(Ordering::Acquire) < 2
    }
    pub fn reserve(&self) -> Result<OutputCredit> {
        for lease in &self.leases {
            if lease
                .leased
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            lease.readers.store(1, Ordering::Relaxed);
            self.shared.used.fetch_add(1, Ordering::AcqRel);
            return Ok(OutputCredit {
                lease: lease.clone(),
            });
        }
        Err(Error::new(
            ErrorCode::Capacity,
            "output completion credits exhausted",
        ))
    }
    #[cfg(test)]
    pub fn submit(&self, inputs: Vec<OutputJob>, credit: OutputCredit) -> Result<OutputTicket> {
        let mut indexed: Vec<_> = inputs.into_iter().enumerate().collect();
        self.submit_indexed(&mut indexed, credit)
    }
    pub fn submit_indexed(
        &self,
        inputs: &mut Vec<(usize, OutputJob)>,
        credit: OutputCredit,
    ) -> Result<OutputTicket> {
        if inputs.is_empty()
            || inputs.len() > credit.lease.replies.len()
            || !Arc::ptr_eq(&self.shared, &credit.lease.shared)
        {
            return Err(Error::invariant("foreign or oversized output credit"));
        }
        let mut pending = 0u64;
        for (index, _) in inputs.iter() {
            if *index >= credit.lease.replies.len() || pending & (1u64 << index) != 0 {
                return Err(Error::invariant("duplicate or invalid output index"));
            }
            pending |= 1u64 << index;
        }
        for (index, input) in inputs.drain(..) {
            self.jobs
                .try_send(Job {
                    input,
                    credit: credit.clone(),
                    index,
                })
                .map_err(|error| match error {
                    mpsc::TrySendError::Full(_) => {
                        Error::invariant("reserved output slot unavailable")
                    }
                    mpsc::TrySendError::Disconnected(_) => {
                        Error::new(ErrorCode::Backend, "output worker stopped")
                    }
                })?;
        }
        Ok(OutputTicket { pending, credit })
    }
}
