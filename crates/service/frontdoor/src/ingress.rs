//! Admission credits survive abandoned futures until CPU jobs and deferred commands drop their payloads.
use infer_core::{Error, ErrorCode, RequestId, Result};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    sync::Mutex,
    sync::atomic::AtomicBool,
    sync::atomic::Ordering,
    time::Instant,
};
use tokio::sync::Notify;
use tokio::time::{Instant as TokioInstant, timeout_at};

struct Usage {
    entries: BTreeMap<RequestId, (Arc<AtomicBool>, usize)>,
    bytes: usize,
    waiters: VecDeque<GrowthDemand>,
    drains: usize,
    stopped: bool,
}
struct GrowthDemand {
    id: RequestId,
    target: usize,
}
pub struct Registry {
    usage: Mutex<Usage>,
    changed: Notify,
    max_jobs: usize,
    max_bytes: usize,
}
pub struct Pending {
    registry: Arc<Registry>,
    id: RequestId,
    cancelled: Arc<AtomicBool>,
}
impl Registry {
    pub fn new(max_jobs: usize, max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            usage: Mutex::new(Usage {
                entries: BTreeMap::new(),
                bytes: 0,
                waiters: VecDeque::new(),
                drains: 0,
                stopped: false,
            }),
            changed: Notify::new(),
            max_jobs,
            max_bytes,
        })
    }
    pub fn reserve(self: &Arc<Self>, id: RequestId, bytes: usize) -> Result<Arc<Pending>> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        if usage.stopped {
            return Err(Error::new(ErrorCode::Capacity, "ingress is stopping"));
        }
        if usage.entries.contains_key(&id) {
            return Err(Error::new(ErrorCode::Conflict, "request already preparing"));
        }
        let total = usage
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("ingress bytes overflow"))?;
        if usage.entries.len() >= self.max_jobs || total > self.max_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pending ingress budget exhausted",
            ));
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        usage.entries.insert(id, (cancelled.clone(), bytes));
        usage.bytes = total;
        drop(usage);
        Ok(Arc::new(Pending {
            registry: self.clone(),
            id,
            cancelled,
        }))
    }
    /// Reserve the full preparation quote when possible, otherwise retain the input's credit.
    pub(super) fn reserve_staged(
        self: &Arc<Self>,
        id: RequestId,
        retained: usize,
        peak: usize,
    ) -> Result<(Arc<Pending>, bool)> {
        if retained > peak {
            return Err(Error::invariant("staged input exceeds its peak quote"));
        }
        match self.reserve(id, peak) {
            Ok(pending) => Ok((pending, true)),
            Err(error)
                if error.code == ErrorCode::Capacity
                    && retained < peak
                    && peak <= self.max_bytes =>
            {
                self.reserve(id, retained).map(|pending| (pending, false))
            }
            Err(error) => Err(error),
        }
    }
    pub fn cancel(&self, id: RequestId) -> bool {
        let Ok(usage) = self.usage.lock() else {
            return false;
        };
        let Some((flag, _)) = usage.entries.get(&id) else {
            return false;
        };
        flag.store(true, Ordering::Release);
        drop(usage);
        self.changed.notify_waiters();
        true
    }

    pub(super) fn stop(&self) -> Result<()> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        usage.stopped = true;
        for (cancelled, _) in usage.entries.values() {
            cancelled.store(true, Ordering::Release);
        }
        drop(usage);
        self.changed.notify_waiters();
        Ok(())
    }

    pub(super) async fn wait_drained(&self, deadline: Instant) -> Result<()> {
        let _drain = DrainWait::enter(self)?;
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            let _ = notified.as_mut().enable();
            let empty = {
                let usage = self
                    .usage
                    .lock()
                    .map_err(|_| Error::invariant("ingress registry poisoned"))?;
                usage.entries.is_empty()
            };
            if empty {
                return Ok(());
            }
            timeout_at(TokioInstant::from_std(deadline), &mut notified)
                .await
                .map_err(|_| {
                    Error::new(
                        ErrorCode::Backend,
                        "shutdown timed out while ingress resources remain owned",
                    )
                })?;
        }
    }
}

struct DrainWait<'a>(&'a Registry);
impl<'a> DrainWait<'a> {
    fn enter(registry: &'a Registry) -> Result<Self> {
        let mut usage = registry
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        usage.drains = usage
            .drains
            .checked_add(1)
            .ok_or_else(|| Error::invariant("ingress drain count overflow"))?;
        drop(usage);
        Ok(Self(registry))
    }
}
impl Drop for DrainWait<'_> {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.0.usage.lock()
            && let Some(drains) = usage.drains.checked_sub(1)
        {
            usage.drains = drains;
        }
    }
}
impl Pending {
    pub(super) fn notifier(&self) -> &Notify {
        &self.registry.changed
    }
    pub fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "request preparation cancelled",
            ));
        }
        Ok(())
    }
    pub fn abandon(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.registry.changed.notify_waiters();
    }

    /// Retain additional request bytes until the last pending owner has finished.
    pub(super) fn increase(&self, extra: usize) -> Result<()> {
        let mut usage = self
            .registry
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        let total = usage
            .bytes
            .checked_add(extra)
            .ok_or_else(|| Error::invalid("ingress bytes overflow"))?;
        if total > self.registry.max_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pending ingress budget exhausted",
            ));
        }
        let (cancelled, bytes) = usage
            .entries
            .get_mut(&self.id)
            .ok_or_else(|| Error::invariant("pending ingress identity missing"))?;
        if !Arc::ptr_eq(cancelled, &self.cancelled) {
            return Err(Error::invariant("pending ingress identity changed"));
        }
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "request preparation cancelled",
            ));
        }
        let updated = bytes
            .checked_add(extra)
            .ok_or_else(|| Error::invalid("ingress request bytes overflow"))?;
        *bytes = updated;
        usage.bytes = total;
        drop(usage);
        Ok(())
    }

    /// Return unused media preparation credit while retaining the staged input.
    pub(super) fn retain_bytes(&self, bytes: usize) -> Result<()> {
        let mut usage = self
            .registry
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        self.check()?;
        let (cancelled, current) = usage
            .entries
            .get(&self.id)
            .ok_or_else(|| Error::invariant("pending ingress identity missing"))?;
        if !Arc::ptr_eq(cancelled, &self.cancelled) {
            return Err(Error::invariant("pending ingress identity changed"));
        }
        let released = current
            .checked_sub(bytes)
            .ok_or_else(|| Error::invariant("staged media exceeds its ingress credit"))?;
        let total = usage
            .bytes
            .checked_sub(released)
            .ok_or_else(|| Error::invariant("ingress retained bytes underflow"))?;
        let (_, current) = usage
            .entries
            .get_mut(&self.id)
            .ok_or_else(|| Error::invariant("pending ingress identity missing"))?;
        *current = bytes;
        usage.bytes = total;
        let wake = !usage.waiters.is_empty();
        drop(usage);
        if wake {
            self.registry.changed.notify_waiters();
        }
        Ok(())
    }

    /// Wait without occupying a CPU worker until this request's full quote fits.
    pub(super) async fn wait_for_bytes(&self, target: usize, deadline: Instant) -> Result<()> {
        if target > self.registry.max_bytes {
            return Err(Error::new(
                ErrorCode::Capacity,
                "image staging quote exceeds ingress budget",
            ));
        }
        let mut waiter = GrowthWait::enter(self, target)?;
        loop {
            if Instant::now() >= deadline {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "image staging wait timed out",
                ));
            }
            let notified = self.registry.changed.notified();
            tokio::pin!(notified);
            let _ = notified.as_mut().enable();
            if waiter.try_grant()? {
                return Ok(());
            }
            timeout_at(TokioInstant::from_std(deadline), &mut notified)
                .await
                .map_err(|_| Error::new(ErrorCode::Capacity, "image staging wait timed out"))?;
        }
    }
}

struct GrowthWait<'a> {
    pending: &'a Pending,
    target: usize,
    queued: bool,
}
impl<'a> GrowthWait<'a> {
    fn enter(pending: &'a Pending, target: usize) -> Result<Self> {
        let mut usage = pending
            .registry
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        pending.check()?;
        if usage.stopped || usage.waiters.iter().any(|waiter| waiter.id == pending.id) {
            return Err(Error::new(
                ErrorCode::Conflict,
                "ingress growth is unavailable or duplicated",
            ));
        }
        if !usage
            .entries
            .get(&pending.id)
            .is_some_and(|(flag, _)| Arc::ptr_eq(flag, &pending.cancelled))
        {
            return Err(Error::invariant("pending ingress identity changed"));
        }
        usage.waiters.push_back(GrowthDemand {
            id: pending.id,
            target,
        });
        drop(usage);
        Ok(Self {
            pending,
            target,
            queued: true,
        })
    }

    fn try_grant(&mut self) -> Result<bool> {
        let mut usage = self
            .pending
            .registry
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        self.pending.check()?;
        if usage.stopped {
            return Err(Error::new(ErrorCode::Capacity, "ingress is stopping"));
        }
        let free = self
            .pending
            .registry
            .max_bytes
            .checked_sub(usage.bytes)
            .ok_or_else(|| Error::invariant("ingress free bytes underflow"))?;
        let mut selected = None;
        for waiter in &usage.waiters {
            let (_, current) = usage
                .entries
                .get(&waiter.id)
                .ok_or_else(|| Error::invariant("staged ingress waiter identity missing"))?;
            if waiter.target.saturating_sub(*current) <= free {
                selected = Some(waiter.id);
                break;
            }
        }
        if selected.is_none()
            && usage.waiters.len() == usage.entries.len()
            && usage.waiters.len() > 1
            && usage
                .waiters
                .back()
                .is_some_and(|waiter| waiter.id == self.pending.id)
        {
            usage.waiters.pop_back();
            self.queued = false;
            drop(usage);
            self.pending.registry.changed.notify_waiters();
            return Err(Error::new(
                ErrorCode::Capacity,
                "staged images cannot jointly fit the ingress budget",
            ));
        }
        if selected != Some(self.pending.id) {
            return Ok(false);
        }
        let (cancelled, current) = usage
            .entries
            .get(&self.pending.id)
            .ok_or_else(|| Error::invariant("pending ingress identity missing"))?;
        if !Arc::ptr_eq(cancelled, &self.pending.cancelled) {
            return Err(Error::invariant("pending ingress identity changed"));
        }
        let extra = self.target.saturating_sub(*current);
        let total = usage
            .bytes
            .checked_add(extra)
            .ok_or_else(|| Error::invalid("ingress bytes overflow"))?;
        if total > self.pending.registry.max_bytes {
            return Ok(false);
        }
        let index = usage
            .waiters
            .iter()
            .position(|waiter| waiter.id == self.pending.id)
            .ok_or_else(|| Error::invariant("staged ingress waiter disappeared"))?;
        let (_, current) = usage
            .entries
            .get_mut(&self.pending.id)
            .ok_or_else(|| Error::invariant("pending ingress identity missing"))?;
        *current = (*current).max(self.target);
        usage.bytes = total;
        usage.waiters.remove(index);
        self.queued = false;
        let wake = !usage.waiters.is_empty();
        drop(usage);
        if wake {
            self.pending.registry.changed.notify_waiters();
        }
        Ok(true)
    }
}
impl Drop for GrowthWait<'_> {
    fn drop(&mut self) {
        if self.queued
            && let Ok(mut usage) = self.pending.registry.usage.lock()
        {
            if let Some(index) = usage
                .waiters
                .iter()
                .position(|waiter| waiter.id == self.pending.id)
            {
                usage.waiters.remove(index);
            }
            let wake = !usage.waiters.is_empty();
            drop(usage);
            if wake {
                self.pending.registry.changed.notify_waiters();
            }
        }
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.registry.usage.lock()
            && let Some((cancelled, bytes)) = usage.entries.get(&self.id)
            && Arc::ptr_eq(cancelled, &self.cancelled)
            && let Some(remaining) = usage.bytes.checked_sub(*bytes)
        {
            usage.entries.remove(&self.id);
            usage.bytes = remaining;
            let wake = !usage.waiters.is_empty() || usage.drains != 0;
            drop(usage);
            if wake {
                self.registry.changed.notify_waiters();
            }
        }
    }
}
pub struct Abandon(pub Arc<Pending>);
impl Drop for Abandon {
    fn drop(&mut self) {
        self.0.abandon();
    }
}
