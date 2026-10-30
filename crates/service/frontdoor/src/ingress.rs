//! Admission credits survive abandoned futures until CPU jobs and deferred commands drop their payloads.
use infer_core::{Error, ErrorCode, RequestId, Result};
use std::{
    collections::BTreeMap, sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::Ordering,
};
use tokio::sync::Notify;

struct Cancellation {
    flag: AtomicBool,
    wake: Notify,
}

impl Cancellation {
    fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.wake.notify_waiters();
    }
}

struct Usage {
    entries: BTreeMap<RequestId, (Arc<Cancellation>, usize)>,
    bytes: usize,
}
pub struct Registry {
    usage: Mutex<Usage>,
    max_jobs: usize,
}
pub struct Pending {
    registry: Arc<Registry>,
    id: RequestId,
    cancelled: Arc<Cancellation>,
}
impl Registry {
    pub fn new(max_jobs: usize) -> Arc<Self> {
        Arc::new(Self {
            usage: Mutex::new(Usage {
                entries: BTreeMap::new(),
                bytes: 0,
            }),
            max_jobs,
        })
    }
    pub fn reserve(self: &Arc<Self>, id: RequestId, bytes: usize) -> Result<Arc<Pending>> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        if usage.entries.contains_key(&id) {
            return Err(Error::new(ErrorCode::Conflict, "request already preparing"));
        }
        let total = usage
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("ingress bytes overflow"))?;
        if usage.entries.len() >= self.max_jobs {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pending ingress budget exhausted",
            ));
        }
        let cancelled = Arc::new(Cancellation {
            flag: AtomicBool::new(false),
            wake: Notify::new(),
        });
        usage.entries.insert(id, (cancelled.clone(), bytes));
        usage.bytes = total;
        drop(usage);
        Ok(Arc::new(Pending {
            registry: self.clone(),
            id,
            cancelled,
        }))
    }
    pub fn cancel(&self, id: RequestId) -> bool {
        let Ok(usage) = self.usage.lock() else {
            return false;
        };
        let Some((cancelled, _)) = usage.entries.get(&id) else {
            return false;
        };
        let cancelled = cancelled.clone();
        drop(usage);
        cancelled.cancel();
        true
    }
}
impl Pending {
    pub fn add_bytes(&self, additional: usize) -> Result<()> {
        let mut usage = self
            .registry
            .usage
            .lock()
            .map_err(|_| Error::invariant("ingress registry poisoned"))?;
        let total = usage
            .bytes
            .checked_add(additional)
            .ok_or_else(|| Error::invalid("ingress bytes overflow"))?;
        let entry = usage
            .entries
            .get_mut(&self.id)
            .ok_or_else(|| Error::invariant("pending ingress entry missing"))?;
        let retained = entry
            .1
            .checked_add(additional)
            .ok_or_else(|| Error::invalid("ingress request bytes overflow"))?;
        entry.1 = retained;
        usage.bytes = total;
        drop(usage);
        Ok(())
    }
    pub fn check(&self) -> Result<()> {
        if self.cancelled.flag.load(Ordering::Acquire) {
            return Err(Error::new(
                ErrorCode::Capacity,
                "request preparation cancelled",
            ));
        }
        Ok(())
    }
    pub async fn cancelled(&self) {
        loop {
            let notified = self.cancelled.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.cancelled.flag.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
    pub fn abandon(&self) {
        self.cancelled.cancel();
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.registry.usage.lock()
            && let Some((_, bytes)) = usage.entries.remove(&self.id)
        {
            usage.bytes = usage.bytes.saturating_sub(bytes);
        }
    }
}
pub struct Abandon(pub Arc<Pending>);
impl Drop for Abandon {
    fn drop(&mut self) {
        self.0.abandon();
    }
}

#[cfg(test)]
#[path = "../tests/unit/ingress.rs"]
mod tests;
