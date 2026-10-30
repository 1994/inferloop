//! Admission credits survive abandoned futures until CPU jobs and deferred commands drop their payloads.
use infer_core::{Error, ErrorCode, RequestId, Result};
use std::{
    collections::BTreeMap, sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::Ordering,
};

struct Usage {
    entries: BTreeMap<RequestId, (Arc<AtomicBool>, usize)>,
    bytes: usize,
}
pub struct Registry {
    usage: Mutex<Usage>,
    max_jobs: usize,
    max_bytes: usize,
}
pub struct Pending {
    registry: Arc<Registry>,
    id: RequestId,
    bytes: usize,
    cancelled: Arc<AtomicBool>,
}
impl Registry {
    pub fn new(max_jobs: usize, max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            usage: Mutex::new(Usage {
                entries: BTreeMap::new(),
                bytes: 0,
            }),
            max_jobs,
            max_bytes,
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
            bytes,
            cancelled,
        }))
    }
    pub fn cancel(&self, id: RequestId) -> bool {
        let Ok(usage) = self.usage.lock() else {
            return false;
        };
        let Some((flag, _)) = usage.entries.get(&id) else {
            return false;
        };
        flag.store(true, Ordering::Release);
        true
    }
}
impl Pending {
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
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.registry.usage.lock() {
            usage.entries.remove(&self.id);
            usage.bytes -= self.bytes;
        }
    }
}
pub struct Abandon(pub Arc<Pending>);
impl Drop for Abandon {
    fn drop(&mut self) {
        self.0.abandon();
    }
}
