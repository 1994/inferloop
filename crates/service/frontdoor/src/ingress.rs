//! Admission credits survive abandoned futures until CPU jobs and deferred commands drop their payloads.
use infer_core::{Error, ErrorCode, RequestId, RequestStage, Result};
use std::future::Future;
use std::{
    cell::Cell, collections::BTreeMap, sync::Arc, sync::Mutex, sync::atomic::AtomicBool,
    sync::atomic::Ordering, time::Instant,
};
use tokio::sync::Notify;

tokio::task_local! {
    /// The request identity an HTTP exchange reserved, so its access line can name it.
    static HTTP_REQUEST: Cell<Option<RequestId>>;
}

/// Run one HTTP exchange and report which request identity it reserved, if any.
pub async fn correlate<F: Future>(exchange: F) -> (F::Output, Option<RequestId>) {
    HTTP_REQUEST
        .scope(Cell::new(None), async move {
            let output = exchange.await;
            (output, HTTP_REQUEST.with(Cell::get))
        })
        .await
}

/// Record the identity being prepared for the enclosing HTTP exchange. Outside one there is
/// nothing to correlate, which is not an error.
fn note_http_request(id: RequestId) {
    HTTP_REQUEST
        .try_with(|current| current.set(Some(id)))
        .unwrap_or_default();
}

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
        note_http_request(id);
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
    fn is_cancelled(&self) -> bool {
        self.cancelled.flag.load(Ordering::Acquire)
    }
    /// Bytes this preparation still holds; diagnostic, read only when it ends early.
    fn retained_bytes(&self) -> usize {
        self.registry
            .usage
            .lock()
            .ok()
            .and_then(|usage| usage.entries.get(&self.id).map(|(_, bytes)| *bytes))
            .unwrap_or(0)
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
/// Ownership of a preparation that has not reached the engine yet.
///
/// The engine logs every request it admitted or rejected, but it never sees a preparation that
/// ended first. Dropping this guard while it is armed means the submitter's future went away mid
/// preparation: it cancels the CPU work and writes the request's one terminal line. A preparation
/// that returned is settled instead, which reports the endings the engine cannot. Once the
/// prepared request reached the engine ([`RequestStage::Admission`]) the engine owns the line:
/// it rejects the request or admits it and then cancels it when nobody reads the reply.
pub struct Abandon {
    pending: Arc<Pending>,
    started: Instant,
    stage: RequestStage,
    armed: bool,
}
impl Abandon {
    pub fn new(pending: Arc<Pending>, stage: RequestStage) -> Self {
        Self {
            pending,
            started: Instant::now(),
            stage,
            armed: true,
        }
    }
    pub const fn pending(&self) -> &Arc<Pending> {
        &self.pending
    }
    /// The preparation moved on; an abandonment from here on reports this stage.
    pub const fn enter(&mut self, stage: RequestStage) {
        self.stage = stage;
    }
    /// Account for a preparation that returned.
    ///
    /// Success and anything after the engine received the request leave the ending to the engine.
    /// Anything else ended before the engine saw the request: an explicit cancel is an
    /// abandonment the requester asked for, any other error rejected the request here.
    pub fn settle<T>(mut self, result: &Result<T>) {
        self.armed = false;
        let Err(error) = result else {
            return;
        };
        if self.stage == RequestStage::Admission {
            return;
        }
        if self.pending.is_cancelled() {
            self.report("abandoned", "requested", None);
        } else {
            self.report("rejected", "none", Some(error.code));
        }
    }
    fn report(&self, phase: &'static str, cancel: &'static str, error: Option<ErrorCode>) {
        tracing::warn!(
            target: "infer::request",
            request = self.pending.id.get(),
            phase,
            cancel,
            stage = self.stage.as_str(),
            error_code = error.map_or("none", ErrorCode::as_str),
            elapsed_us = u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX),
            bytes = self.pending.retained_bytes(),
            "request ended before admission"
        );
    }
}
impl Drop for Abandon {
    fn drop(&mut self) {
        if self.armed && self.stage != RequestStage::Admission {
            self.report("abandoned", "client_disconnected", None);
        }
        self.pending.abandon();
    }
}

#[cfg(test)]
mod tests;
