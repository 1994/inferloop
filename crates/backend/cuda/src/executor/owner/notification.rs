//! Completion edges gate driver polling; periodic probes still detect missing callbacks.
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Wake, Waker},
    time::{Duration, Instant},
};

/// Bound fault detection when a failed stream cannot deliver its completion callback.
const FAULT_PROBE_INTERVAL: Duration = Duration::from_millis(2);

struct Signal {
    ready: Arc<AtomicBool>,
    owner: Option<Arc<dyn Fn() + Send + Sync>>,
}
impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.ready.store(true, Ordering::Release);
        if let Some(owner) = &self.owner {
            owner();
        }
    }
}

pub(super) struct Notification {
    ready: Arc<AtomicBool>,
    waker: Waker,
    last_probe: Instant,
}
impl Notification {
    pub(super) fn new() -> Self {
        let ready = Arc::new(AtomicBool::new(true));
        let waker = Waker::from(Arc::new(Signal {
            ready: Arc::clone(&ready),
            owner: None,
        }));
        Self {
            ready,
            waker,
            last_probe: Instant::now(),
        }
    }
    pub(super) fn set_waker(&mut self, owner: Arc<dyn Fn() + Send + Sync>) {
        self.waker = Waker::from(Arc::new(Signal {
            ready: Arc::clone(&self.ready),
            owner: Some(owner),
        }));
        // Force a poll to register the replacement waker with an already pending future.
        self.waker.wake_by_ref();
    }
    pub(super) fn arm(&self) {
        self.ready.store(true, Ordering::Release);
    }
    pub(super) const fn waker(&self) -> &Waker {
        &self.waker
    }
    pub(super) fn should_poll(&mut self) -> bool {
        self.should_poll_at(Instant::now())
    }
    fn should_poll_at(&mut self, now: Instant) -> bool {
        // This flag only schedules a poll. The device future must still establish DMA /
        // stream completion before any result or physical owner can be published.
        let signaled = self.ready.swap(false, Ordering::Acquire);
        if !signaled && now.saturating_duration_since(self.last_probe) < FAULT_PROBE_INTERVAL {
            return false;
        }
        self.last_probe = now;
        true
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/executor_notification.rs"]
mod tests;
