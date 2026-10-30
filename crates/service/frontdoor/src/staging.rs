//! Host staging limits. Credits account for work; they do not allocate memory.
use infer_core::credits::CreditLimit;
use infer_runtime::host_memory::{snapshot, staging_bytes};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Bound the cost of refreshing the operating-system memory snapshot.
pub const REFRESH_INTERVAL: Duration = Duration::from_secs(1);

pub struct StagingLimit {
    bytes: AtomicUsize,
    refreshed: Option<Mutex<Instant>>,
    host_limit: Option<CreditLimit>,
}

impl StagingLimit {
    pub const fn fixed(bytes: usize) -> Self {
        Self {
            bytes: AtomicUsize::new(bytes),
            refreshed: None,
            host_limit: None,
        }
    }

    pub(super) fn automatic(bytes: usize, host_limit: Option<CreditLimit>) -> Self {
        if let Some(limit) = &host_limit {
            limit.set(bytes);
        }
        Self {
            bytes: AtomicUsize::new(bytes),
            refreshed: Some(Mutex::new(Instant::now())),
            host_limit,
        }
    }

    pub(super) fn current(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    pub(super) const fn is_automatic(&self) -> bool {
        self.refreshed.is_some()
    }

    /// Called by existing CPU workers, never by an asynchronous request handler.
    pub(super) fn refresh_if_due(&self) {
        let Some(refreshed) = &self.refreshed else {
            return;
        };
        let Ok(mut last) = refreshed.try_lock() else {
            return;
        };
        if last.elapsed() < REFRESH_INTERVAL {
            return;
        }
        *last = Instant::now();
        let next = default_staging_bytes();
        self.bytes.store(next, Ordering::Relaxed);
        if let Some(limit) = &self.host_limit {
            limit.set(next);
        }
    }
}

/// A zero result means the operating system could not provide a memory snapshot.
pub fn default_staging_bytes() -> usize {
    snapshot().map_or(0, |memory| staging_bytes(memory.available))
}

#[cfg(test)]
#[path = "../tests/unit/staging.rs"]
mod tests;
