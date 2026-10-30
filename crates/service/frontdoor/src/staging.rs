//! Host staging limits. Credits account for work; they do not allocate memory.
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use sysinfo::System;

/// Portion of currently available memory made available to each staging lane.
const AVAILABLE_MEMORY_DIVISOR: u64 = 8;
/// Conservative previous default when a memory snapshot is unavailable.
const FALLBACK_BYTES: usize = 64 << 20;
/// Bound the cost of refreshing the operating-system memory snapshot.
pub(super) const REFRESH_INTERVAL: Duration = Duration::from_secs(1);

pub(super) struct StagingLimit {
    bytes: AtomicUsize,
    refreshed: Option<Mutex<Instant>>,
}

impl StagingLimit {
    pub(super) fn fixed(bytes: usize) -> Self {
        Self {
            bytes: AtomicUsize::new(bytes),
            refreshed: None,
        }
    }

    pub(super) fn automatic() -> Self {
        let bytes = available_memory().map_or(FALLBACK_BYTES, budget_for_available);
        Self {
            bytes: AtomicUsize::new(bytes),
            refreshed: Some(Mutex::new(Instant::now())),
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
        let next = available_memory().map_or(FALLBACK_BYTES, budget_for_available);
        self.bytes.store(next, Ordering::Relaxed);
    }
}

fn budget_for_available(bytes: u64) -> usize {
    usize::try_from(bytes / AVAILABLE_MEMORY_DIVISOR)
        .unwrap_or(usize::MAX)
        .max(1)
}

fn available_memory() -> Option<u64> {
    let mut system = System::new();
    system.refresh_memory();
    let available = system.available_memory();
    if available == 0 || system.total_memory() == 0 {
        return None;
    }
    let available = system
        .cgroup_limits()
        .map_or(available, |limit| available.min(limit.free_memory));
    #[cfg(target_os = "linux")]
    let available = cgroup_v2_available().map_or(available, |limit| available.min(limit));
    Some(available)
}

// sysinfo checks a cgroup namespace root. A systemd service can instead be nested
// beneath that root, so include the effective limit of each ancestor of this process.
#[cfg(target_os = "linux")]
fn cgroup_v2_available() -> Option<u64> {
    use std::fs::read_to_string;
    use std::path::{Component, Path};

    let groups = read_to_string("/proc/self/cgroup").ok()?;
    let group = groups.lines().find_map(|line| line.strip_prefix("0::"))?;
    let relative = Path::new(group).strip_prefix("/").ok()?;
    if relative
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
    {
        return None;
    }
    let root = Path::new("/sys/fs/cgroup");
    let leaf = root.join(relative);
    leaf.ancestors()
        .take_while(|path| path.starts_with(root))
        .filter_map(|path| {
            let max = read_to_string(path.join("memory.max"))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()?;
            let current = read_to_string(path.join("memory.current"))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()?;
            Some(max.saturating_sub(current))
        })
        .min()
}

#[cfg(test)]
#[path = "../tests/unit/staging.rs"]
mod tests;
