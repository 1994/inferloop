//! Cross-platform host memory snapshots and cgroup-aware limits.
#[cfg(target_os = "linux")]
use std::path::Path;
use sysinfo::System;

/// Portion of host memory available to a managed staging lane.
const MEMORY_DIVISOR: u64 = 8;

/// Physical host memory available to this process, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    /// Total physical memory, capped by any known container limit.
    pub total: u64,
    /// Currently available physical memory, capped by container headroom.
    pub available: u64,
}

/// Return a memory snapshot, or `None` when the operating system cannot provide one.
#[must_use]
pub fn snapshot() -> Option<HostMemory> {
    let mut system = System::new();
    system.refresh_memory();
    let (mut total, mut available) = (system.total_memory(), system.available_memory());
    if total == 0 || available == 0 {
        return None;
    }
    if let Some(limit) = system.cgroup_limits() {
        total = total.min(limit.total_memory);
        available = available.min(limit.free_memory);
    }
    #[cfg(target_os = "linux")]
    if let Some((maximum, free)) = cgroup_v2_limits() {
        total = total.min(maximum);
        available = available.min(free);
    }
    Some(HostMemory { total, available })
}

/// Derive a bounded logical staging capacity from a memory byte count.
#[must_use]
pub fn staging_bytes(bytes: u64) -> usize {
    usize::try_from(bytes / MEMORY_DIVISOR)
        .unwrap_or(usize::MAX)
        .max(1)
}

// sysinfo checks the namespace root; systemd can place the process in a nested cgroup.
#[cfg(target_os = "linux")]
fn cgroup_v2_limits() -> Option<(u64, u64)> {
    use std::fs::read_to_string;
    let groups = read_to_string("/proc/self/cgroup").ok()?;
    let group = groups.lines().find_map(|line| line.strip_prefix("0::"))?;
    cgroup_v2_limits_at(Path::new("/sys/fs/cgroup"), Path::new(group))
}

#[cfg(target_os = "linux")]
fn cgroup_v2_limits_at(root: &Path, group: &Path) -> Option<(u64, u64)> {
    use std::fs::read_to_string;
    use std::path::Component;
    let relative = group.strip_prefix("/").ok()?;
    if relative
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
    {
        return None;
    }
    let leaf = root.join(relative);
    leaf.ancestors()
        .take_while(|path| path.starts_with(root))
        .filter_map(|path| {
            let maximum = read_to_string(path.join("memory.max"))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()?;
            let current = read_to_string(path.join("memory.current"))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()?;
            Some((maximum, maximum.saturating_sub(current)))
        })
        .reduce(|left, right| (left.0.min(right.0), left.1.min(right.1)))
}

#[cfg(test)]
#[path = "../tests/unit/host_memory.rs"]
mod tests;
