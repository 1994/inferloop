//! What a completed tuning preparation measured, and the device identity it is valid for.
//!
//! Platform differences stay here: callers name their invocation in opaque bytes and never see
//! how this backend identifies a driver or device.
use crate::device::CudaDevice;
use crate::tuning::table_path;
use std::fs::{read_to_string, rename, write};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::process::id;

/// Build line of the kernel driver. Where it is unavailable, tuning decisions stay in the
/// process instead of persisting without a driver identity.
#[cfg(target_os = "linux")]
pub(crate) fn driver_build() -> Option<String> {
    read_to_string("/proc/driver/nvidia/version")
        .ok()?
        .lines()
        .find(|line| line.starts_with("NVRM version:"))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn driver_build() -> Option<String> {
    None
}

/// Device and driver identity; `None` when the driver cannot be named, so nothing is reused.
fn device_identity() -> Option<Vec<u8>> {
    let device = CudaDevice::new(0).ok()?;
    let hardware = serde_json::to_string(device.profile().ok()?).ok()?;
    let version = CudaDevice::driver_version().ok()?;
    Some(format!("{hardware}:{version}:{}", driver_build()?).into_bytes())
}

/// The stamp lives beside the table it describes, so removing the cache removes both.
fn stamp_path(table: &Path, identity: &[Vec<u8>]) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    table.with_file_name(format!("prepared-{:016x}", hasher.finish()))
}

fn read_stamp(table: &Path, stamp: &Path) -> Option<usize> {
    if !table.exists() {
        return None;
    }
    read_to_string(stamp)
        .ok()?
        .trim()
        .parse()
        .ok()
        .filter(|&width| width > 0)
}

fn write_stamp(stamp: &Path, width: usize) -> io::Result<()> {
    let temporary = stamp.with_extension(format!("{}.tmp", id()));
    write(&temporary, width.to_string())?;
    rename(&temporary, stamp)
}

fn stamp_for(invocation: &[Vec<u8>]) -> Option<(PathBuf, PathBuf)> {
    let table = table_path()?;
    let mut identity = invocation.to_vec();
    identity.push(device_identity()?);
    let stamp = stamp_path(&table, &identity);
    Some((table, stamp))
}

/// The pool width a completed preparation of this invocation measured on this device.
#[must_use]
pub fn prepared_width(invocation: &[Vec<u8>]) -> Option<usize> {
    let (table, stamp) = stamp_for(invocation)?;
    read_stamp(&table, &stamp)
}

/// Record a finished preparation. A failure only means the next start prepares again.
pub fn record_prepared(invocation: &[Vec<u8>], width: usize) {
    let Some((_, stamp)) = stamp_for(invocation) else {
        return;
    };
    if let Err(error) = write_stamp(&stamp, width) {
        tracing::warn!(target: "infer::tuning", %error, "CUDA preparation stamp not written");
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_preparation.rs"]
mod tests;
