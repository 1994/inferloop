//! Compile-time host platform facts for CUDA platform-specific artifacts.
use std::fs::read_to_string;

/// Whether the host target is Linux on x86-64.
///
/// This does not establish that any library or device artifact is available or compatible.
pub const fn is_linux_x86_64() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
}

/// Installed NVIDIA driver build, when the host exposes a release identity.
///
/// This is distinct from the CUDA API version. Absence prevents a tuning cache hit.
pub fn driver_build() -> Option<String> {
    read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|text| parse_driver_build(&text).map(str::to_owned))
}

fn parse_driver_build(text: &str) -> Option<&str> {
    text.lines()
        .find(|line| line.starts_with("NVRM version:"))
        .map(str::trim)
}

#[cfg(test)]
#[path = "../../tests/unit/device_platform.rs"]
mod tests;
