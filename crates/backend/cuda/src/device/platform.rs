//! Compile-time host platform facts for CUDA platform-specific artifacts.

/// Whether the host target is Linux on x86-64.
///
/// This does not establish that any library or device artifact is available or compatible.
pub const fn is_linux_x86_64() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
}
