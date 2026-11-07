//! The JIT surface of the cuTile build, answered for precompiled kernels: nothing compiles at
//! runtime, so the compiler identity is fixed and only the options the host records remain.
use infer_core::{Error, Result};
use std::ffi::OsString;
use std::fmt;

/// The fixed identity of precompiled kernels, where the cuTile build reports its compiler.
#[must_use]
#[expect(
    clippy::missing_const_for_fn,
    reason = "Not const in the cuTile build, so host callers compile under either source"
)]
pub fn compiler() -> &'static str {
    "aot"
}

/// No bytecode replaces precompiled kernels.
#[must_use]
#[expect(
    clippy::missing_const_for_fn,
    reason = "Not const in the cuTile build, so host callers compile under either source"
)]
pub fn bytecode_override() -> Option<OsString> {
    None
}

/// Nothing compiles at runtime, so there is no kernel cache to enable.
/// # Errors
/// Never fails; the signature matches the cuTile build's.
#[expect(
    clippy::missing_const_for_fn,
    reason = "Not const in the cuTile build, so host callers compile under either source"
)]
pub fn enable_cache() -> Result<()> {
    Ok(())
}

/// Programmatic dependent launch needs the cuTile runtime's launch attributes.
/// # Errors
/// Always: this kernel source does not provide it.
pub fn validate_dependent_launch(_ordinal: usize) -> Result<()> {
    Err(Error::unsupported(
        "programmatic dependent launch is not provided by the AOT kernel source",
    ))
}

/// The options a precompiled kernel was built with, as far as the host records them.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct JitOptions {
    occupancy: Option<i32>,
}

impl JitOptions {
    /// Options recording an occupancy hint.
    #[must_use]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "Not const in the cuTile build, so host callers compile under either source"
    )]
    pub fn occupancy(occupancy: i32) -> Self {
        Self {
            occupancy: Some(occupancy),
        }
    }

    /// The occupancy hint, when one is set.
    #[must_use]
    pub const fn occupancy_hint(&self) -> Option<i32> {
        self.occupancy
    }

    /// No level is pinned: precompiled kernels were built optimized.
    #[must_use]
    pub const fn opt_level(&self) -> Option<u8> {
        None
    }

    /// Precompiled kernels are never device-debug builds.
    #[must_use]
    pub const fn device_debug(&self) -> bool {
        false
    }
}

impl fmt::Debug for JitOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JitOptions")
            .field("occupancy", &self.occupancy)
            .finish()
    }
}
