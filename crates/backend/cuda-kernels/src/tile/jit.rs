//! The JIT compiler's identity and options, as tuning identities record them and checks read them.
//!
//! Tuning keys embed these values in their own formats, so each is exposed as the compiler reports
//! it: changing a formatting here would invalidate every cached entry.
use cutile::cutile_compiler::cuda_tile_runtime_utils as compiler;
use cutile::cutile_compiler::hints::CompileOptions;
use std::ffi::OsString;
use std::fmt;

/// The installed tile compiler's fingerprint.
#[must_use]
pub fn compiler() -> &'static str {
    compiler::tileiras_fingerprint()
}

/// A bytecode file that replaces the compiler's output, when one is configured.
#[must_use]
pub fn bytecode_override() -> Option<OsString> {
    compiler::bytecode_override()
}

/// Compiler options for one kernel; `Debug` prints exactly what the compiler's options print.
#[derive(Clone, Default)]
pub struct JitOptions(CompileOptions);

impl JitOptions {
    /// The defaults with the compiler's occupancy hint fixed to `occupancy`.
    #[must_use]
    pub fn occupancy(occupancy: i32) -> Self {
        Self(CompileOptions::default().occupancy(occupancy))
    }

    /// The occupancy hint, when one is set.
    #[must_use]
    pub const fn occupancy_hint(&self) -> Option<i32> {
        self.0.occupancy
    }

    /// The pinned `tileiras` optimization level; `None` leaves the compiler's default.
    #[must_use]
    pub const fn opt_level(&self) -> Option<u8> {
        self.0.opt_level
    }

    /// Whether kernels compile for device debugging.
    #[must_use]
    pub const fn device_debug(&self) -> bool {
        self.0.device_debug
    }
}

impl fmt::Debug for JitOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, formatter)
    }
}
