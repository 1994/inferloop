//! Where a kernel launch is recorded.
//!
//! The resident graph captures launches once and replays them, so a family method records and never
//! waits. The two forms differ only in how an implementation reaches the capture: cuTile records
//! through its scope, a driver launch goes to a stream in capture mode.
#[cfg(all(feature = "tile", target_os = "linux"))]
use crate::tile::Scope;
use std::ffi::c_void;
use std::marker::PhantomData;

/// Destination of one kernel launch.
pub enum Recording<'a> {
    /// A cuTile graph being captured.
    #[cfg(all(feature = "tile", target_os = "linux"))]
    Scope(&'a Scope),
    /// A driver stream; capture mode records the launch, otherwise it is enqueued.
    Stream {
        /// Stream handle owned by the caller.
        handle: *mut c_void,
        /// Ties the handle to the caller's borrow of its stream.
        stream: PhantomData<&'a ()>,
    },
}

impl Recording<'_> {
    /// Driver stream to launch on, when this recording is one.
    #[must_use]
    pub const fn stream(&self) -> Option<*mut c_void> {
        match self {
            Self::Stream { handle, .. } => Some(*handle),
            #[cfg(all(feature = "tile", target_os = "linux"))]
            Self::Scope(_) => None,
        }
    }
}
