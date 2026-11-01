//! Where a kernel launch is recorded.
//!
//! The resident graph captures launches once and replays them, so a family method records and never
//! waits. The forms differ only in how an implementation reaches the capture: cuTile records
//! through its scope, a driver launch goes to a stream in capture mode.
#[cfg(all(feature = "tile", target_os = "linux"))]
use crate::contract::buffer::{Buffer, Owner};
#[cfg(all(feature = "tile", target_os = "linux"))]
use crate::tile::Scope;
#[cfg(all(feature = "tile", target_os = "linux"))]
use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
#[cfg(all(feature = "tile", target_os = "linux"))]
use std::sync::Arc;

/// Destination of one kernel launch.
pub enum Recording<'a> {
    /// A cuTile graph being captured.
    #[cfg(all(feature = "tile", target_os = "linux"))]
    Scope {
        /// The graph's capture scope.
        scope: &'a Scope,
        /// Filled with what the finished graph must keep alive; `None` when the caller owns it.
        owners: Option<&'a RefCell<Vec<Owner>>>,
    },
    /// A driver stream; capture mode records the launch, otherwise it is enqueued.
    Stream {
        /// Stream handle owned by the caller.
        handle: *mut c_void,
        /// Ties the handle to the caller's borrow of its stream.
        stream: PhantomData<&'a ()>,
    },
}

#[cfg(all(feature = "tile", target_os = "linux"))]
impl<'a> Recording<'a> {
    /// A capture whose caller keeps every buffer alive, as resident programs do today.
    #[must_use]
    pub const fn scope(scope: &'a Scope) -> Self {
        Self::Scope {
            scope,
            owners: None,
        }
    }

    /// A capture whose graph keeps every buffer alive.
    #[must_use]
    pub const fn retaining(scope: &'a Scope, owners: &'a RefCell<Vec<Owner>>) -> Self {
        Self::Scope {
            scope,
            owners: Some(owners),
        }
    }

    /// Keep this operand's memory alive for as long as the graph lives. Capture time only.
    pub fn retain<T>(&self, buffer: Buffer<'_, T>) {
        if let (
            Self::Scope {
                owners: Some(owners),
                ..
            },
            Some(owner),
        ) = (self, buffer.owner())
        {
            owners.borrow_mut().push(Arc::clone(owner));
        }
    }

    /// The capture scope, for recordings that have not moved behind a family yet.
    #[must_use]
    pub const fn cutile_scope(&self) -> Option<&'a Scope> {
        match self {
            Self::Scope { scope, .. } => Some(*scope),
            Self::Stream { .. } => None,
        }
    }
}

impl Recording<'_> {
    /// Driver stream to launch on, when this recording is one.
    #[must_use]
    pub const fn stream(&self) -> Option<*mut c_void> {
        match self {
            Self::Stream { handle, .. } => Some(*handle),
            #[cfg(all(feature = "tile", target_os = "linux"))]
            Self::Scope { .. } => None,
        }
    }
}
