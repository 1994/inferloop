//! Raw device address passed across the contract.
//!
//! The type carries no lifetime or aliasing guarantee. A kernel implementation wraps it in its
//! own borrowed view for exactly one launch; the host keeps ownership of the allocation.

/// A device address owned by the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);
