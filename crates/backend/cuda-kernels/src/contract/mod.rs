//! Device, geometry and kernel contracts shared by the kernel profiles.
//!
//! The host speaks only these types; `tile` and `aot` implement the traits behind them. A raw
//! [`DevicePtr`] carries no lifetime or aliasing guarantee by itself; the kernel implementation
//! wraps it in its own borrowed view for the duration of one launch.
pub mod kernels;
pub mod plan;
pub mod ptr;
pub mod target;

pub use kernels::Projection;
pub use plan::LinearPlan;
pub use ptr::DevicePtr;
pub use target::{Arch, Profile, Target};
