//! Device, geometry and kernel contracts shared by the kernel sources.
//!
//! The host speaks only these types; `tile` and `aot` implement the traits behind them. A raw
//! [`DevicePtr`] carries no lifetime or aliasing guarantee by itself; the kernel implementation
//! wraps it in its own view for the duration of one recording.
pub mod kernels;
pub mod plan;
pub mod ptr;
pub mod recording;
pub mod target;

pub use kernels::Projection;
pub use plan::PoolGemvPlan;
pub use ptr::DevicePtr;
pub use recording::Recording;
pub use target::Arch;
