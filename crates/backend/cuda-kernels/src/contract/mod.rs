//! Device, geometry and kernel contracts shared by the kernel sources.
//!
//! The host speaks only these types; `tile` and `aot` implement the traits behind them.
pub mod buffer;
pub mod kernels;
pub mod plan;
pub mod recording;
pub mod target;

pub use buffer::{Bf16, Buffer, Owner};
pub use kernels::{Kernels, Projection, Rope, RopeBuffers};
pub use plan::{PoolGemvPlan, RopeMode, RopePlan};
pub use recording::Recording;
pub use target::Arch;
