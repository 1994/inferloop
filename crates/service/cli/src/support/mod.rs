//! Shared CLI facilities, grouped by responsibility.
#[cfg(native_backend)]
mod engine;
#[cfg(native_backend)]
pub mod memory;
#[cfg(native_backend)]
mod serving;
#[cfg(native_backend)]
pub use engine::*;
mod json;
pub use json::*;
mod logging;
#[cfg(cuda_backend)]
pub use logging::human_bytes;
pub use logging::{flush as flush_logs, init as init_logging};
#[cfg(native_backend)]
mod fixtures;
#[cfg(native_backend)]
pub use fixtures::*;
#[cfg(native_backend)]
mod execution;
#[cfg(native_backend)]
pub use execution::*;
#[cfg(native_backend)]
mod server;
#[cfg(native_backend)]
pub use server::*;
#[cfg(native_backend)]
mod agent;

#[cfg(native_backend)]
pub use agent::agent;
