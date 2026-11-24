//! Transport-independent, typed control commands for inference diagnostics and experiments.
mod commands;
mod constants;
mod context;
pub mod experiment;
pub mod graph;
pub mod protocol;
pub mod registry;
mod service;
pub mod transport;

pub use context::{AgentBackend, AgentContext, CallRecord};
pub use service::AgentDispatcher;

#[cfg(test)]
#[path = "../tests/unit/cases.rs"]
mod tests;
