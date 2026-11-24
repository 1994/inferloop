//! Logical sequence ownership, physical KV leases and prefix-cache management.
pub mod blocks;
pub mod cache;
pub mod kv;
mod logical;
pub mod physical;
pub use logical::{PrefixKey, SequenceState, SequenceStates, StateSnapshot};
