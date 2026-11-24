//! Host-only fixtures shared by the runtime, frontdoor and agent test targets.
//!
//! The doubles declare a model, its operations and a backend that completes work with fixed
//! outputs, so a scene can exercise admission, completion identity, checkpointing or resource
//! ownership without a full inference executor. Nothing here runs a model or a device.
pub mod fixtures;
pub mod rank_group;
