//! Server-side agentic orchestration for the Responses protocol.
//!
//! This crate owns response lifecycles independently of any transport: persistence through an
//! installed [`infer_spi::ResponseStore`], chained-response validation, bounded background runs
//! and cancellation. Protocol adapters supply the wire document and admitted generations through
//! [`ResponseDocument`] and [`Upstream`]; they never touch the store directly.
mod constants;
mod document;
mod persist;
mod responses;
mod store;
mod upstream;
pub use document::{DocumentMetadata, FailedDocument, ResponseDocument, Retention};
pub use persist::StoreLease;
pub use responses::{BackgroundStart, Responses};
pub use store::{InMemoryResponseStore, InMemoryResponseStoreConfig};
pub use upstream::Upstream;
