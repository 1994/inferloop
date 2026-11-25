//! Server-side agentic orchestration for the Responses protocol.
//!
//! This crate owns response lifecycles independently of any transport: persistence through an
//! installed [`infer_spi::ResponseStore`], chained-response validation, bounded background runs
//! and cancellation. Protocol adapters supply the wire document and admitted generations through
//! [`ResponseDocument`] and [`Upstream`]; they never touch the store directly.
//!
//! [`ToolLoop`] runs multi-round turns: the adapter's [`RoundUpstream`] runs each model round,
//! engine tools from an installed [`infer_spi::ToolExecutor`] run as soon as the model closes a
//! call, and the loop decides after each round whether to continue, hand control to the client,
//! or stop at a limit.
mod constants;
mod document;
mod persist;
mod responses;
mod store;
mod tools;
mod turn;
mod upstream;
pub use document::{DocumentMetadata, FailedDocument, ResponseDocument, Retention};
pub use persist::StoreLease;
pub use responses::{BackgroundStart, Responses};
pub use store::{InMemoryResponseStore, InMemoryResponseStoreConfig};
pub use tools::ToolSet;
pub use turn::{IncompleteCause, ToolLoop, Turn, TurnEnd, TurnLimits};
pub use upstream::{
    RoundEvent, RoundEvents, RoundFinish, RoundStop, RoundUpstream, ToolCall, ToolResult, Upstream,
};
