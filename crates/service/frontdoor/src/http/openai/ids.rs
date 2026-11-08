//! Wire identifiers that a client replays inside its own conversation.
//!
//! `RequestId` is process-local and starts at 1 again after a restart, so an identifier built
//! from it alone names two different calls in one client conversation once the server has been
//! restarted. A client that keeps its history replays both, and the request is rejected for
//! duplicate ids. Every replayed identifier is therefore a version 7 UUID: unique per response,
//! ordered by creation time, and independent of any process-local counter.

use uuid::Uuid;

/// Identifier of one generated tool call.
#[must_use]
pub(super) fn tool_call() -> String {
    format!("call_{}", Uuid::now_v7().simple())
}

/// Identifier of one response.
#[must_use]
pub(super) fn response() -> String {
    format!("resp_{}", Uuid::now_v7().simple())
}

#[cfg(test)]
#[path = "../../../tests/unit/ids.rs"]
mod tests;
