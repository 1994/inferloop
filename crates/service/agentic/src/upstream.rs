//! The model side of the agentic layer: admitted generations, their cancellation, and the
//! rounds of a tool-loop turn.
use crate::document::ResponseDocument;
use infer_core::{RequestId, Result};
use infer_spi::{ToolName, ToolOutput};
use std::future::Future;

/// Runs admitted generations for one protocol request.
///
/// The protocol adapter implements this over the engine runtime: it has already encoded and
/// admitted the generation, and it turns the completed generation into its own wire document.
/// Tests substitute a scripted implementation.
pub trait Upstream: Send + Sync + 'static {
    /// One generation the engine has admitted.
    type Admitted: Send + 'static;
    /// The protocol response body this upstream produces.
    type Document: ResponseDocument;

    /// Wait for an admitted generation to finish and build its response body.
    ///
    /// # Errors
    /// Returns the backend, decoding or parsing error that ended the generation.
    fn complete(
        &self,
        admitted: Self::Admitted,
    ) -> impl Future<Output = Result<Self::Document>> + Send;

    /// Cancel an engine request.
    ///
    /// # Errors
    /// Returns an error when the request is unknown or has already stopped.
    fn cancel(&self, request: RequestId) -> impl Future<Output = Result<()>> + Send;
}

/// One tool call the model closed during a round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Protocol call identifier the matching [`ToolResult`] answers.
    pub call_id: Box<str>,
    pub name: ToolName,
    /// The model's arguments, as JSON text.
    pub arguments: Box<str>,
}

/// The output of one engine tool call, for the call with the same `call_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub call_id: Box<str>,
    pub output: ToolOutput,
}

/// Why a model round stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundStop {
    /// The model ended its reply (end of sequence or a stop sequence).
    Finished,
    /// The round reached its output token budget.
    OutputLimit,
}

/// The end of one model round.
#[derive(Debug)]
pub struct RoundFinish<O> {
    pub stop: RoundStop,
    pub output_tokens: u64,
    /// The round's model output, handed back to [`RoundUpstream::append_round`].
    pub output: O,
}

/// What a running round reports, in generation order.
#[derive(Debug)]
pub enum RoundEvent<O> {
    /// A tool call whose arguments are complete. Engine tools start running at once.
    Call(ToolCall),
    /// The round ended; no further event follows.
    Finished(RoundFinish<O>),
}

/// The events of one running round. Dropping it cancels the generation.
pub trait RoundEvents: Send {
    /// The round's model output, owned by the upstream's conversation format.
    type Output: Send;

    /// The next event. The loop stops reading after [`RoundEvent::Finished`]. The future need
    /// not be cancel-safe: the loop keeps it until it resolves.
    ///
    /// # Errors
    /// Returns the backend or parsing error that ended the round, including a stream that
    /// closed before finishing.
    fn next(&mut self) -> impl Future<Output = Result<RoundEvent<Self::Output>>> + Send;
}

/// Runs the model rounds of a tool-loop turn over a conversation the upstream owns.
///
/// The protocol adapter keeps the conversation in its own format (rendered messages, a token
/// ledger), starts each round from it and appends each round's output and tool results to it.
/// Tests substitute a scripted implementation.
pub trait RoundUpstream: Send + Sync {
    /// The conversation a turn extends, round by round.
    type Conversation: Send + Sync;
    /// The events of one round.
    type Events: RoundEvents;

    /// Start one round over the conversation. `max_output_tokens` is what remains of the turn's
    /// output budget, when it has one.
    ///
    /// # Errors
    /// Returns admission, encoding or capacity errors before any token is produced.
    fn round<'a>(
        &'a self,
        conversation: &'a Self::Conversation,
        max_output_tokens: Option<u64>,
    ) -> impl Future<Output = Result<Self::Events>> + Send + 'a;

    /// Append a finished round's output and the results of the engine calls it made, in call
    /// order. Calls without a result (client calls, or calls past a limit) stay unanswered.
    ///
    /// # Errors
    /// Returns an error when the output cannot be added to the conversation.
    fn append_round(
        &self,
        conversation: &mut Self::Conversation,
        output: <Self::Events as RoundEvents>::Output,
        results: Vec<ToolResult>,
    ) -> Result<()>;
}
