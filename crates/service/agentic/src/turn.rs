//! The tool loop: model rounds, engine tool calls between them, and the decision after each.
use crate::constants::{MAX_CONCURRENT_TOOL_CALLS, MAX_TOOL_CALLS, MAX_TOOL_ROUNDS};
use crate::tools::{Dispatch, Owner, ToolSet};
use crate::upstream::{
    RoundEvent, RoundEvents, RoundFinish, RoundStop, RoundUpstream, ToolCall, ToolResult,
};
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{ResponsesExtensions, ToolExecutor, ToolName, ToolSpec};
use std::collections::BTreeMap;
use std::pin::pin;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Why a turn stopped before the model finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompleteCause {
    /// The turn's output token budget ran out.
    OutputLimit,
    /// The model called engine tools in the last allowed round.
    RoundLimit,
    /// The model called more engine tools than the turn may execute.
    ToolCallLimit,
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEnd {
    /// The model answered without calling a tool.
    Done,
    /// The model called a client tool; the client runs it and continues the conversation.
    /// Engine calls of the same round have already run and their results are appended.
    ClientAction,
    /// A limit stopped the turn; the conversation holds everything produced so far.
    Incomplete(IncompleteCause),
}

/// Limits of one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnLimits {
    rounds: u32,
    tool_calls: u32,
    output_tokens: Option<u64>,
}

impl TurnLimits {
    /// Limits for one request. `max_tool_calls` is the request's cap on engine tool calls and
    /// `max_output_tokens` its output budget across all rounds. The service caps of 10 rounds
    /// and 64 engine tool calls per turn always apply.
    #[must_use]
    pub fn new(max_tool_calls: Option<u32>, max_output_tokens: Option<u64>) -> Self {
        Self {
            rounds: MAX_TOOL_ROUNDS,
            tool_calls: max_tool_calls.map_or(MAX_TOOL_CALLS, |cap| cap.min(MAX_TOOL_CALLS)),
            output_tokens: max_output_tokens,
        }
    }
}

/// A finished turn and the conversation it extended.
#[derive(Debug)]
pub struct Turn<C> {
    pub conversation: C,
    pub end: TurnEnd,
    pub rounds: u32,
    /// Engine tool calls that ran.
    pub tool_calls: u32,
    pub output_tokens: u64,
}

/// Runs tool-loop turns against the installed tool executor.
///
/// Cloning shares the executor and the process-wide bound on concurrent tool calls. Without an
/// executor every tool belongs to the client, so a turn is a single round.
#[derive(Clone)]
pub struct ToolLoop {
    executor: Option<Arc<dyn ToolExecutor>>,
    engine: Arc<BTreeMap<ToolName, ToolSpec>>,
    permits: Arc<Semaphore>,
}

impl ToolLoop {
    /// Use the executor installed in `extensions`, if any. Call
    /// [`ResponsesExtensions::validate`] first; a duplicate tool name keeps its first spec.
    #[must_use]
    pub fn new(extensions: &ResponsesExtensions) -> Self {
        let executor = extensions.tools().cloned();
        let mut engine = BTreeMap::new();
        if let Some(executor) = &executor {
            for spec in executor.tools() {
                engine
                    .entry(spec.name().clone())
                    .or_insert_with(|| spec.clone());
            }
        }
        Self {
            executor,
            engine: Arc::new(engine),
            permits: Arc::new(Semaphore::new(MAX_CONCURRENT_TOOL_CALLS)),
        }
    }

    /// The tools of one request, given the names of its own function tools.
    ///
    /// # Errors
    /// Returns an invalid-input error when a function tool has the name of a server tool.
    pub fn tool_set(&self, client: impl IntoIterator<Item = ToolName>) -> Result<ToolSet> {
        ToolSet::new(Arc::clone(&self.engine), client)
    }

    /// Run rounds until the model answers, needs the client, or reaches a limit.
    ///
    /// Engine calls start as soon as the model closes them and run concurrently; their results
    /// are appended in call order. Dropping the returned future cancels the running round and
    /// every running call.
    ///
    /// # Errors
    /// Returns the upstream's error, or a backend error when the model calls a tool the request
    /// did not declare. The conversation is dropped with the error.
    pub async fn run<U: RoundUpstream>(
        &self,
        upstream: &U,
        tools: &ToolSet,
        mut conversation: U::Conversation,
        limits: TurnLimits,
    ) -> Result<Turn<U::Conversation>> {
        let mut tally = Tally::default();
        loop {
            let remaining = limits
                .output_tokens
                .map(|budget| budget.saturating_sub(tally.output_tokens));
            if remaining == Some(0) {
                return Ok(tally.end(
                    conversation,
                    TurnEnd::Incomplete(IncompleteCause::OutputLimit),
                ));
            }
            let last = tally.rounds + 1 >= limits.rounds;
            let events = upstream.round(&conversation, remaining).await?;
            let round = self.round(events, tools, &limits, &tally, last).await?;
            tally.rounds += 1;
            tally.tool_calls += round.calls.started;
            tally.output_tokens = tally
                .output_tokens
                .saturating_add(round.finish.output_tokens);
            let decision = round.decision();
            upstream.append_round(&mut conversation, round.finish.output, round.results)?;
            if let Some(end) = decision {
                tracing::debug!(
                    target: "infer::agentic",
                    rounds = tally.rounds,
                    tool_calls = tally.tool_calls,
                    output_tokens = tally.output_tokens,
                    end = ?end,
                    "tool loop turn finished"
                );
                return Ok(tally.end(conversation, end));
            }
        }
    }

    async fn round<E: RoundEvents>(
        &self,
        mut events: E,
        tools: &ToolSet,
        limits: &TurnLimits,
        tally: &Tally,
        last: bool,
    ) -> Result<Round<E::Output>> {
        let mut dispatch = Dispatch::default();
        let mut calls = Calls::default();
        let finish = loop {
            let event = {
                let mut next = pin!(events.next());
                loop {
                    tokio::select! {
                        biased;
                        event = &mut next => break event?,
                        () = dispatch.progress(), if dispatch.is_running() => {}
                    }
                }
            };
            match event {
                RoundEvent::Call(call) => {
                    let budget = limits.tool_calls.saturating_sub(tally.tool_calls);
                    self.place(&mut dispatch, &mut calls, tools, call, budget, last)?;
                }
                RoundEvent::Finished(finish) => break finish,
            }
        };
        drop(events);
        let results = dispatch.finish().await;
        Ok(Round {
            finish,
            results,
            calls,
        })
    }

    fn place(
        &self,
        dispatch: &mut Dispatch,
        calls: &mut Calls,
        tools: &ToolSet,
        call: ToolCall,
        budget: u32,
        last: bool,
    ) -> Result<()> {
        match tools.owner(&call.name) {
            None => Err(Error::new(
                ErrorCode::Backend,
                format!("the model called undeclared tool {}", call.name),
            )),
            Some(Owner::Client) => {
                calls.client = true;
                Ok(())
            }
            Some(Owner::Engine(spec)) => {
                let Some(executor) = &self.executor else {
                    return Err(Error::invariant("engine tool without an executor"));
                };
                if last {
                    calls.over_round_cap = true;
                } else if calls.started >= budget {
                    calls.over_call_cap = true;
                } else {
                    dispatch.start(executor, &self.permits, spec, call);
                    calls.started += 1;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Default)]
struct Tally {
    rounds: u32,
    tool_calls: u32,
    output_tokens: u64,
}

impl Tally {
    const fn end<C>(self, conversation: C, end: TurnEnd) -> Turn<C> {
        Turn {
            conversation,
            end,
            rounds: self.rounds,
            tool_calls: self.tool_calls,
            output_tokens: self.output_tokens,
        }
    }
}

#[derive(Debug, Default)]
struct Calls {
    started: u32,
    client: bool,
    over_round_cap: bool,
    over_call_cap: bool,
}

struct Round<O> {
    finish: RoundFinish<O>,
    results: Vec<ToolResult>,
    calls: Calls,
}

impl<O> Round<O> {
    /// How the turn ends after this round, or `None` to continue with the tool results.
    const fn decision(&self) -> Option<TurnEnd> {
        if matches!(self.finish.stop, RoundStop::OutputLimit) {
            return Some(TurnEnd::Incomplete(IncompleteCause::OutputLimit));
        }
        if self.calls.over_call_cap {
            return Some(TurnEnd::Incomplete(IncompleteCause::ToolCallLimit));
        }
        if self.calls.over_round_cap {
            return Some(TurnEnd::Incomplete(IncompleteCause::RoundLimit));
        }
        if self.calls.client {
            return Some(TurnEnd::ClientAction);
        }
        if self.calls.started == 0 {
            return Some(TurnEnd::Done);
        }
        None
    }
}
