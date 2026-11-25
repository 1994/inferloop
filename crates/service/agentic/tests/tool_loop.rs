//! Tool-loop turns against a scripted upstream and a mock tool executor.
use infer_agentic::{
    IncompleteCause, RoundEvent, RoundEvents, RoundFinish, RoundStop, RoundUpstream, ToolCall,
    ToolLoop, ToolResult, Turn, TurnEnd, TurnLimits,
};
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{
    ResponsesExtensions, ToolEffect, ToolExecutor, ToolFuture, ToolInvocation, ToolName,
    ToolOutput, ToolSpec,
};
use std::collections::VecDeque;
use std::future::{Future, pending, ready};
use std::pin::pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use tokio::sync::{Notify, mpsc};
use tokio::time::{sleep, timeout};

const SCHEMA: &str = r#"{"type":"object"}"#;
const TIMEOUT: Duration = Duration::from_secs(5);
const SHORT_TIMEOUT: Duration = Duration::from_millis(20);
const OUTPUT_BYTES: usize = 64;
const ROUND_TOKENS: u64 = 10;

/// One scripted round: the calls the model makes, then how it stops.
#[derive(Clone)]
struct Script {
    calls: Vec<(&'static str, &'static str)>,
    stop: RoundStop,
    /// Hold `Finished` until this many tool calls have started.
    finish_after_starts: usize,
}

fn round(calls: &[(&'static str, &'static str)]) -> Script {
    Script {
        calls: calls.to_vec(),
        stop: RoundStop::Finished,
        finish_after_starts: 0,
    }
}

/// The conversation: each appended round with its results, and the budget each round saw.
#[derive(Debug, Default)]
struct Log {
    rounds: Vec<(usize, Vec<ToolResult>)>,
    budgets: Vec<Option<u64>>,
}

struct Upstream {
    scripts: Mutex<VecDeque<Script>>,
    budgets: Mutex<Vec<Option<u64>>>,
    executor: Arc<Executor>,
}

struct Events {
    receiver: mpsc::UnboundedReceiver<RoundEvent<usize>>,
}

impl RoundEvents for Events {
    type Output = usize;
    async fn next(&mut self) -> Result<RoundEvent<usize>> {
        self.receiver
            .recv()
            .await
            .ok_or_else(|| Error::new(ErrorCode::Backend, "round closed"))
    }
}

impl RoundUpstream for Upstream {
    type Conversation = Log;
    type Events = Events;

    fn round<'a>(
        &'a self,
        conversation: &'a Log,
        max_output_tokens: Option<u64>,
    ) -> impl Future<Output = Result<Events>> + Send + 'a {
        ready(self.start(conversation, max_output_tokens))
    }

    fn append_round(
        &self,
        conversation: &mut Log,
        output: usize,
        results: Vec<ToolResult>,
    ) -> Result<()> {
        conversation.rounds.push((output, results));
        Ok(())
    }
}

impl Upstream {
    fn start(&self, conversation: &Log, max_output_tokens: Option<u64>) -> Result<Events> {
        self.budgets
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(max_output_tokens);
        let script = self
            .scripts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
            .ok_or_else(|| Error::invalid("no scripted round left"))?;
        let index = conversation.rounds.len();
        let (sender, receiver) = mpsc::unbounded_channel();
        let executor = Arc::clone(&self.executor);
        tokio::spawn(async move {
            for (position, (name, arguments)) in script.calls.iter().enumerate() {
                let call = ToolCall {
                    call_id: format!("r{index}c{position}").into(),
                    name: ToolName::new(name)?,
                    arguments: (*arguments).into(),
                };
                if sender.send(RoundEvent::Call(call)).is_err() {
                    return Ok::<(), Error>(());
                }
            }
            loop {
                let mut notified = pin!(executor.start_signal.notified());
                notified.as_mut().enable();
                if executor.started.load(Ordering::SeqCst) >= script.finish_after_starts {
                    break;
                }
                notified.await;
            }
            let finish = RoundFinish {
                stop: script.stop,
                output_tokens: ROUND_TOKENS,
                output: index,
            };
            let _closed = sender.send(RoundEvent::Finished(finish));
            Ok(())
        });
        Ok(Events { receiver })
    }
}

/// Engine tools: `echo` returns its arguments after an optional per-argument delay, `slow`
/// never finishes, `big` returns more than its output limit.
#[derive(Default)]
struct Executor {
    tools: Vec<ToolSpec>,
    started: AtomicUsize,
    running: AtomicUsize,
    peak: AtomicUsize,
    dropped: AtomicUsize,
    start_signal: Notify,
}

struct RunningGuard<'a>(&'a Executor);

impl Drop for RunningGuard<'_> {
    fn drop(&mut self) {
        self.0.running.fetch_sub(1, Ordering::SeqCst);
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl ToolExecutor for Executor {
    fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    fn call<'a>(&'a self, invocation: ToolInvocation<'a>) -> ToolFuture<'a> {
        Box::pin(async move {
            self.started.fetch_add(1, Ordering::SeqCst);
            let running = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(running, Ordering::SeqCst);
            self.start_signal.notify_waiters();
            let _guard = RunningGuard(self);
            match invocation.name.as_str() {
                "slow" => pending().await,
                "big" => ToolOutput::Text("x".repeat(OUTPUT_BYTES + 1)),
                _ => {
                    let delay = invocation.arguments.parse().unwrap_or(0);
                    sleep(Duration::from_millis(delay)).await;
                    ToolOutput::Text(format!("echo {}", invocation.arguments))
                }
            }
        })
    }
}

fn spec(name: &str, timeout: Duration) -> Result<ToolSpec> {
    ToolSpec::new(
        ToolName::new(name)?,
        "",
        SCHEMA,
        ToolEffect::ReadOnly,
        timeout,
        OUTPUT_BYTES,
    )
}

struct Harness {
    tool_loop: ToolLoop,
    upstream: Upstream,
    executor: Arc<Executor>,
}

fn harness(scripts: Vec<Script>) -> Result<Harness> {
    let executor = Arc::new(Executor {
        tools: vec![
            spec("echo", TIMEOUT)?,
            spec("slow", SHORT_TIMEOUT)?,
            spec("big", TIMEOUT)?,
        ],
        ..Executor::default()
    });
    let extensions =
        ResponsesExtensions::new().with_tools(Arc::clone(&executor) as Arc<dyn ToolExecutor>);
    extensions.validate()?;
    Ok(Harness {
        tool_loop: ToolLoop::new(&extensions),
        upstream: Upstream {
            scripts: Mutex::new(scripts.into()),
            budgets: Mutex::default(),
            executor: Arc::clone(&executor),
        },
        executor,
    })
}

impl Harness {
    async fn run(&self, limits: TurnLimits) -> Result<Turn<Log>> {
        let tools = self.tool_loop.tool_set([ToolName::new("client_search")?])?;
        let mut turn = self
            .tool_loop
            .run(&self.upstream, &tools, Log::default(), limits)
            .await?;
        let budgets = self
            .upstream
            .budgets
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        turn.conversation.budgets.clone_from(&budgets);
        drop(budgets);
        Ok(turn)
    }
}

fn unlimited() -> TurnLimits {
    TurnLimits::new(None, None)
}

fn outputs(results: &[ToolResult]) -> Vec<(&str, &ToolOutput)> {
    results
        .iter()
        .map(|result| (&*result.call_id, &result.output))
        .collect()
}

fn text(value: &str) -> ToolOutput {
    ToolOutput::Text(value.to_owned())
}

#[tokio::test]
async fn engine_calls_continue_and_results_keep_call_order() -> Result<()> {
    // The first call finishes last; results still follow call order.
    let harness = harness(vec![round(&[("echo", "60"), ("echo", "0")]), round(&[])])?;
    let turn = harness.run(unlimited()).await?;
    assert_eq!(
        (turn.end, turn.rounds, turn.tool_calls),
        (TurnEnd::Done, 2, 2)
    );
    let (_, first) = &turn.conversation.rounds[0];
    assert_eq!(
        outputs(first),
        vec![("r0c0", &text("echo 60")), ("r0c1", &text("echo 0"))]
    );
    assert_eq!(turn.conversation.rounds[1].1, Vec::new());
    assert_eq!(harness.executor.peak.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn calls_start_before_the_round_finishes() -> Result<()> {
    let mut first = round(&[("echo", "0")]);
    first.finish_after_starts = 1;
    let harness = harness(vec![first, round(&[])])?;
    // Without early dispatch the round never finishes, so bound the wait.
    let turn = timeout(TIMEOUT, harness.run(unlimited()))
        .await
        .map_err(|_| Error::invalid("the call did not start before the round finished"))??;
    assert_eq!(turn.end, TurnEnd::Done);
    Ok(())
}

#[tokio::test]
async fn a_client_call_returns_control_with_engine_results() -> Result<()> {
    let harness = harness(vec![round(&[("echo", "0"), ("client_search", "{}")])])?;
    let turn = harness.run(unlimited()).await?;
    assert_eq!((turn.end, turn.rounds), (TurnEnd::ClientAction, 1));
    assert_eq!(
        outputs(&turn.conversation.rounds[0].1),
        vec![("r0c0", &text("echo 0"))]
    );
    Ok(())
}

#[tokio::test]
async fn the_last_round_does_not_run_engine_calls() -> Result<()> {
    let scripts = vec![round(&[("echo", "0")]); 10];
    let harness = harness(scripts)?;
    let turn = harness.run(unlimited()).await?;
    assert_eq!(turn.end, TurnEnd::Incomplete(IncompleteCause::RoundLimit));
    assert_eq!((turn.rounds, turn.tool_calls), (10, 9));
    assert_eq!(turn.conversation.rounds[9].1, Vec::new());
    Ok(())
}

#[tokio::test]
async fn calls_past_the_request_cap_are_not_run() -> Result<()> {
    let harness = harness(vec![
        round(&[("echo", "0")]),
        round(&[("echo", "0"), ("echo", "0")]),
    ])?;
    let turn = harness.run(TurnLimits::new(Some(2), None)).await?;
    assert_eq!(
        turn.end,
        TurnEnd::Incomplete(IncompleteCause::ToolCallLimit)
    );
    assert_eq!(turn.tool_calls, 2);
    assert_eq!(
        outputs(&turn.conversation.rounds[1].1),
        vec![("r1c0", &text("echo 0"))]
    );
    Ok(())
}

#[tokio::test]
async fn the_output_budget_spans_rounds() -> Result<()> {
    let harness = harness(vec![
        round(&[("echo", "0")]),
        round(&[("echo", "0")]),
        round(&[]),
    ])?;
    let turn = harness
        .run(TurnLimits::new(None, Some(2 * ROUND_TOKENS)))
        .await?;
    assert_eq!(turn.end, TurnEnd::Incomplete(IncompleteCause::OutputLimit));
    assert_eq!(
        turn.conversation.budgets,
        vec![Some(2 * ROUND_TOKENS), Some(ROUND_TOKENS)]
    );
    assert_eq!(turn.output_tokens, 2 * ROUND_TOKENS);
    Ok(())
}

#[tokio::test]
async fn a_truncated_round_ends_incomplete() -> Result<()> {
    let mut cut = round(&[("echo", "0")]);
    cut.stop = RoundStop::OutputLimit;
    let harness = harness(vec![cut])?;
    let turn = harness.run(unlimited()).await?;
    assert_eq!(turn.end, TurnEnd::Incomplete(IncompleteCause::OutputLimit));
    assert_eq!(turn.conversation.rounds[0].1.len(), 1);
    Ok(())
}

#[tokio::test]
async fn timeouts_and_oversized_outputs_fail_the_call_not_the_turn() -> Result<()> {
    let harness = harness(vec![round(&[("slow", "{}"), ("big", "{}")]), round(&[])])?;
    let turn = harness.run(unlimited()).await?;
    assert_eq!(turn.end, TurnEnd::Done);
    let results = &turn.conversation.rounds[0].1;
    let failed: Vec<_> = results
        .iter()
        .map(|result| matches!(result.output, ToolOutput::Failed(_)))
        .collect();
    assert_eq!(failed, vec![true, true]);
    assert_eq!(harness.executor.dropped.load(Ordering::SeqCst), 2);
    Ok(())
}

#[tokio::test]
async fn an_undeclared_tool_fails_the_turn() -> Result<()> {
    let harness = harness(vec![round(&[("unknown", "{}")])])?;
    let error = harness.run(unlimited()).await.err().map(|error| error.code);
    assert_eq!(error, Some(ErrorCode::Backend));
    Ok(())
}

#[tokio::test]
async fn a_function_tool_cannot_shadow_a_server_tool() -> Result<()> {
    let harness = harness(Vec::new())?;
    let error = harness
        .tool_loop
        .tool_set([ToolName::new("echo")?])
        .err()
        .map(|error| error.code);
    assert_eq!(error, Some(ErrorCode::InvalidInput));
    Ok(())
}

#[tokio::test]
async fn dropping_the_turn_cancels_running_calls() -> Result<()> {
    let harness = harness(vec![round(&[("echo", "60000")])])?;
    let tools = harness.tool_loop.tool_set([])?;
    let turn = harness
        .tool_loop
        .run(&harness.upstream, &tools, Log::default(), unlimited());
    let stopped = timeout(SHORT_TIMEOUT * 10, turn).await;
    assert!(stopped.is_err(), "the call should still be running");
    assert_eq!(harness.executor.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(harness.executor.running.load(Ordering::SeqCst), 0);
    Ok(())
}

#[tokio::test]
async fn without_an_executor_every_turn_is_one_round() -> Result<()> {
    let tool_loop = ToolLoop::new(&ResponsesExtensions::new());
    let upstream = harness(vec![round(&[("client_search", "{}")])])?.upstream;
    let tools = tool_loop.tool_set([ToolName::new("client_search")?])?;
    let turn = tool_loop
        .run(&upstream, &tools, Log::default(), unlimited())
        .await?;
    assert_eq!((turn.end, turn.rounds), (TurnEnd::ClientAction, 1));
    Ok(())
}
