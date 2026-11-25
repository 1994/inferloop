//! Tool ownership for one request and the bounded execution of engine calls.
use crate::constants::MAX_TOOL_OUTPUT_BYTES;
use crate::upstream::{ToolCall, ToolResult};
use infer_core::{Error, Result};
use infer_spi::{ToolExecutor, ToolFuture, ToolInvocation, ToolName, ToolOutput, ToolSpec};
use std::collections::{BTreeMap, BTreeSet};
use std::future::{Future, poll_fn};
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use tokio::sync::Semaphore;
use tokio::time;

/// The tools one request may call, split by who runs them.
///
/// Engine tools come from the installed executor; client tools are the request's own function
/// tools, which the client runs. A name in both sets is rejected when the set is built.
#[derive(Debug, Clone)]
pub struct ToolSet {
    engine: Arc<BTreeMap<ToolName, ToolSpec>>,
    client: BTreeSet<ToolName>,
}

/// Who runs a tool the model called.
pub enum Owner<'a> {
    Engine(&'a ToolSpec),
    Client,
}

impl ToolSet {
    pub(crate) fn new(
        engine: Arc<BTreeMap<ToolName, ToolSpec>>,
        client: impl IntoIterator<Item = ToolName>,
    ) -> Result<Self> {
        let client: BTreeSet<_> = client.into_iter().collect();
        if let Some(name) = client.iter().find(|name| engine.contains_key(*name)) {
            return Err(Error::invalid(format!(
                "function tool {name} has the name of a server tool"
            )));
        }
        Ok(Self { engine, client })
    }

    /// The engine tools to offer the model next to the request's own tools.
    pub fn engine_tools(&self) -> impl Iterator<Item = &ToolSpec> {
        self.engine.values()
    }

    pub(crate) fn owner(&self, name: &ToolName) -> Option<Owner<'_>> {
        if let Some(spec) = self.engine.get(name) {
            return Some(Owner::Engine(spec));
        }
        self.client.contains(name).then_some(Owner::Client)
    }
}

type Running = Pin<Box<dyn Future<Output = ToolOutput> + Send>>;

enum Slot {
    Running(Box<str>, Running),
    Done(ToolResult),
}

/// The engine calls of one round, kept in call order while they run concurrently.
///
/// The calls are polled by the turn's own task, so a round never spawns tasks; dropping the
/// dispatch cancels every call still running.
#[derive(Default)]
pub struct Dispatch {
    slots: Vec<Slot>,
    running: usize,
}

impl Dispatch {
    /// Start one call. It waits for a process-wide permit, then runs under the tool's timeout.
    pub fn start(
        &mut self,
        executor: &Arc<dyn ToolExecutor>,
        permits: &Arc<Semaphore>,
        spec: &ToolSpec,
        call: ToolCall,
    ) {
        let call_id = call.call_id.clone();
        let future = run_call(Arc::clone(executor), Arc::clone(permits), spec, call);
        self.slots.push(Slot::Running(call_id, Box::pin(future)));
        self.running += 1;
    }

    pub const fn is_running(&self) -> bool {
        self.running > 0
    }

    /// Resolve once at least one running call has finished; pending forever when none runs.
    pub fn progress(&mut self) -> impl Future<Output = ()> + Send + '_ {
        poll_fn(move |context| {
            let mut finished = 0;
            for slot in &mut self.slots {
                let Slot::Running(call_id, future) = slot else {
                    continue;
                };
                if let Poll::Ready(output) = future.as_mut().poll(context) {
                    let call_id = mem::take(call_id);
                    *slot = Slot::Done(ToolResult { call_id, output });
                    finished += 1;
                }
            }
            self.running -= finished;
            if finished > 0 {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
    }

    /// Wait for every call and return the results in call order.
    pub async fn finish(mut self) -> Vec<ToolResult> {
        while self.is_running() {
            self.progress().await;
        }
        self.slots
            .into_iter()
            .filter_map(|slot| match slot {
                Slot::Done(result) => Some(result),
                Slot::Running(..) => None,
            })
            .collect()
    }
}

fn run_call(
    executor: Arc<dyn ToolExecutor>,
    permits: Arc<Semaphore>,
    spec: &ToolSpec,
    call: ToolCall,
) -> impl Future<Output = ToolOutput> + Send + 'static {
    let timeout = spec.timeout();
    let limit = spec.max_output_bytes().min(MAX_TOOL_OUTPUT_BYTES);
    async move {
        let Ok(_permit) = permits.acquire_owned().await else {
            return ToolOutput::Failed("the tool pool is shut down".to_owned());
        };
        let invocation = ToolInvocation {
            name: &call.name,
            arguments: &call.arguments,
        };
        let running: ToolFuture<'_> = executor.call(invocation);
        match time::timeout(timeout, running).await {
            Err(_) => ToolOutput::Failed(format!(
                "tool {} did not finish within {} ms",
                call.name,
                timeout.as_millis()
            )),
            Ok(ToolOutput::Text(text)) if text.len() > limit => ToolOutput::Failed(format!(
                "tool {} returned {} bytes, over the {limit}-byte limit",
                call.name,
                text.len()
            )),
            Ok(output) => output,
        }
    }
}
