//! Handle responsibilities.
use super::{Command, RuntimeHandle, async_mpsc, mpsc, oneshot, preparation_bytes};
use crate::cpu::CpuContext;
use crate::cpu::CpuInspection;
use crate::cpu::CpuPool;
use crate::cpu::StagingPlan;
use crate::cpu::StagingQuote;
use crate::ingress::Abandon;
use crate::ingress::Pending;
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_ir::CanonicalRequest;
use infer_observe::{ObservationQuery, trace::TraceContext};
use infer_runtime::{EngineOutput, RuntimeInspection};
use std::sync::Arc;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::{time::Duration, time::Instant};
use tokio::time::{Instant as TokioInstant, timeout, timeout_at};

/// Capacity of the output channel carrying each accepted request's engine events.
const OUTPUT_CHANNEL_CAPACITY: usize = 16;
/// Staging budget reserved for a metrics observation response, in bytes.
const METRICS_RESPONSE_BUDGET_BYTES: usize = 64 << 10;
/// Staging budget reserved for a diagnostics observation response, in bytes.
const DIAGNOSTICS_RESPONSE_BUDGET_BYTES: usize = 2 << 20;
/// Staging bytes reserved per requested semantic event.
const EVENT_RESPONSE_BYTES: usize = 1024;
/// Fixed staging bytes reserved for an event query in addition to its per-event budget.
const EVENTS_RESPONSE_BASE_BYTES: usize = 256 << 10;
/// Seconds to wait for the actor to acknowledge shutdown.
const SHUTDOWN_ACK_TIMEOUT_SECS: u64 = 5;

#[allow(
    clippy::large_enum_variant,
    reason = "Keep the funded image path inline instead of allocating an asynchronous result box"
)]
enum InitialStage<T> {
    Ready(infer_runtime::PreparedRequest),
    Deferred(T, StagingQuote, infer_runtime::RequestPreparer),
}

// Fields drop in declaration order: staged input must disappear before its ingress owner.
struct RetainedStage<T> {
    input: T,
    pending: Arc<Pending>,
}
impl<T> RetainedStage<T> {
    fn with_owner<R>(self, work: impl FnOnce(T, &Pending) -> Result<R>) -> Result<R> {
        let Self { input, pending } = self;
        work(input, &pending)
    }
}

impl RuntimeHandle {
    #[must_use]
    pub fn vocabulary(&self) -> usize {
        self.preparer.vocabulary()
    }
    /// Allocate a process-local request identity shared by all clones of this handle.
    /// Explicit submissions advance the same counter; allocated identities are never reused.
    /// # Errors
    /// Returns capacity if the request identity space is exhausted.
    pub fn allocate_request_id(&self) -> Result<RequestId> {
        use std::sync::atomic::Ordering;
        let mut previous = self.request_ids.load(Ordering::Relaxed);
        loop {
            let next = previous
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorCode::Capacity, "request identities exhausted"))?;
            match self.request_ids.compare_exchange_weak(
                previous,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return RequestId::new(next),
                Err(current) => previous = current,
            }
        }
    }
    #[must_use]
    pub fn cpu_pool(&self) -> CpuPool {
        self.cpu.clone()
    }
    #[must_use]
    pub fn delivery_pool(&self) -> CpuPool {
        self.delivery.clone()
    }
    #[must_use]
    pub fn cpu_inspection(&self) -> CpuInspection {
        self.cpu.inspect()
    }
    pub(super) async fn reply<T>(&self, response: oneshot::Receiver<T>) -> Result<T> {
        Self::reply_with_budget(
            response,
            Duration::from_millis(self.config.command_timeout_ms),
        )
        .await
    }

    async fn reply_with_budget<T>(response: oneshot::Receiver<T>, budget: Duration) -> Result<T> {
        timeout(budget, response)
            .await
            .map_err(|_| {
                Error::new(
                    ErrorCode::Backend,
                    "runtime command acknowledgement timed out",
                )
            })?
            .map_err(|_| Error::new(ErrorCode::Backend, "engine actor stopped"))
    }
    pub(super) fn send(&self, command: Command) -> Result<()> {
        let sender = if matches!(command, Command::Submit { .. }) {
            &self.commands
        } else {
            &self.control
        };
        sender.try_send(command).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => {
                Error::new(ErrorCode::Capacity, "frontdoor command queue full")
            }
            mpsc::TrySendError::Disconnected(_) => {
                Error::new(ErrorCode::Backend, "engine actor stopped")
            }
        })?;
        self.wake.unpark();
        Ok(())
    }
    /// # Errors
    /// Returns an invalid-input, capacity, or conflict error if the runtime rejects admission or its command channel is closed.
    pub async fn submit(
        &self,
        request: CanonicalRequest,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        self.submit_with_trace(request, None).await
    }
    /// # Errors
    /// Returns admission, capacity, conflict, or actor lifecycle errors.
    pub async fn submit_with_trace(
        &self,
        request: CanonicalRequest,
        trace: Option<TraceContext>,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        let bytes = preparation_bytes(&request)?;
        let id = request.id;
        self.submit_preparing_until(id, bytes, trace, request.qos.deadline_us, move |_| {
            Ok(request)
        })
        .await
    }
    /// Perform text/workload preparation under the same request identity and bounded ingress credit.
    /// # Errors
    /// Returns duplicate, capacity, cancellation, preparation or engine admission errors.
    pub async fn submit_preparing(
        &self,
        id: RequestId,
        bytes: usize,
        trace: Option<TraceContext>,
        work: impl FnOnce(&CpuContext) -> Result<CanonicalRequest> + Send + 'static,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        self.submit_preparing_until(id, bytes, trace, None, work)
            .await
    }
    /// # Errors
    /// Returns identity, byte budget, deadline, preparation or admission errors.
    pub async fn submit_preparing_until(
        &self,
        id: RequestId,
        bytes: usize,
        trace: Option<TraceContext>,
        deadline_us: Option<u64>,
        work: impl FnOnce(&CpuContext) -> Result<CanonicalRequest> + Send + 'static,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        self.submit_constrained_preparing(id, bytes, trace, deadline_us, move |context| {
            Ok((work(context)?, None))
        })
        .await
    }
    /// Prepare a request and optional matcher on the bounded CPU pool.
    /// # Errors
    /// Returns preparation, constraint capability, or admission errors.
    pub async fn submit_constrained_preparing(
        &self,
        id: RequestId,
        bytes: usize,
        trace: Option<TraceContext>,
        deadline_us: Option<u64>,
        work: impl FnOnce(
            &CpuContext,
        ) -> Result<(CanonicalRequest, Option<infer_spi::ConstraintHandle>)>
        + Send
        + 'static,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        let deadline = self.preparation_deadline(deadline_us);
        let pending = self.ingress.reserve(id, bytes)?;
        self.request_ids
            .fetch_max(id.get(), AtomicOrdering::Relaxed);
        let _abandon = Abandon(pending.clone());
        let preparing = pending.clone();
        let preparer = self.preparer.clone();
        let request = self
            .cpu
            .run_preparing(bytes, deadline, &pending, move |context| {
                preparing.check()?;
                let (request, constraint) = work(context)?;
                if request.id != id {
                    return Err(Error::invalid("CPU preparation changed request identity"));
                }
                preparing.check()?;
                preparer.prepare(request)?.with_constraint(constraint)
            })
            .await?;
        pending.check()?;
        self.enqueue_prepared(request, pending, trace, deadline)
            .await
    }

    /// Prepare a staged input without keeping a CPU worker while its full quote is pending.
    pub(crate) async fn submit_staged_preparing<T: Send + 'static>(
        &self,
        id: RequestId,
        bytes: usize,
        trace: Option<TraceContext>,
        deadline_us: Option<u64>,
        load: impl FnOnce(&CpuContext) -> Result<StagingPlan<T>> + Send + 'static,
        finish: impl FnOnce(
            T,
            &CpuContext,
        ) -> Result<(CanonicalRequest, Option<infer_spi::ConstraintHandle>)>
        + Send
        + 'static,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        let request_deadline = self.preparation_deadline(deadline_us);
        let deadline = request_deadline.map_or_else(
            || Instant::now() + Duration::from_millis(self.config.job_timeout_ms),
            |deadline| {
                deadline.min(Instant::now() + Duration::from_millis(self.config.job_timeout_ms))
            },
        );
        let pending = self.ingress.reserve(id, bytes)?;
        self.request_ids
            .fetch_max(id.get(), AtomicOrdering::Relaxed);
        let _abandon = Abandon(pending.clone());
        let preparing = pending.clone();
        let preparer = self.preparer.clone();
        let initial = self
            .cpu
            .run_preparing(bytes, Some(deadline), &pending, move |context| {
                preparing.check()?;
                let result = match load(context)? {
                    StagingPlan::Ready(request, constraint) => {
                        if request.id != id {
                            return Err(Error::invalid("CPU preparation changed request identity"));
                        }
                        InitialStage::Ready(preparer.prepare(request)?.with_constraint(constraint)?)
                    }
                    StagingPlan::Deferred(stage, quote) => {
                        InitialStage::Deferred(stage, quote, preparer)
                    }
                };
                preparing.check()?;
                Ok(result)
            })
            .await?;
        pending.check()?;
        let (stage, quote, preparer) = match initial {
            InitialStage::Ready(request) => {
                return self
                    .enqueue_prepared(request, pending, trace, request_deadline)
                    .await;
            }
            InitialStage::Deferred(stage, quote, preparer) => (stage, quote, preparer),
        };
        let staged = RetainedStage {
            input: stage,
            pending: pending.clone(),
        };
        pending.retain_bytes(quote.retained())?;
        pending.wait_for_bytes(quote.peak(), deadline).await?;
        let request = self
            .cpu
            .run_preparing_wait(quote.peak(), deadline, &pending, move |context| {
                staged.with_owner(|stage, preparing| {
                    preparing.check()?;
                    let (request, constraint) = finish(stage, context)?;
                    if request.id != id {
                        return Err(Error::invalid("CPU preparation changed request identity"));
                    }
                    preparing.check()?;
                    preparer.prepare(request)?.with_constraint(constraint)
                })
            })
            .await?;
        pending.check()?;
        self.enqueue_prepared(request, pending, trace, request_deadline)
            .await
    }

    fn preparation_deadline(&self, deadline_us: Option<u64>) -> Option<Instant> {
        let now_us = self.clock_origin_us.saturating_add(
            u64::try_from(self.clock_started.elapsed().as_micros()).unwrap_or(u64::MAX),
        );
        deadline_us
            .map(|deadline| Instant::now() + Duration::from_micros(deadline.saturating_sub(now_us)))
    }

    async fn enqueue_prepared(
        &self,
        request: infer_runtime::PreparedRequest,
        pending: Arc<Pending>,
        trace: Option<TraceContext>,
        deadline: Option<Instant>,
    ) -> Result<async_mpsc::Receiver<Result<EngineOutput>>> {
        let (output, receiver) = async_mpsc::channel(OUTPUT_CHANNEL_CAPACITY);
        let (reply, response) = oneshot::channel();
        self.send(Command::Submit {
            request: Box::new(request),
            pending,
            trace,
            output,
            reply,
        })?;
        // Admission may wait behind resource capture on the device owner. It uses
        // that existing resource budget; inspection/cancellation keep the control budget.
        let budget = Duration::from_millis(
            self.config
                .command_timeout_ms
                .max(self.config.device_control_timeout_ms),
        );
        let budget = deadline.map_or(budget, |deadline| {
            budget.min(deadline.saturating_duration_since(Instant::now()))
        });
        Self::reply_with_budget(response, budget).await??;
        Ok(receiver)
    }
    /// # Errors
    /// Returns a not-found or conflict error if the request is unknown or the runtime channel is closed.
    pub async fn cancel(&self, request: RequestId) -> Result<()> {
        let (reply, response) = oneshot::channel();
        let preparing = self.ingress.cancel(request);
        self.send(Command::Cancel {
            request,
            preparing,
            reply,
        })?;
        self.reply(response).await?
    }
    /// # Errors
    /// Returns a conflict error if the runtime channel is closed or inspection cannot be delivered.
    pub async fn inspect(&self) -> Result<RuntimeInspection> {
        let (reply, response) = oneshot::channel();
        self.send(Command::Inspect { reply })?;
        self.reply(response).await
    }
    /// # Errors
    /// Returns a capacity, query-validation, or stopped-actor error.
    pub async fn observe(&self, query: ObservationQuery) -> Result<serde_json::Value> {
        let bytes = match query {
            ObservationQuery::Metrics => METRICS_RESPONSE_BUDGET_BYTES,
            ObservationQuery::Diagnostics => DIAGNOSTICS_RESPONSE_BUDGET_BYTES,
            ObservationQuery::Events { limit, .. } => limit
                .checked_mul(EVENT_RESPONSE_BYTES)
                .and_then(|bytes| bytes.checked_add(EVENTS_RESPONSE_BASE_BYTES))
                .ok_or_else(|| Error::invalid("observation response budget overflow"))?,
            _ => self.observation_bytes,
        };
        let credit = self.observation.reserve(bytes)?;
        let (reply, response) = oneshot::channel();
        self.send(Command::Observe {
            query,
            credit,
            reply,
        })?;
        let (snapshot, credit) = self.reply(response).await??;
        self.observation
            .run_reserved(credit, None, move |_| snapshot.render())
            .await
    }
    /// # Errors
    /// Returns a conflict error if the runtime channel is closed or shutdown cannot be acknowledged.
    pub async fn shutdown(&self) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(SHUTDOWN_ACK_TIMEOUT_SECS);
        self.ingress.stop()?;
        let (reply, response) = oneshot::channel();
        self.send(Command::Stop { reply })?;
        timeout_at(TokioInstant::from_std(deadline),response)
            .await.map_err(|_|Error::new(ErrorCode::Backend,"shutdown acknowledgement timed out; in-flight resources remain owned by the actor"))?
            .map_err(|_| Error::new(ErrorCode::Backend, "engine actor stopped"))??;
        self.ingress.wait_drained(deadline).await
    }
}
