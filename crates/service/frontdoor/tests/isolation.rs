#[path = "../../../engine/runtime/tests/support/mod.rs"]
mod support;

use infer_core::{Error, FinishReason, ModelId, RequestId, Result, StateId};
use infer_frontdoor::{RuntimeHandle, cpu::CpuConfig};
use infer_ir::{
    CanonicalRequest, DeviceCapabilities, ExecutionProgram, ExecutionTask, ModelIr, OutputReadout,
    PrecisionPlan, StepPlan, TaskOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, EngineOutput, RuntimeConfig};
use infer_spi::BackendProvider;
use std::{
    sync::Arc, sync::atomic::AtomicUsize, sync::atomic::Ordering, sync::mpsc, time::Duration,
};
use support::ProtocolBackend;
use tokio::sync::mpsc::Receiver;
use tokio::time::sleep;
use tokio::time::timeout;

#[derive(Clone, Copy, PartialEq, Eq)]
enum GateAt {
    Quote,
    Reserve,
    Submit,
}
struct Gated {
    stage: GateAt,
    inner: ProtocolBackend,
    gate: Option<mpsc::Receiver<()>>,
    entered: mpsc::Sender<()>,
    owned: Arc<AtomicUsize>,
}
impl Gated {
    fn wait(&mut self, stage: GateAt) -> Result<()> {
        if self.stage == stage
            && let Some(gate) = self.gate.take()
        {
            self.entered
                .send(())
                .map_err(|e| Error::invariant(e.to_string()))?;
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
        }
        Ok(())
    }
}
impl BackendProvider for Gated {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn formats(&self) -> infer_ir::BackendFormats {
        self.inner.formats()
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn prepare_state_reservation(&mut self, capacity: usize, readout: OutputReadout) -> Result<()> {
        self.wait(GateAt::Quote)?;
        self.inner.prepare_state_reservation(capacity, readout)
    }
    fn reserve_state(&mut self, state: StateId, capacity: usize) -> Result<()> {
        self.inner.reserve_state(state, capacity)?;
        self.owned.fetch_add(1, Ordering::AcqRel);
        self.wait(GateAt::Reserve)
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.inner.release_state(state)?;
        self.owned.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        // The engine creates sequences through this entry point, so the gate and the ownership
        // count have to be here as well as on `reserve_state`.
        self.inner.reserve_state_for(state, capacity, readout)?;
        self.owned.fetch_add(1, Ordering::AcqRel);
        self.wait(GateAt::Reserve)
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.wait(GateAt::Submit)?;
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        self.inner.poll(ticket)
    }
}
fn request(id: u64) -> Result<CanonicalRequest> {
    serde_json::from_value(serde_json::json!({"id":id,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":2}}})).map_err(|e| Error::invalid(e.to_string()))
}
async fn terminal(receiver: &mut Receiver<Result<EngineOutput>>) -> Result<FinishReason> {
    loop {
        let event = timeout(Duration::from_secs(3), receiver.recv())
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
            .ok_or_else(|| Error::invariant("stream closed"))??;
        if let EngineOutput::Finished(done) = event {
            return Ok(done.reason);
        }
    }
}
#[tokio::test]
async fn blocked_cpu_encoding_preserves_control_progress_and_fence_owned_state() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let (release, gate) = mpsc::channel();
    let (entered, started) = mpsc::channel();
    let owned = Arc::new(AtomicUsize::new(0));
    let backend = Gated {
        stage: GateAt::Submit,
        inner: ProtocolBackend::new(16, 8, &ir)?,
        gate: Some(gate),
        entered,
        owned: owned.clone(),
    };
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let handle = RuntimeHandle::start_with_config(
        Engine::new(
            backend,
            ir,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig::default(),
        )?,
        CpuConfig {
            workers: 1,
            max_jobs: 4,
            ..Default::default()
        },
    )?;
    let mut first = handle.submit(request(1)?).await?;
    for _ in 0..1000 {
        if started.try_recv().is_ok() {
            break;
        }
        sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(owned.load(Ordering::Acquire), 1);
    handle.cancel(RequestId::ONE).await?;
    assert_eq!(terminal(&mut first).await?, FinishReason::Cancelled);
    assert!(handle.inspect().await?.resource_release_pending);
    assert_eq!(owned.load(Ordering::Acquire), 1);
    assert_eq!(handle.delivery_pool().run(32, |_| Ok(7)).await?, 7);
    let next_handle = handle.clone();
    let next = tokio::spawn(async move { next_handle.submit(request(2)?).await });
    assert!(handle.inspect().await?.inflight_step.is_some());
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    let mut second = next.await.map_err(|e| Error::invariant(e.to_string()))??;
    assert_eq!(terminal(&mut second).await?, FinishReason::Length);
    handle.shutdown().await?;
    assert_eq!(owned.load(Ordering::Acquire), 0);
    Ok(())
}
#[tokio::test]
async fn cancellation_during_preparation_prevents_late_admission() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let handle = RuntimeHandle::start(Engine::new(
        ProtocolBackend::new(16, 8, &ir)?,
        ir,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?)?;
    let (release, gate) = mpsc::channel();
    let (entered, started) = mpsc::channel();
    let preparing = handle.clone();
    let job = tokio::spawn(async move {
        preparing
            .submit_preparing(RequestId::ONE, 32, None, move |_| {
                entered
                    .send(())
                    .map_err(|e| Error::invariant(e.to_string()))?;
                gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
                request(1)
            })
            .await
    });
    let mut observed = false;
    for _ in 0..1000 {
        if started.try_recv().is_ok() {
            observed = true;
            break;
        }
        sleep(Duration::from_millis(1)).await;
    }
    assert!(observed);
    handle.cancel(RequestId::ONE).await?;
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    assert!(
        job.await
            .map_err(|e| Error::invariant(e.to_string()))?
            .is_err()
    );
    assert_eq!(handle.inspect().await?.active_requests, 0);
    handle.shutdown().await
}

#[tokio::test]
async fn blocked_resource_owner_keeps_controls_live_and_delivers_cancel_or_timeout() -> Result<()> {
    for cancel in [true, false] {
        let ir = support::model(ModelId::ONE);
        let (release, gate) = mpsc::channel();
        let (entered, start) = mpsc::channel();
        let owned = Arc::new(AtomicUsize::new(0));
        let backend = Gated {
            stage: GateAt::Reserve,
            inner: ProtocolBackend::new(16, 8, &ir)?,
            gate: Some(gate),
            entered,
            owned: owned.clone(),
        };
        let mut registry = KernelRegistry::default();
        registry.register(&support::DeclaredKernels)?;
        let handle = RuntimeHandle::start_with_config(
            Engine::new(
                backend,
                ir,
                PrecisionPlan::f32(),
                &registry,
                RuntimeConfig::default(),
            )?,
            CpuConfig {
                // Cancellation must precede its independent resource deadline. Keep the
                // timeout branch short; the closed gate proves control liveness in both.
                device_control_timeout_ms: if cancel { 30_000 } else { 50 },
                ..Default::default()
            },
        )?;
        let mut stream = handle.submit(request(1)?).await?;
        let mut entered = false;
        for _ in 0..1000 {
            if start.try_recv().is_ok() {
                entered = true;
                break;
            }
            sleep(Duration::from_millis(1)).await;
        }
        assert!(entered);
        // A response while the gate remains closed proves independence from the owner;
        // a 25-ms host scheduling budget does not, and flakes on shared CI machines.
        let inspection = timeout(Duration::from_secs(3), handle.inspect())
            .await
            .map_err(|e| Error::invariant(e.to_string()))??;
        assert!(inspection.resource_release_pending);
        assert_eq!(owned.load(Ordering::Acquire), 1);
        if cancel {
            timeout(Duration::from_secs(3), handle.cancel(RequestId::ONE))
                .await
                .map_err(|e| {
                    Error::invariant(format!("cancel blocked by resource owner: {e}"))
                })??;
        }
        let reason = terminal(&mut stream).await?;
        if cancel {
            assert_eq!(reason, FinishReason::Cancelled);
        } else {
            assert!(matches!(reason, FinishReason::Failed(_)));
        }
        assert_eq!(handle.delivery_pool().run(32, |_| Ok(7)).await?, 7);
        assert_eq!(owned.load(Ordering::Acquire), 1);
        release
            .send(())
            .map_err(|e| Error::invariant(e.to_string()))?;
        let mut next = handle.submit(request(2)?).await?;
        assert_eq!(terminal(&mut next).await?, FinishReason::Length);
        handle.shutdown().await?;
        assert_eq!(owned.load(Ordering::Acquire), 0);
    }
    Ok(())
}

#[tokio::test]
async fn admission_uses_resource_budget_while_controls_remain_responsive() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let (release, gate) = mpsc::channel();
    let (entered, started) = mpsc::channel();
    let owned = Arc::new(AtomicUsize::new(0));
    let backend = Gated {
        stage: GateAt::Quote,
        inner: ProtocolBackend::new(4, 4, &ir)?,
        gate: Some(gate),
        entered,
        owned: owned.clone(),
    };
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let handle = RuntimeHandle::start_with_config(
        Engine::new(
            backend,
            ir,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig::default(),
        )?,
        CpuConfig {
            workers: 1,
            max_jobs: 4,
            command_timeout_ms: 3_000,
            device_control_timeout_ms: 30_000,
            ..Default::default()
        },
    )?;
    let submitting = handle.clone();
    let submission = tokio::spawn(async move { submitting.submit(request(1)?).await });
    for _ in 0..5000 {
        if started.try_recv().is_ok() {
            break;
        }
        sleep(Duration::from_millis(1)).await;
    }
    // The backend quote is still gated. Inspection uses the independent control path.
    handle.inspect().await?;
    assert_eq!(owned.load(Ordering::Acquire), 0);
    sleep(Duration::from_secs(5)).await;
    let still_pending = !submission.is_finished();
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    assert!(
        still_pending,
        "admission expired at the shorter control deadline"
    );
    let mut output = submission
        .await
        .map_err(|e| Error::invariant(e.to_string()))??;
    assert!(!matches!(
        terminal(&mut output).await?,
        FinishReason::Failed(_)
    ));
    handle.shutdown().await?;
    assert_eq!(owned.load(Ordering::Acquire), 0);
    Ok(())
}
