#[path = "../../../../engine/runtime/tests/support/mod.rs"]
mod support;
use crate::cpu::CpuConfig;
use infer_core::{ErrorCode, ModelId};
use infer_ir::PrecisionPlan;
use infer_kernel_api::KernelRegistry;
use infer_runtime::RuntimeConfig;
use std::collections::BTreeSet;
use std::thread::spawn;
use std::time::Duration;
use tokio::time::sleep;
use tokio::time::timeout;

use super::*;

#[tokio::test]
async fn automatic_request_ids_are_shared_concurrent_and_never_wrap() -> Result<()> {
    let handle = handle()?;
    let mut workers = Vec::new();
    for _ in 0..8 {
        let handle = handle.clone();
        workers.push(spawn(move || {
            (0..32)
                .map(|_| handle.allocate_request_id())
                .collect::<Result<Vec<_>>>()
        }));
    }
    let mut ids = BTreeSet::new();
    for worker in workers {
        for id in worker.join().unwrap()? {
            assert!(ids.insert(id));
        }
    }
    assert_eq!(ids.len(), 256);
    assert_eq!(handle.allocate_request_id()?.get(), 257);
    handle.request_ids.store(u64::MAX - 1, Ordering::Relaxed);
    assert_eq!(handle.allocate_request_id()?.get(), u64::MAX);
    assert_eq!(
        handle.allocate_request_id().unwrap_err().code,
        ErrorCode::Capacity
    );
    assert_eq!(
        handle.allocate_request_id().unwrap_err().code,
        ErrorCode::Capacity
    );
    handle.shutdown().await
}
fn handle() -> Result<RuntimeHandle> {
    let ir = support::model(ModelId::ONE);
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    RuntimeHandle::start_with_config(
        Engine::new(
            support::ProtocolBackend::new(16, 8, &ir)?,
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
    )
}
#[tokio::test]
async fn blocked_exporter_cannot_block_requests_controls_or_delivery() -> Result<()> {
    let handle = handle()?;
    let pool = handle.observation.clone();
    let (release, gate) = mpsc::channel();
    let (entered, started) = oneshot::channel();
    let blocker = tokio::spawn(async move {
        pool.run(32, move |_| {
            let _ = entered.send(());
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
            Ok(())
        })
        .await
    });
    started.await.map_err(|e| Error::invariant(e.to_string()))?;
    let observer = handle.clone();
    let query = tokio::spawn(async move { observer.observe(ObservationQuery::Timeline).await });
    for _ in 0..100 {
        if handle.observation.inspect().retained_jobs == 2 {
            break;
        }
        sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(handle.observation.inspect().retained_jobs, 2);
    assert!(!query.is_finished());
    assert_eq!(
        handle
            .observe(ObservationQuery::Metrics)
            .await
            .err()
            .map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    assert!(handle.inspect().await?.ready);
    assert_eq!(handle.delivery.run(32, |_| Ok(7)).await?, 7);
    let request = serde_json::from_value(serde_json::json!({"id":1,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":2}}})).map_err(|e| Error::invalid(e.to_string()))?;
    let mut stream = handle.submit(request).await?;
    loop {
        let output = timeout(Duration::from_secs(2), stream.recv())
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
            .ok_or_else(|| Error::invariant("output closed before terminal"))??;
        if matches!(output, EngineOutput::Finished(_)) {
            break;
        }
    }
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    blocker
        .await
        .map_err(|e| Error::invariant(e.to_string()))??;
    let snapshot = query.await.map_err(|e| Error::invariant(e.to_string()))??;
    assert_eq!(snapshot["clock"], "runtime_logical");
    assert_eq!(handle.observation.inspect().retained_jobs, 0);
    handle.shutdown().await
}
#[tokio::test]
async fn abandoned_query_releases_snapshot_credit_and_invalid_queries_do_not_leak() -> Result<()> {
    let handle = handle()?;
    assert!(
        handle
            .observe(ObservationQuery::Events {
                after: 0,
                limit: 0,
                request: None
            })
            .await
            .is_err()
    );
    assert_eq!(handle.observation.inspect().retained_jobs, 0);
    let credit = handle.observation.reserve(handle.observation_bytes)?;
    let (reply, response) = oneshot::channel();
    handle.send(Command::Observe {
        query: ObservationQuery::Otlp,
        credit,
        reply,
    })?;
    drop(response);
    // This ordered acknowledgement ensures the actor processed the abandoned snapshot command.
    handle.inspect().await?;
    assert_eq!(handle.observation.inspect().retained_bytes, 0);
    assert!(
        handle
            .observe(ObservationQuery::Summary)
            .await?
            .get("trace_coverage")
            .is_some()
    );
    handle.shutdown().await
}

#[tokio::test]
async fn inspection_in_the_same_command_batch_observes_retired_cancellation() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        support::ProtocolBackend::new(16, 8, &ir)?,
        ir,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?;
    let request = serde_json::from_value(serde_json::json!({"id":1,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":2}}})).map_err(|e| Error::invalid(e.to_string()))?;
    engine.submit(request)?;
    let (_sender, receiver) = mpsc::sync_channel(1);
    let (_control, control_rx) = mpsc::sync_channel(1);
    let mut actor = RuntimeActor::new(
        engine,
        receiver,
        control_rx,
        Arc::new(AtomicBool::new(false)),
        Instant::now(),
    );
    let mut generated = Vec::new();
    let (reply, cancel) = oneshot::channel();
    actor.handle_command(
        Command::Cancel {
            request: RequestId::ONE,
            preparing: false,
            reply,
        },
        &mut generated,
    );
    cancel
        .await
        .map_err(|e| Error::invariant(e.to_string()))??;
    assert_eq!(actor.engine.inspect().completed_requests, 1);
    let (reply, response) = oneshot::channel();
    actor.handle_command(Command::Inspect { reply }, &mut generated);
    let inspection = response
        .await
        .map_err(|e| Error::invariant(e.to_string()))?;
    assert_eq!(inspection.completed_requests, 0);
    assert_eq!(inspection.state.allocated_pages, 0);
    assert!(!inspection.resource_release_pending);
    assert_eq!(generated, [] as [EngineOutput; 0]);
    Ok(())
}

#[tokio::test]
async fn cancelled_image_quote_retains_ingress_until_owner_acknowledges() -> Result<()> {
    use infer_spi::{ResourceReply, ResourceTicket};

    #[derive(Clone, Copy)]
    enum Cause {
        Cancel,
        Stop,
        ReplyClosed,
        OutputClosed,
    }

    for (cause, issued) in [
        Cause::Cancel,
        Cause::Stop,
        Cause::ReplyClosed,
        Cause::OutputClosed,
    ]
    .into_iter()
    .flat_map(|cause| [(cause, false), (cause, true)])
    {
        let ir = support::model(ModelId::ONE);
        let mut kernels = KernelRegistry::default();
        kernels.register(&support::DeclaredKernels)?;
        let backend = support::ProtocolBackend::new(16, 8, &ir)?.multimodal();
        let engine = Engine::new(
            backend,
            ir,
            PrecisionPlan::f32(),
            &kernels,
            RuntimeConfig::default(),
        )?;
        let (_sender, receiver) = mpsc::sync_channel(1);
        let (_control, control_rx) = mpsc::sync_channel(1);
        let mut actor = RuntimeActor::new(
            engine,
            receiver,
            control_rx,
            Arc::new(AtomicBool::new(false)),
            Instant::now(),
        );
        let (request, owner_pixels) = image_quote_request()?;
        let weak = Arc::downgrade(&owner_pixels);
        let prepared = actor.engine.request_preparer()?.prepare(request)?;
        let ingress = Registry::new(1, 128);
        let pending = ingress.reserve(RequestId::ONE, 128)?;
        let (ticket, owner_ack) = if issued {
            let (ticket, acknowledgement) = ResourceTicket::channel();
            (Some(ticket), Some(acknowledgement))
        } else {
            (None, None)
        };
        let (output, reader) = async_mpsc::channel(1);
        let (reply, response) = oneshot::channel();
        let mut reader = Some(reader);
        let mut response = Some(response);
        actor.quoting.push_back((
            Command::Submit {
                request: Box::new(prepared),
                pending,
                trace: None,
                output,
                reply,
            },
            ticket,
        ));
        match cause {
            Cause::Cancel => assert!(ingress.cancel(RequestId::ONE)),
            Cause::Stop => actor.stopping = true,
            Cause::ReplyClosed => drop(response.take()),
            Cause::OutputClosed => drop(reader.take()),
        }
        actor.poll_admissions(&mut Vec::new());
        if let Some(response) = response {
            assert!(
                response
                    .await
                    .map_err(|error| Error::invariant(error.to_string()))?
                    .is_err()
            );
        }
        if let Some(owner_ack) = owner_ack {
            assert!(ingress.reserve(RequestId::ONE, 128).is_err());
            assert_eq!(actor.quoting.len(), 1);
            drop(owner_pixels);
            owner_ack
                .send(Ok(ResourceReply::ReservationBytes(Some(1))))
                .map_err(|_| Error::invariant("quote acknowledgement dropped"))?;
            actor.poll_admissions(&mut Vec::new());
        } else {
            drop(owner_pixels);
        }
        assert!(actor.quoting.is_empty());
        assert!(weak.upgrade().is_none());
        drop(ingress.reserve(RequestId::ONE, 128)?);
    }
    Ok(())
}

fn image_quote_request() -> Result<(CanonicalRequest, Arc<[f32]>)> {
    use infer_ir::{MediaInput, MediaItem, Modality, RequestInput};

    let mut request: CanonicalRequest = serde_json::from_value(serde_json::json!({
        "id": 1, "model": 1, "input": {"Sequence": {"tokens": [1, 2, 3]}},
        "workload": {"Generate": {"max_new_tokens": 2}}
    }))
    .map_err(|error| Error::invalid(error.to_string()))?;
    let pixels: Arc<[f32]> = Arc::from([0.0; 4]);
    let owner_pixels = pixels.clone();
    if let RequestInput::Sequence { media, .. } = &mut request.input {
        media.push(MediaInput {
            id: infer_core::MediaId::ONE,
            modality: Modality::Image,
            content_digest: "fixture".into(),
            item: Some(MediaItem {
                modality: Modality::Image,
                pixels,
                grid: (1, 1, 1),
                span: 0..1,
            }),
        });
    }
    Ok((request, owner_pixels))
}

struct QuoteGate {
    inner: support::ProtocolBackend,
    entered: mpsc::Sender<()>,
    gate: Option<mpsc::Receiver<()>>,
}

impl BackendProvider for QuoteGate {
    type Ticket = <support::ProtocolBackend as BackendProvider>::Ticket;

    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn supports_multimodal(&self) -> bool {
        true
    }
    fn capabilities(&self) -> infer_ir::DeviceCapabilities {
        self.inner.capabilities()
    }
    fn validate_program(
        &self,
        model: &infer_ir::ModelIr,
        program: &infer_ir::ExecutionProgram,
    ) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &infer_ir::ModelIr) -> Result<infer_ir::DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn prepare_state_reservation(
        &mut self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
    ) -> Result<()> {
        if let Some(gate) = self.gate.take() {
            self.entered
                .send(())
                .map_err(|error| Error::invariant(error.to_string()))?;
            gate.recv()
                .map_err(|error| Error::invariant(error.to_string()))?;
        }
        self.inner.prepare_state_reservation(capacity, readout)
    }
    fn state_reservation_bytes_for_media(
        &self,
        capacity: usize,
        readout: infer_ir::OutputReadout,
        media: &infer_ir::MediaPlan,
    ) -> Result<Option<u64>> {
        self.inner
            .state_reservation_bytes_for_media(capacity, readout, media)
    }
    fn submit(
        &mut self,
        program: &infer_ir::ExecutionProgram,
        step: &infer_ir::StepPlan,
        tasks: Vec<infer_ir::ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<infer_ir::TaskOutput>>> {
        self.inner.poll(ticket)
    }
}

#[tokio::test]
async fn cancelled_image_quote_waits_for_threaded_owner() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let (release, gate) = mpsc::channel();
    let (entered, started) = mpsc::channel();
    let backend = QuoteGate {
        inner: support::ProtocolBackend::new(16, 8, &ir)?
            .multimodal()
            .priced_quotes(1024, 4096),
        entered,
        gate: Some(gate),
    };
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        backend,
        ir,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )?
    .into_threaded(4, Duration::from_secs(5))?;
    let first: CanonicalRequest = serde_json::from_value(serde_json::json!({
        "id": 2, "model": 1, "input": {"Sequence": {"tokens": [1, 2, 3]}},
        "workload": {"Generate": {"max_new_tokens": 2}}
    }))
    .map_err(|error| Error::invalid(error.to_string()))?;
    let first = engine.request_preparer()?.prepare(first)?;
    let _first_ticket = engine.admission_quote(&first)?;
    started
        .recv_timeout(Duration::from_secs(3))
        .map_err(|error| Error::invariant(error.to_string()))?;
    let (_sender, receiver) = mpsc::sync_channel(1);
    let (_control, control_rx) = mpsc::sync_channel(1);
    let mut actor = RuntimeActor::new(
        engine,
        receiver,
        control_rx,
        Arc::new(AtomicBool::new(false)),
        Instant::now(),
    );
    let (request, pixels) = image_quote_request()?;
    let weak = Arc::downgrade(&pixels);
    drop(pixels);
    let prepared = actor.engine.request_preparer()?.prepare(request)?;
    let ingress = Registry::new(1, 128);
    let pending = ingress.reserve(RequestId::ONE, 128)?;
    let (output, _reader) = async_mpsc::channel(1);
    let (reply, response) = oneshot::channel();
    actor.handle_command(
        Command::Submit {
            request: Box::new(prepared),
            pending,
            trace: None,
            output,
            reply,
        },
        &mut Vec::new(),
    );
    assert_eq!(actor.quoting.len(), 1);
    assert!(ingress.cancel(RequestId::ONE));
    actor.poll_admissions(&mut Vec::new());
    assert!(
        response
            .await
            .map_err(|error| Error::invariant(error.to_string()))?
            .is_err()
    );
    assert!(ingress.reserve(RequestId::ONE, 128).is_err());
    assert!(weak.upgrade().is_some());
    release
        .send(())
        .map_err(|error| Error::invariant(error.to_string()))?;
    for _ in 0..3000 {
        actor.poll_admissions(&mut Vec::new());
        if actor.quoting.is_empty() {
            break;
        }
        sleep(Duration::from_millis(1)).await;
    }
    assert!(
        actor.quoting.is_empty(),
        "owner did not finish its queued quote"
    );
    assert!(weak.upgrade().is_none());
    drop(ingress.reserve(RequestId::ONE, 128)?);
    Ok(())
}
