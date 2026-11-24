use crate::cpu::CpuConfig;
use infer_core::{ErrorCode, ModelId};
use infer_ir::PrecisionPlan;
use infer_kernel_api::KernelRegistry;
use infer_runtime::RuntimeConfig;
use infer_test_support::fixtures as support;
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
