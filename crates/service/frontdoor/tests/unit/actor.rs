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
    handle_with_cpu(CpuConfig {
        workers: 1,
        max_jobs: 4,
        ..Default::default()
    })
}

fn handle_with_cpu(config: CpuConfig) -> Result<RuntimeHandle> {
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
        config,
    )
}

#[tokio::test]
async fn media_budget_growth_is_shared_and_retained_until_ingress_release() -> Result<()> {
    let registry = Registry::new(2, 100);
    let pending = registry.reserve(RequestId::ONE, 20)?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 100,
        ..CpuConfig::default()
    })?;
    let observed_pool = pool.clone();
    let observed_registry = registry.clone();
    let late_context = pool
        .run_preparing(20, None, &pending, move |context| {
            context.ensure_media_bytes(70, 20)?;
            context.ensure_media_bytes(70, 20)?;
            assert_eq!(observed_pool.inspect().retained_bytes, 70);
            assert!(observed_registry.reserve(RequestId::new(2)?, 31).is_err());
            Ok(context.clone())
        })
        .await?;
    assert_eq!(pool.inspect().retained_bytes, 0);
    assert!(late_context.ensure_media_bytes(80, 20).is_err());
    assert!(registry.reserve(RequestId::new(2)?, 31).is_err());
    drop(pending);
    drop(registry.reserve(RequestId::new(2)?, 100)?);
    Ok(())
}

#[tokio::test]
async fn failed_media_budget_growth_rolls_back_cpu_credit() -> Result<()> {
    let registry = Registry::new(2, 64);
    let pending = registry.reserve(RequestId::ONE, 20)?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 100,
        ..CpuConfig::default()
    })?;
    let observed_pool = pool.clone();
    let observed_registry = registry.clone();
    pool.run_preparing(20, None, &pending, move |context| {
        assert_eq!(
            context.ensure_media_bytes(70, 20).unwrap_err().code,
            ErrorCode::Capacity
        );
        assert_eq!(observed_pool.inspect().retained_bytes, 20);
        drop(observed_registry.reserve(RequestId::new(2)?, 44)?);
        Ok(())
    })
    .await?;
    assert_eq!(pool.inspect().retained_bytes, 0);
    drop(pending);
    drop(registry.reserve(RequestId::new(2)?, 64)?);
    Ok(())
}

#[tokio::test]
async fn exhausted_cpu_media_budget_does_not_grow_ingress_credit() -> Result<()> {
    let registry = Registry::new(2, 100);
    let pending = registry.reserve(RequestId::ONE, 20)?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 64,
        ..CpuConfig::default()
    })?;
    let observed_pool = pool.clone();
    let observed_registry = registry.clone();
    pool.run_preparing(20, None, &pending, move |context| {
        assert_eq!(
            context.ensure_media_bytes(70, 20).unwrap_err().code,
            ErrorCode::Capacity
        );
        assert_eq!(observed_pool.inspect().retained_bytes, 20);
        drop(observed_registry.reserve(RequestId::new(2)?, 80)?);
        Ok(())
    })
    .await?;
    assert_eq!(pool.inspect().retained_bytes, 0);
    drop(pending);
    drop(registry.reserve(RequestId::new(2)?, 100)?);
    Ok(())
}

#[tokio::test]
async fn cancelled_ingress_cannot_grow_its_media_budget() -> Result<()> {
    let registry = Registry::new(2, 100);
    let pending = registry.reserve(RequestId::ONE, 20)?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 100,
        ..CpuConfig::default()
    })?;
    let observed_pool = pool.clone();
    let observed_registry = registry.clone();
    pool.run_preparing(20, None, &pending, move |context| {
        assert!(observed_registry.cancel(RequestId::ONE));
        assert_eq!(
            context.ensure_media_bytes(70, 20).unwrap_err().code,
            ErrorCode::Capacity
        );
        assert_eq!(observed_pool.inspect().retained_bytes, 20);
        drop(observed_registry.reserve(RequestId::new(2)?, 80)?);
        Ok(())
    })
    .await?;
    assert_eq!(pool.inspect().retained_bytes, 0);
    drop(pending);
    drop(registry.reserve(RequestId::new(2)?, 100)?);
    Ok(())
}

#[tokio::test]
async fn cancelled_media_growth_keeps_credit_until_the_worker_finishes() -> Result<()> {
    let registry = Registry::new(2, 70);
    let pending = registry.reserve(RequestId::ONE, 20)?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 70,
        ..CpuConfig::default()
    })?;
    let (started, entered) = oneshot::channel();
    let (release, gate) = mpsc::channel();
    let dropped_under_credit = Arc::new(AtomicBool::new(false));
    let payload = CreditDropProbe {
        registry: registry.clone(),
        other: RequestId::new(2)?,
        observed: dropped_under_credit.clone(),
    };
    let worker_pool = pool.clone();
    let worker_pending = pending.clone();
    let task = tokio::spawn(async move {
        worker_pool
            .run_preparing(20, None, &worker_pending, move |context| {
                context.ensure_media_bytes(70, 20)?;
                started
                    .send(())
                    .map_err(|()| Error::invariant("start receiver dropped"))?;
                gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
                Ok(payload)
            })
            .await
    });
    timeout(Duration::from_secs(2), entered)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))?;
    task.abort();
    drop(pending);
    assert_eq!(pool.inspect().retained_bytes, 70);
    let next = RequestId::new(2)?;
    assert!(registry.reserve(next, 1).is_err());
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    timeout(Duration::from_secs(2), async {
        while pool.inspect().retained_jobs != 0 || registry.reserve(next, 70).is_err() {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|e| Error::invariant(e.to_string()))?;
    assert!(dropped_under_credit.load(Ordering::Acquire));
    drop(registry.reserve(next, 70)?);
    Ok(())
}

#[tokio::test]
async fn staged_image_waits_for_credit_without_occupying_a_worker() -> Result<()> {
    let registry = Registry::new(2, 64);
    let first = registry.reserve(RequestId::ONE, 25)?;
    let second = registry.reserve(RequestId::new(2)?, 25)?;
    first.retain_bytes(3)?;
    second.retain_bytes(3)?;
    let deadline = Instant::now() + Duration::from_secs(2);
    first.wait_for_bytes(48, deadline).await?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 64,
        ..CpuConfig::default()
    })?;
    let waiting = second.clone();
    let mut task = tokio::spawn(async move { waiting.wait_for_bytes(48, deadline).await });
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    assert_eq!(pool.run(1, |_| Ok(7)).await?, 7);
    drop(first);
    timeout(Duration::from_secs(2), task)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))??;
    drop(second);
    drop(registry.reserve(RequestId::new(3)?, 64)?);
    Ok(())
}

#[tokio::test]
async fn new_image_does_not_take_credit_needed_by_a_waiting_stage() -> Result<()> {
    let registry = Registry::new(3, 100);
    let waiting = registry.reserve(RequestId::ONE, 10)?;
    let blocker = registry.reserve(RequestId::new(2)?, 40)?;
    let worker = waiting.clone();
    let mut completion = tokio::spawn(async move {
        worker
            .wait_for_bytes(80, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(
        timeout(Duration::from_millis(20), &mut completion)
            .await
            .is_err()
    );
    let (incoming, funded) = registry.reserve_staged(RequestId::new(3)?, 3, 40)?;
    assert!(
        !funded,
        "new input must not consume a waiting stage's credit"
    );
    drop(blocker);
    timeout(Duration::from_secs(2), completion)
        .await
        .map_err(|error| Error::invariant(error.to_string()))?
        .map_err(|error| Error::invariant(error.to_string()))??;
    drop(waiting);
    drop(incoming);
    drop(registry.reserve(RequestId::new(4)?, 100)?);
    Ok(())
}

#[tokio::test]
async fn decoded_image_waiting_for_credit_precedes_a_new_decode() -> Result<()> {
    let registry = Registry::new(3, 100);
    let decoded = registry.reserve(RequestId::ONE, 10)?;
    let blocker = registry.reserve(RequestId::new(2)?, 60)?;
    let worker = decoded.clone();
    let mut processed = tokio::spawn(async move {
        worker
            .wait_for_decoded_bytes(10, 80, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(
        timeout(Duration::from_millis(20), &mut processed)
            .await
            .is_err()
    );
    let (incoming, funded) = registry.reserve_staged(RequestId::new(3)?, 2, 20)?;
    assert!(!funded);
    let worker = incoming.clone();
    let mut decoded_next = tokio::spawn(async move {
        worker
            .wait_for_bytes(20, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(
        timeout(Duration::from_millis(20), &mut decoded_next)
            .await
            .is_err()
    );
    drop(blocker);
    timeout(Duration::from_secs(2), processed)
        .await
        .map_err(|error| Error::invariant(error.to_string()))?
        .map_err(|error| Error::invariant(error.to_string()))??;
    timeout(Duration::from_secs(2), decoded_next)
        .await
        .map_err(|error| Error::invariant(error.to_string()))?
        .map_err(|error| Error::invariant(error.to_string()))??;
    drop(decoded);
    drop(incoming);
    drop(registry.reserve(RequestId::new(4)?, 100)?);
    Ok(())
}

#[tokio::test]
async fn cancelled_decoded_stage_releases_priority_and_wakes_new_input() -> Result<()> {
    let registry = Registry::new(3, 100);
    let decoded = registry.reserve(RequestId::ONE, 10)?;
    let blocker = registry.reserve(RequestId::new(2)?, 60)?;
    let worker = decoded.clone();
    let mut processed = tokio::spawn(async move {
        worker
            .wait_for_decoded_bytes(10, 80, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(
        timeout(Duration::from_millis(20), &mut processed)
            .await
            .is_err()
    );
    let (incoming, funded) = registry.reserve_staged(RequestId::new(3)?, 2, 20)?;
    assert!(!funded);
    let worker = incoming.clone();
    let mut next = tokio::spawn(async move {
        worker
            .wait_for_bytes(20, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(timeout(Duration::from_millis(20), &mut next).await.is_err());
    assert!(registry.cancel(RequestId::ONE));
    let error = timeout(Duration::from_secs(2), processed)
        .await
        .map_err(|error| Error::invariant(error.to_string()))?
        .map_err(|error| Error::invariant(error.to_string()))?
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Capacity);
    drop(decoded);
    timeout(Duration::from_secs(2), next)
        .await
        .map_err(|error| Error::invariant(error.to_string()))?
        .map_err(|error| Error::invariant(error.to_string()))??;
    drop(blocker);
    drop(incoming);
    drop(registry.reserve(RequestId::new(4)?, 100)?);
    Ok(())
}

#[test]
fn initial_staging_checks_identities_and_retained_capacity() -> Result<()> {
    let registry = Registry::new(3, 64);
    let occupied = registry.reserve(RequestId::new(2)?, 48)?;
    let (pending, funded) = registry.reserve_staged(RequestId::ONE, 3, 25)?;
    assert!(!funded);
    assert_eq!(
        registry
            .reserve_staged(RequestId::ONE, 3, 25)
            .err()
            .unwrap()
            .code,
        ErrorCode::Conflict
    );
    assert_eq!(
        registry
            .reserve_staged(RequestId::ONE, 3, 65)
            .err()
            .unwrap()
            .code,
        ErrorCode::Conflict
    );
    assert_eq!(
        registry.reserve(RequestId::new(3)?, 14).err().unwrap().code,
        ErrorCode::Capacity
    );
    assert_eq!(
        registry
            .reserve_staged(RequestId::new(3)?, 3, 65)
            .err()
            .unwrap()
            .code,
        ErrorCode::Capacity
    );
    let (small, funded) = registry.reserve_staged(RequestId::new(3)?, 3, 13)?;
    assert!(funded);
    drop(small);
    drop(pending);
    drop(occupied);
    drop(registry.reserve(RequestId::new(4)?, 64)?);
    Ok(())
}

#[tokio::test]
async fn smaller_staged_request_can_release_credit_before_a_large_waiter() -> Result<()> {
    let registry = Registry::new(2, 100);
    let large = registry.reserve(RequestId::ONE, 30)?;
    let small = registry.reserve(RequestId::new(2)?, 30)?;
    let waiting = large.clone();
    let mut task = tokio::spawn(async move {
        waiting
            .wait_for_bytes(80, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    small
        .wait_for_bytes(40, Instant::now() + Duration::from_secs(2))
        .await?;
    drop(small);
    timeout(Duration::from_secs(2), task)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))??;
    drop(large);
    Ok(())
}

#[tokio::test]
async fn staged_inputs_that_cannot_coexist_release_one_for_progress() -> Result<()> {
    let registry = Registry::new(2, 100);
    let first = registry.reserve(RequestId::ONE, 30)?;
    let second = registry.reserve(RequestId::new(2)?, 30)?;
    let waiting = first.clone();
    let mut task = tokio::spawn(async move {
        waiting
            .wait_for_bytes(90, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    assert_eq!(
        second
            .wait_for_bytes(90, Instant::now() + Duration::from_secs(2))
            .await
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    drop(second);
    timeout(Duration::from_secs(2), task)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))??;
    drop(first);
    Ok(())
}

#[tokio::test]
async fn cancelled_and_timed_out_staging_waiters_release_their_queue_position() -> Result<()> {
    let registry = Registry::new(3, 64);
    let first = registry.reserve(RequestId::ONE, 48)?;
    let second = registry.reserve(RequestId::new(2)?, 3)?;
    let third = registry.reserve(RequestId::new(3)?, 3)?;
    let waiting = second.clone();
    let mut task = tokio::spawn(async move {
        waiting
            .wait_for_bytes(48, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    assert!(registry.cancel(RequestId::new(2)?));
    assert_eq!(
        timeout(Duration::from_secs(2), task)
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
            .map_err(|e| Error::invariant(e.to_string()))?
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    let timed_out = third
        .wait_for_bytes(48, Instant::now() + Duration::from_millis(20))
        .await;
    assert_eq!(timed_out.unwrap_err().code, ErrorCode::Capacity);
    drop(first);
    third
        .wait_for_bytes(48, Instant::now() + Duration::from_secs(2))
        .await?;
    drop(second);
    drop(third);
    drop(registry.reserve(RequestId::new(4)?, 64)?);
    Ok(())
}

#[tokio::test]
async fn shutdown_wakes_staging_waiters_and_drains_their_credits() -> Result<()> {
    let registry = Registry::new(2, 64);
    let first = registry.reserve(RequestId::ONE, 48)?;
    let second = registry.reserve(RequestId::new(2)?, 3)?;
    let waiting = second.clone();
    let mut task = tokio::spawn(async move {
        waiting
            .wait_for_bytes(48, Instant::now() + Duration::from_secs(2))
            .await
    });
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    registry.stop()?;
    assert_eq!(
        task.await
            .map_err(|e| Error::invariant(e.to_string()))?
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    assert!(registry.reserve(RequestId::new(3)?, 1).is_err());
    drop(first);
    drop(second);
    registry
        .wait_drained(Instant::now() + Duration::from_secs(2))
        .await?;
    Ok(())
}

#[tokio::test]
async fn handle_shutdown_waits_for_live_ingress_owners() -> Result<()> {
    let handle = handle()?;
    let pending = handle.ingress.reserve(RequestId::ONE, 1)?;
    let closing = handle.clone();
    let mut task = tokio::spawn(async move { closing.shutdown().await });
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    assert!(pending.check().is_err());
    drop(pending);
    timeout(Duration::from_secs(2), task)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))??;
    Ok(())
}

#[tokio::test]
async fn staged_cpu_credit_wait_wakes_after_an_existing_job_finishes() -> Result<()> {
    let registry = Registry::new(1, 64);
    let pending = registry.reserve(RequestId::ONE, 48)?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 64,
        ..CpuConfig::default()
    })?;
    let (started, entered) = oneshot::channel();
    let (release, gate) = mpsc::channel();
    let occupied = pool.clone();
    let first = tokio::spawn(async move {
        occupied
            .run(64, move |_| {
                started
                    .send(())
                    .map_err(|()| Error::invariant("start receiver dropped"))?;
                gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
                Ok(())
            })
            .await
    });
    timeout(Duration::from_secs(2), entered)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))?;
    let waiting_pool = pool.clone();
    let waiting_pending = pending.clone();
    let mut second = tokio::spawn(async move {
        waiting_pool
            .run_preparing_wait(
                48,
                Instant::now() + Duration::from_secs(2),
                &waiting_pending,
                |_| Ok(7),
            )
            .await
    });
    assert!(
        timeout(Duration::from_millis(20), &mut second)
            .await
            .is_err()
    );
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    first.await.map_err(|e| Error::invariant(e.to_string()))??;
    assert_eq!(
        timeout(Duration::from_secs(2), second)
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
            .map_err(|e| Error::invariant(e.to_string()))??,
        7
    );
    drop(pending);
    Ok(())
}

#[tokio::test]
async fn cancelling_a_cpu_credit_wait_does_not_need_the_worker_to_finish() -> Result<()> {
    let registry = Registry::new(1, 64);
    let pending = registry.reserve(RequestId::ONE, 48)?;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        max_bytes: 64,
        ..CpuConfig::default()
    })?;
    let (started, entered) = oneshot::channel();
    let (release, gate) = mpsc::channel();
    let occupied = pool.clone();
    let first = tokio::spawn(async move {
        occupied
            .run(64, move |_| {
                started
                    .send(())
                    .map_err(|()| Error::invariant("start receiver dropped"))?;
                gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
                Ok(())
            })
            .await
    });
    timeout(Duration::from_secs(2), entered)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))?;
    let waiting_pool = pool.clone();
    let waiting_pending = pending.clone();
    let mut second = tokio::spawn(async move {
        waiting_pool
            .run_preparing_wait(
                48,
                Instant::now() + Duration::from_secs(2),
                &waiting_pending,
                |_| Ok(7),
            )
            .await
    });
    assert!(
        timeout(Duration::from_millis(20), &mut second)
            .await
            .is_err()
    );
    assert!(registry.cancel(RequestId::ONE));
    assert_eq!(
        timeout(Duration::from_secs(2), second)
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
            .map_err(|e| Error::invariant(e.to_string()))?
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    assert_eq!(pool.inspect().retained_bytes, 64);
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    first.await.map_err(|e| Error::invariant(e.to_string()))??;
    assert_eq!(pool.run(64, |_| Ok(9)).await?, 9);
    drop(pending);
    Ok(())
}

#[tokio::test]
async fn staged_initial_credit_waits_without_rejecting_a_retained_request() -> Result<()> {
    use crate::cpu::{StagingPlan, StagingQuote};

    let handle = handle_with_cpu(CpuConfig {
        workers: 1,
        max_jobs: 3,
        max_bytes: 64,
        ..CpuConfig::default()
    })?;
    let occupied = handle.ingress.reserve(RequestId::new(2)?, 48)?;
    let submitting = handle.clone();
    let mut task = tokio::spawn(async move {
        submitting
            .submit_staged_preparing(
                RequestId::ONE,
                StagingQuote::new(3, 25)?,
                None,
                None,
                |_| {
                    let request = serde_json::from_value(serde_json::json!({
                        "id": 1, "model": 1, "input": {"Sequence": {"tokens": [1, 2, 3]}},
                        "workload": {"Generate": {"max_new_tokens": 2}}
                    }))
                    .map_err(|e| Error::invalid(e.to_string()))?;
                    Ok(StagingPlan::<()>::Ready(request, None))
                },
                |(), _| Err(Error::invariant("unexpected second preparation stage")),
            )
            .await
    });
    let waiting = timeout(Duration::from_millis(50), &mut task).await;
    if waiting.is_err() {
        assert_eq!(handle.cpu_pool().run(1, |_| Ok(7)).await?, 7);
    }
    drop(occupied);
    assert!(waiting.is_err(), "initial staging credit should wait");
    let mut output = timeout(Duration::from_secs(5), task)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))??;
    while let Some(event) = timeout(Duration::from_secs(5), output.recv())
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
    {
        if matches!(event?, EngineOutput::Finished(_)) {
            break;
        }
    }
    handle.shutdown().await
}

#[tokio::test]
async fn staged_requests_make_progress_under_a_shared_credit_limit() -> Result<()> {
    use crate::cpu::{StagingPlan, StagingQuote};
    use std::sync::Barrier;

    let ir = support::model(ModelId::ONE);
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels)?;
    let handle = RuntimeHandle::start_with_config(
        Engine::new(
            support::ProtocolBackend::new(16, 8, &ir)?,
            ir,
            PrecisionPlan::f32(),
            &kernels,
            RuntimeConfig::default(),
        )?,
        CpuConfig {
            workers: 2,
            max_jobs: 2,
            max_bytes: 64,
            ..CpuConfig::default()
        },
    )?;
    let barrier = Arc::new(Barrier::new(2));
    let mut tasks = Vec::new();
    for value in 1..=2 {
        let handle = handle.clone();
        let barrier = barrier.clone();
        let id = RequestId::new(value)?;
        tasks.push(tokio::spawn(async move {
            handle
                .submit_staged_preparing(
                    id,
                    StagingQuote::new(25, 25)?,
                    None,
                    None,
                    move |_| {
                        barrier.wait();
                        Ok(StagingPlan::Deferred(id, StagingQuote::new(3, 48)?))
                    },
                    |id, _| {
                        let request = serde_json::from_value(serde_json::json!({
                        "id": id.get(), "model": 1, "input": {"Sequence": {"tokens": [1, 2, 3]}},
                        "workload": {"Generate": {"max_new_tokens": 2}}
                    })).map_err(|e| Error::invalid(e.to_string()))?;
                        Ok((request, None))
                    },
                )
                .await
        }));
    }
    for task in tasks {
        let mut output = timeout(Duration::from_secs(5), task)
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
            .map_err(|e| Error::invariant(e.to_string()))??;
        while let Some(event) = timeout(Duration::from_secs(5), output.recv())
            .await
            .map_err(|e| Error::invariant(e.to_string()))?
        {
            if matches!(event?, EngineOutput::Finished(_)) {
                break;
            }
        }
    }
    handle.shutdown().await
}

#[tokio::test]
async fn many_staged_images_finish_with_bounded_shared_credit() -> Result<()> {
    use crate::cpu::{StagingPlan, StagingQuote};
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;

    let handle = handle_with_cpu(CpuConfig {
        workers: 2,
        max_jobs: 32,
        max_bytes: 1000,
        ..CpuConfig::default()
    })?;
    let barrier = Arc::new(Barrier::new(2));
    let arrivals = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::new();
    for value in 1..=16 {
        let handle = handle.clone();
        let barrier = barrier.clone();
        let arrivals = arrivals.clone();
        let id = RequestId::new(value)?;
        tasks.push(tokio::spawn(async move {
            handle
                .submit_staged_preparing(
                    id,
                    StagingQuote::new(2, 260)?,
                    None,
                    None,
                    move |_| {
                        if arrivals.fetch_add(1, Ordering::Relaxed) < 2 {
                            barrier.wait();
                        }
                        Ok(StagingPlan::Deferred(id, StagingQuote::new(40, 480)?))
                    },
                    |id, _| {
                        let request = serde_json::from_value(serde_json::json!({
                            "id": id.get(), "model": 1,
                            "input": {"Sequence": {"tokens": [1, 2, 3]}},
                            "workload": {"Generate": {"max_new_tokens": 2}}
                        }))
                        .map_err(|error| Error::invalid(error.to_string()))?;
                        Ok((request, None))
                    },
                )
                .await
        }));
    }
    for task in tasks {
        let mut output = timeout(Duration::from_secs(10), task)
            .await
            .map_err(|error| Error::invariant(error.to_string()))?
            .map_err(|error| Error::invariant(error.to_string()))??;
        while let Some(event) = timeout(Duration::from_secs(10), output.recv())
            .await
            .map_err(|error| Error::invariant(error.to_string()))?
        {
            if matches!(event?, EngineOutput::Finished(_)) {
                break;
            }
        }
    }
    handle.shutdown().await
}

#[tokio::test]
async fn funded_staged_request_completes_without_a_second_worker_job() -> Result<()> {
    use crate::cpu::{StagingPlan, StagingQuote};

    let handle = handle()?;
    let second_job = Arc::new(AtomicBool::new(false));
    let observed = second_job.clone();
    let mut output = handle
        .submit_staged_preparing(
            RequestId::ONE,
            StagingQuote::new(48, 48)?,
            None,
            None,
            |_| {
                let request = serde_json::from_value(serde_json::json!({
                    "id": 1, "model": 1, "input": {"Sequence": {"tokens": [1, 2, 3]}},
                    "workload": {"Generate": {"max_new_tokens": 2}}
                }))
                .map_err(|e| Error::invalid(e.to_string()))?;
                Ok(StagingPlan::<()>::Ready(request, None))
            },
            move |(), _| {
                observed.store(true, Ordering::Release);
                Err(Error::invariant("unneeded second preparation job"))
            },
        )
        .await?;
    while let Some(event) = timeout(Duration::from_secs(5), output.recv())
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
    {
        if matches!(event?, EngineOutput::Finished(_)) {
            break;
        }
    }
    assert!(!second_job.load(Ordering::Acquire));
    handle.shutdown().await
}

struct CreditDropProbe {
    registry: Arc<Registry>,
    other: RequestId,
    observed: Arc<AtomicBool>,
}

impl Drop for CreditDropProbe {
    fn drop(&mut self) {
        if self.registry.reserve(self.other, 1).is_err() {
            self.observed.store(true, Ordering::Release);
        }
    }
}

#[tokio::test]
async fn cancelled_initial_staging_drops_both_closures_under_credit() -> Result<()> {
    use crate::cpu::{StagingPlan, StagingQuote};

    let handle = handle_with_cpu(CpuConfig {
        workers: 1,
        max_jobs: 4,
        max_bytes: 64,
        ..CpuConfig::default()
    })?;
    let blocker = handle.ingress.reserve(RequestId::new(2)?, 61)?;
    let load_dropped = Arc::new(AtomicBool::new(false));
    let finish_dropped = Arc::new(AtomicBool::new(false));
    let load_payload = CreditDropProbe {
        registry: handle.ingress.clone(),
        other: RequestId::new(3)?,
        observed: load_dropped.clone(),
    };
    let finish_payload = CreditDropProbe {
        registry: handle.ingress.clone(),
        other: RequestId::new(3)?,
        observed: finish_dropped.clone(),
    };
    let submitting = handle.clone();
    let task = tokio::spawn(async move {
        submitting
            .submit_staged_preparing(
                RequestId::ONE,
                StagingQuote::new(3, 25)?,
                None,
                None,
                move |_| {
                    drop(load_payload);
                    Err::<StagingPlan<()>, _>(Error::invariant("unexpected load"))
                },
                move |(), _| {
                    drop(finish_payload);
                    Err(Error::invariant("unexpected finish"))
                },
            )
            .await
    });
    timeout(Duration::from_secs(2), async {
        while handle.ingress.reserve(RequestId::new(3)?, 1).is_ok() {
            sleep(Duration::from_millis(1)).await;
        }
        Result::<()>::Ok(())
    })
    .await
    .map_err(|e| Error::invariant(e.to_string()))??;
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    assert!(load_dropped.load(Ordering::Acquire));
    assert!(finish_dropped.load(Ordering::Acquire));
    drop(blocker);
    drop(handle.ingress.reserve(RequestId::new(4)?, 64)?);
    handle.shutdown().await
}

#[tokio::test]
async fn cancelled_staging_drops_its_input_before_releasing_ingress_credit() -> Result<()> {
    use crate::cpu::{StagingPlan, StagingQuote};

    const BUDGET: usize = 64 << 20;
    let handle = handle()?;
    let blocker = handle.ingress.reserve(RequestId::new(2)?, BUDGET - 4)?;
    let observed = Arc::new(AtomicBool::new(false));
    let payload = CreditDropProbe {
        registry: handle.ingress.clone(),
        other: RequestId::new(3)?,
        observed: observed.clone(),
    };
    let (started, entered) = oneshot::channel();
    let submitting = handle.clone();
    let mut task = tokio::spawn(async move {
        submitting
            .submit_staged_preparing(
                RequestId::ONE,
                StagingQuote::new(4, 4)?,
                None,
                None,
                move |_| {
                    started
                        .send(())
                        .map_err(|()| Error::invariant("start receiver dropped"))?;
                    Ok(StagingPlan::Deferred(payload, StagingQuote::new(4, 48)?))
                },
                |_, _| Err(Error::invariant("blocked input unexpectedly ran")),
            )
            .await
    });
    timeout(Duration::from_secs(2), entered)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))?;
    timeout(Duration::from_secs(2), async {
        while handle.cpu_inspection().retained_jobs != 0 {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|e| Error::invariant(e.to_string()))?;
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    assert!(observed.load(Ordering::Acquire));
    drop(blocker);
    handle.shutdown().await
}

#[tokio::test]
async fn cancelled_cpu_wait_drops_its_staged_input_under_ingress_credit() -> Result<()> {
    use crate::cpu::{StagingPlan, StagingQuote};

    const BUDGET: usize = 64 << 20;
    let handle = handle_with_cpu(CpuConfig {
        workers: 2,
        max_jobs: 4,
        max_bytes: BUDGET,
        ..CpuConfig::default()
    })?;
    let pool = handle.cpu_pool();
    let (started, entered) = oneshot::channel();
    let (release, gate) = mpsc::channel();
    let occupied = tokio::spawn(async move {
        pool.run(BUDGET - 4, move |_| {
            started
                .send(())
                .map_err(|()| Error::invariant("start receiver dropped"))?;
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
            Ok(())
        })
        .await
    });
    timeout(Duration::from_secs(2), entered)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))?;
    let observed = Arc::new(AtomicBool::new(false));
    let payload = CreditDropProbe {
        registry: handle.ingress.clone(),
        other: RequestId::new(2)?,
        observed: observed.clone(),
    };
    let (loaded, ready) = oneshot::channel();
    let submitting = handle.clone();
    let mut task = tokio::spawn(async move {
        submitting
            .submit_staged_preparing(
                RequestId::ONE,
                StagingQuote::new(4, 4)?,
                None,
                None,
                move |_| {
                    loaded
                        .send(())
                        .map_err(|()| Error::invariant("start receiver dropped"))?;
                    Ok(StagingPlan::Deferred(
                        payload,
                        StagingQuote::new(4, BUDGET)?,
                    ))
                },
                |_, _| Err(Error::invariant("blocked input unexpectedly ran")),
            )
            .await
    });
    timeout(Duration::from_secs(2), ready)
        .await
        .map_err(|e| Error::invariant(e.to_string()))?
        .map_err(|e| Error::invariant(e.to_string()))?;
    timeout(Duration::from_secs(2), async {
        while handle.ingress.reserve(RequestId::new(2)?, 1).is_ok() {
            sleep(Duration::from_millis(1)).await;
        }
        Result::<()>::Ok(())
    })
    .await
    .map_err(|e| Error::invariant(e.to_string()))??;
    assert!(timeout(Duration::from_millis(20), &mut task).await.is_err());
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    assert!(observed.load(Ordering::Acquire));
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    occupied
        .await
        .map_err(|e| Error::invariant(e.to_string()))??;
    handle.shutdown().await
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
