use infer_core::{Error, Result};
use infer_frontdoor::{cpu::CpuConfig, cpu::CpuPool};
use std::panic::resume_unwind;
use std::thread::{current, yield_now};
use std::{sync::Arc, sync::Barrier, sync::mpsc, time::Duration};
use tokio::sync::mpsc as async_mpsc;
use tokio::time::{sleep, timeout};

fn config() -> CpuConfig {
    CpuConfig {
        workers: 2,
        max_jobs: 4,
        ..Default::default()
    }
}

#[tokio::test]
async fn default_preparation_accepts_a_large_staging_request() -> Result<()> {
    const LARGE_REQUEST_BYTES: usize = 128 * 1024 * 1024;
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 2,
        ..CpuConfig::default()
    })?;
    assert_eq!(pool.run(LARGE_REQUEST_BYTES, |_| Ok(7)).await?, 7);
    assert_eq!(pool.inspect().retained_bytes, 0);
    Ok(())
}

#[tokio::test]
async fn sequential_preparation_reuses_the_ready_worker() -> Result<()> {
    let pool = CpuPool::new(CpuConfig {
        workers: 4,
        max_jobs: 8,
        ..CpuConfig::default()
    })?;
    let barrier = Arc::new(Barrier::new(4));
    let mut startup = Vec::new();
    for _ in 0..4 {
        let pool = pool.clone();
        let barrier = barrier.clone();
        startup.push(tokio::spawn(async move {
            pool.run(0, move |_| {
                barrier.wait();
                Ok(())
            })
            .await
        }));
    }
    for job in startup {
        timeout(Duration::from_secs(5), job)
            .await
            .map_err(|error| Error::invariant(error.to_string()))?
            .map_err(|error| Error::invariant(error.to_string()))??;
    }
    let first = pool.run(0, |_| Ok(current().id())).await?;
    for _ in 0..100 {
        assert_eq!(pool.run(0, |_| Ok(current().id())).await?, first);
    }
    Ok(())
}

#[tokio::test]
async fn simultaneous_preparation_uses_all_available_workers() -> Result<()> {
    const WORKERS: usize = 4;
    let pool = CpuPool::new(CpuConfig {
        workers: WORKERS,
        max_jobs: WORKERS,
        ..CpuConfig::default()
    })?;
    let (started, mut entered) = async_mpsc::channel(WORKERS);
    let mut releases = Vec::new();
    let mut jobs = Vec::new();
    for index in 0..WORKERS {
        let (release, gate) = mpsc::channel();
        releases.push(release);
        let pool = pool.clone();
        let started = started.clone();
        jobs.push(tokio::spawn(async move {
            pool.run(0, move |_| {
                started
                    .blocking_send(index)
                    .map_err(|error| Error::invariant(error.to_string()))?;
                gate.recv()
                    .map_err(|error| Error::invariant(error.to_string()))?;
                Ok(index)
            })
            .await
        }));
    }
    let mut seen = Vec::new();
    for _ in 0..WORKERS {
        let index = timeout(Duration::from_secs(5), entered.recv())
            .await
            .map_err(|error| Error::invariant(error.to_string()))?
            .ok_or_else(|| Error::invariant("worker did not start"))?;
        seen.push(index);
    }
    seen.sort_unstable();
    assert_eq!(seen, (0..WORKERS).collect::<Vec<_>>());
    for release in releases {
        release
            .send(())
            .map_err(|error| Error::invariant(error.to_string()))?;
    }
    for job in jobs {
        timeout(Duration::from_secs(5), job)
            .await
            .map_err(|error| Error::invariant(error.to_string()))?
            .map_err(|error| Error::invariant(error.to_string()))??;
    }
    assert_eq!(pool.inspect().peak_workers, WORKERS);
    assert_eq!(pool.inspect().retained_jobs, 0);
    Ok(())
}

#[tokio::test]
async fn repeated_bursts_finish_without_lost_wakeups() -> Result<()> {
    const WORKERS: usize = 4;
    const BURST: usize = 32;
    const ROUNDS: usize = 32;
    let pool = CpuPool::new(CpuConfig {
        workers: WORKERS,
        max_jobs: BURST,
        ..CpuConfig::default()
    })?;
    for round in 0..ROUNDS {
        let mut jobs = Vec::with_capacity(BURST);
        for index in 0..BURST {
            let pool = pool.clone();
            jobs.push(tokio::spawn(async move {
                pool.run(0, move |_| {
                    if index % WORKERS == 0 {
                        yield_now();
                    }
                    Ok((round, index))
                })
                .await
            }));
        }
        for (index, job) in jobs.into_iter().enumerate() {
            let value = timeout(Duration::from_secs(5), job)
                .await
                .map_err(|error| Error::invariant(error.to_string()))?
                .map_err(|error| Error::invariant(error.to_string()))??;
            assert_eq!(value, (round, index));
        }
        assert_eq!(pool.inspect().retained_jobs, 0);
    }
    Ok(())
}
#[tokio::test]
async fn preparation_workers_run_concurrently_and_a_blocked_job_does_not_block_delivery()
-> Result<()> {
    let pool = CpuPool::new(config())?;
    let barrier = Arc::new(Barrier::new(2));
    let first = pool.clone();
    let a = barrier.clone();
    let one = tokio::spawn(async move {
        first
            .run(32, move |_| {
                a.wait();
                Ok(1)
            })
            .await
    });
    let second = pool.clone();
    let two = tokio::spawn(async move {
        second
            .run(32, move |_| {
                barrier.wait();
                Ok(2)
            })
            .await
    });
    assert_eq!(one.await.map_err(|e| Error::invariant(e.to_string()))??, 1);
    assert_eq!(two.await.map_err(|e| Error::invariant(e.to_string()))??, 2);
    assert_eq!(pool.inspect().peak_workers, 2);
    let (release, gate) = mpsc::channel();
    let worker = pool.clone();
    let blocked = tokio::spawn(async move {
        worker
            .run(32, move |_| {
                gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
                Ok(())
            })
            .await
    });
    assert_eq!(pool.run(32, |_| Ok(3)).await?, 3);
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    blocked
        .await
        .map_err(|e| Error::invariant(e.to_string()))??;
    assert_eq!(pool.inspect().retained_bytes, 0);
    Ok(())
}
#[tokio::test]
async fn completed_jobs_release_credit_before_delivering_results() -> Result<()> {
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        ..config()
    })?;
    for _ in 0..1000 {
        pool.run(32, |_| Ok(())).await?;
        assert_eq!(pool.inspect().retained_jobs, 0);
        assert_eq!(pool.inspect().retained_bytes, 0);
        assert!(
            pool.run::<()>(32, |_| Err(Error::invalid("provider failed")))
                .await
                .is_err()
        );
        assert_eq!(pool.inspect().retained_jobs, 0);
        assert_eq!(pool.inspect().retained_bytes, 0);
    }
    Ok(())
}
#[tokio::test]
async fn timeout_keeps_byte_credit_until_the_actual_cpu_job_exits() -> Result<()> {
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        max_jobs: 1,
        job_timeout_ms: 20,
        ..config()
    })?;
    let (release, gate) = mpsc::channel();
    let result = pool
        .run(32, move |_| {
            gate.recv().map_err(|e| Error::invariant(e.to_string()))?;
            Ok(())
        })
        .await;
    assert!(result.is_err());
    assert_eq!(pool.inspect().retained_bytes, 32);
    assert!(pool.run(1, |_| Ok(())).await.is_err());
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    for _ in 0..100 {
        if pool.inspect().retained_jobs == 0 {
            break;
        }
        sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(pool.inspect().retained_jobs, 0);
    assert_eq!(pool.run(1, |_| Ok(7)).await?, 7);
    Ok(())
}
#[tokio::test]
async fn failed_provider_does_not_destroy_worker_or_leak_admission_credit() -> Result<()> {
    let pool = CpuPool::new(CpuConfig {
        workers: 1,
        ..config()
    })?;
    let result = pool.run::<()>(32, |_| resume_unwind(Box::new(()))).await;
    assert!(result.is_err());
    assert_eq!(pool.run(32, |_| Ok(9)).await?, 9);
    assert_eq!(pool.inspect().retained_bytes, 0);
    Ok(())
}
