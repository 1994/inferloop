use super::{REFRESH_INTERVAL, StagingLimit, budget_for_available};
use crate::cpu::{CpuConfig, CpuPool};
use crate::ingress::Registry;
use infer_core::{RequestId, Result};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[test]
fn automatic_budget_scales_with_available_memory() {
    const MIB: u64 = 1024 * 1024;
    assert_eq!(budget_for_available(4 * 1024 * MIB), 512 * 1024 * 1024);
    assert_eq!(budget_for_available(2 * 1024 * MIB), 256 * 1024 * 1024);
    assert_eq!(budget_for_available(0), 1);
}

#[test]
fn fixed_limit_is_not_automatically_changed() {
    let limit = StagingLimit::fixed(128);
    assert!(!limit.is_automatic());
    limit.refresh_if_due();
    assert_eq!(limit.current(), 128);
}

#[test]
fn automatic_limit_refreshes_an_outdated_snapshot() {
    let limit = StagingLimit::automatic(1);
    let refreshed = limit.refreshed.as_ref().expect("automatic refresh");
    *refreshed.lock().expect("refresh lock") = Instant::now() - REFRESH_INTERVAL;
    limit.refresh_if_due();
    assert!(limit.current() > 1, "host memory snapshot was unavailable");
}

#[test]
fn admission_uses_the_same_growing_limit_and_releases_credits() -> Result<()> {
    const IMAGE_CREDIT: usize = 25 * 1024 * 1024;
    const INITIAL: usize = 64 * 1024 * 1024;
    const EXPANDED: usize = 512 * 1024 * 1024;
    let limit = Arc::new(StagingLimit::automatic(INITIAL));
    limit.bytes.store(INITIAL, Ordering::Relaxed);
    let registry = Registry::new(17, limit.clone());
    let pool = CpuPool::named_with_limit(
        CpuConfig {
            workers: 1,
            max_jobs: 17,
            ..CpuConfig::default()
        },
        "budget-test",
        limit.clone(),
    )?;
    let mut pending = Vec::new();
    let mut credits = Vec::new();
    for id in 1..=2 {
        pending.push(registry.reserve(RequestId::new(id)?, IMAGE_CREDIT)?);
        credits.push(pool.reserve(IMAGE_CREDIT)?);
    }
    assert!(registry.reserve(RequestId::new(3)?, IMAGE_CREDIT).is_err());
    assert!(pool.reserve(IMAGE_CREDIT).is_err());

    limit.bytes.store(EXPANDED, Ordering::Relaxed);
    for id in 3..=16 {
        pending.push(registry.reserve(RequestId::new(id)?, IMAGE_CREDIT)?);
        credits.push(pool.reserve(IMAGE_CREDIT)?);
    }
    assert_eq!(pool.inspect().retained_jobs, 16);
    limit.bytes.store(INITIAL, Ordering::Relaxed);
    assert!(registry.reserve(RequestId::new(17)?, IMAGE_CREDIT).is_err());
    assert!(pool.reserve(IMAGE_CREDIT).is_err());
    drop(pending);
    drop(credits);
    assert_eq!(pool.inspect().retained_jobs, 0);
    let _pending = registry.reserve(RequestId::new(17)?, IMAGE_CREDIT)?;
    let _credit = pool.reserve(IMAGE_CREDIT)?;
    Ok(())
}

#[tokio::test]
async fn failed_growth_releases_both_admission_and_worker_credits() -> Result<()> {
    let limit = Arc::new(StagingLimit::fixed(32));
    let registry = Registry::new(3, limit.clone());
    let pool = CpuPool::named_with_limit(
        CpuConfig {
            workers: 1,
            max_jobs: 3,
            max_bytes: 32,
            ..CpuConfig::default()
        },
        "growth-test",
        limit,
    )?;
    let other_credit = pool.reserve(16)?;
    let pending = registry.reserve(RequestId::new(1)?, 8)?;
    let worker_pending = pending.clone();
    let result = pool
        .run(8, move |context| {
            context.with_pending(worker_pending).grow_staging(16)
        })
        .await;
    assert!(result.is_err());
    assert_eq!(pool.inspect().retained_bytes, 16);
    let available_after_failure = registry.reserve(RequestId::new(2)?, 24)?;
    drop(available_after_failure);
    drop(pending);
    drop(other_credit);
    assert_eq!(pool.inspect().retained_bytes, 0);
    let _pending = registry.reserve(RequestId::new(2)?, 32)?;
    let _credit = pool.reserve(32)?;
    Ok(())
}
