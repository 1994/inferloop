use super::{StagingLimit, budget_for_available};
use crate::cpu::{CpuConfig, CpuPool};
use crate::ingress::Registry;
use infer_core::{RequestId, Result};
use std::sync::Arc;
use std::sync::atomic::Ordering;

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
fn admission_uses_the_same_growing_limit_and_releases_credits() -> Result<()> {
    const IMAGE_CREDIT: usize = 25 * 1024 * 1024;
    const INITIAL: usize = 64 * 1024 * 1024;
    const EXPANDED: usize = 512 * 1024 * 1024;
    let limit = Arc::new(StagingLimit::automatic());
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
