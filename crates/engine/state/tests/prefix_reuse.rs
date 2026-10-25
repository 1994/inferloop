use infer_core::{Error, Result};
use infer_state::cache::{PrefixCache, PrefixReuse, PrefixSnapshot};
use std::{cell::Cell, rc::Rc};

// Intentionally not Clone: a backend may own unique device snapshots rather than page leases.
struct OwnedSnapshot {
    layout: u32,
    drops: Rc<Cell<usize>>,
}
impl Drop for OwnedSnapshot {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}
struct Consumer {
    layout: u32,
    fail: bool,
    calls: usize,
}
impl PrefixReuse<OwnedSnapshot> for Consumer {
    type Output = usize;
    type Error = Error;
    fn compatible(&self, _: &[u32], value: &OwnedSnapshot) -> bool {
        value.layout == self.layout
    }
    fn reuse(&mut self, tokens: &[u32], _: &OwnedSnapshot) -> Result<usize> {
        self.calls += 1;
        if self.fail {
            return Err(Error::invalid("injected backend failure"));
        }
        Ok(tokens.len())
    }
}
fn publish(
    cache: &mut PrefixCache<OwnedSnapshot>,
    tokens: &[u32],
    layout: u32,
    drops: &Rc<Cell<usize>>,
) -> Result<()> {
    cache
        .publish(PrefixSnapshot {
            tokens: tokens.to_vec(),
            value: OwnedSnapshot {
                layout,
                drops: drops.clone(),
            },
            bytes: 4,
        })
        .map(drop)
}
const fn consumer(layout: u32, fail: bool) -> Consumer {
    Consumer {
        layout,
        fail,
        calls: 0,
    }
}

#[test]
fn backend_failure_retains_source_budget_and_lru_until_cleanup() {
    let drops = Rc::new(Cell::new(0));
    let mut cache = PrefixCache::new(b"test", 2, 8, 2).unwrap();
    publish(&mut cache, &[1, 2], 0, &drops).unwrap();
    publish(&mut cache, &[3, 4], 0, &drops).unwrap();
    let mut backend = consumer(0, true);
    assert!(cache.reuse(&[1, 2, 9], 3, &mut backend).is_err());
    assert_eq!(
        (
            backend.calls,
            cache.lookups,
            cache.hits,
            cache.reused_tokens
        ),
        (1, 1, 0, 0)
    );
    assert_eq!(
        (
            cache.len(),
            cache.bytes(),
            cache.cached_tokens(),
            drops.get()
        ),
        (2, 8, 4, 0)
    );
    let oldest = cache.evict_oldest().unwrap();
    assert!(
        !cache.contains(&[1, 2]),
        "failed reuse must not refresh the oldest snapshot"
    );
    assert!(cache.contains(&[3, 4]));
    assert_eq!(
        drops.get(),
        0,
        "backend cleanup still owns the evicted snapshot"
    );
    drop(oldest);
    assert_eq!(drops.get(), 1);
    drop(cache);
    assert_eq!(drops.get(), 2);
}
#[test]
fn longest_compatible_prefix_wins_without_counting_declined_candidates() {
    let drops = Rc::new(Cell::new(0));
    let mut cache = PrefixCache::new(b"test", 2, 100, 4).unwrap();
    publish(&mut cache, &[1, 2], 0, &drops).unwrap();
    publish(&mut cache, &[1, 2, 3, 4], 1, &drops).unwrap();
    let mut backend = consumer(0, false);
    assert_eq!(
        cache.reuse(&[1, 2, 3, 4, 5], 5, &mut backend).unwrap(),
        Some(2)
    );
    assert_eq!(
        (
            backend.calls,
            cache.lookups,
            cache.hits,
            cache.reused_tokens
        ),
        (1, 1, 1, 2)
    );
    assert_eq!(cache.reuse(&[1, 2], 1, &mut backend).unwrap(), None);
    let mut other = consumer(2, false);
    assert_eq!(cache.reuse(&[1, 2, 3, 4], 4, &mut other).unwrap(), None);
    assert_eq!((other.calls, cache.lookups, cache.hits), (0, 3, 1));
    cache.evict_oldest();
    assert!(
        !cache.contains(&[1, 2, 3, 4]),
        "only the successfully reused short prefix was touched"
    );
    assert!(cache.contains(&[1, 2]));
    assert_eq!(drops.get(), 1);
}
#[test]
fn publication_returns_rejected_and_replaced_resources_and_tracks_coverage() {
    let drops = Rc::new(Cell::new(0));
    let mut cache = PrefixCache::new(b"test", 2, 4, 2).unwrap();
    publish(&mut cache, &[1, 2], 0, &drops).unwrap();
    let rejected = cache
        .publish(PrefixSnapshot {
            tokens: vec![3, 4],
            value: OwnedSnapshot {
                layout: 1,
                drops: drops.clone(),
            },
            bytes: 5,
        })
        .unwrap();
    assert_eq!(rejected.len(), 1);
    assert_eq!(
        (
            cache.len(),
            cache.bytes(),
            cache.cached_tokens(),
            drops.get()
        ),
        (1, 4, 2, 0)
    );
    drop(rejected);
    let replaced = cache
        .publish(PrefixSnapshot {
            tokens: vec![1, 2],
            value: OwnedSnapshot {
                layout: 2,
                drops: drops.clone(),
            },
            bytes: 3,
        })
        .unwrap();
    assert_eq!(replaced.len(), 1);
    assert_eq!(drops.get(), 1);
    drop(replaced);
    assert_eq!(
        (
            cache.len(),
            cache.bytes(),
            cache.cached_tokens(),
            drops.get()
        ),
        (1, 3, 2, 2)
    );
    cache.evict_oldest();
    assert_eq!(
        (
            cache.len(),
            cache.bytes(),
            cache.cached_tokens(),
            drops.get()
        ),
        (0, 0, 0, 3)
    );
}

#[test]
fn scan_and_block_index_share_matching_budget_and_lru_semantics_through_churn() {
    use infer_state::cache::{BlockHash, SnapshotScan};
    let mut indexed = PrefixCache::with_policy(b"index", 2, 12, 3, BlockHash).unwrap();
    let mut scanned = PrefixCache::with_policy(b"scan", 2, 12, 3, SnapshotScan).unwrap();
    let mut random = 42_u64;
    for value in 0..3000_u64 {
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let tokens: Vec<_> = (0..6)
            .map(|i| u32::try_from((random >> (i * 2)) % 3).unwrap())
            .collect();
        if random & 32 == 0 {
            let length = usize::try_from((random % 3 + 1) * 2).unwrap();
            assert_eq!(
                indexed.insert(tokens[..length].to_vec(), value, 4).unwrap(),
                scanned.insert(tokens[..length].to_vec(), value, 4).unwrap()
            );
        } else {
            let maximum = usize::try_from(random % 8).unwrap();
            assert_eq!(
                indexed.lookup_where(&tokens, maximum, |v| v % 2 == 0),
                scanned.lookup_where(&tokens, maximum, |v| v % 2 == 0)
            );
        }
        assert_eq!(
            (indexed.len(), indexed.bytes(), indexed.cached_tokens()),
            (scanned.len(), scanned.bytes(), scanned.cached_tokens())
        );
        assert_eq!(
            (
                indexed.lookups,
                indexed.hits,
                indexed.reused_tokens,
                indexed.evictions
            ),
            (
                scanned.lookups,
                scanned.hits,
                scanned.reused_tokens,
                scanned.evictions
            )
        );
    }
    while !indexed.is_empty() {
        assert_eq!(indexed.evict_oldest(), scanned.evict_oldest());
    }
    assert!(scanned.is_empty());
}

#[test]
fn length_shortcut_keeps_remaining_and_republished_prefixes_reachable() {
    let mut cache = PrefixCache::new(b"lengths", 2, 100, 4).unwrap();
    cache.insert(vec![1, 2], 2, 4).unwrap();
    cache.insert(vec![1, 2, 3, 4], 4, 4).unwrap();
    assert_eq!(cache.evict_oldest(), Some(2));
    assert_eq!(cache.lookup(&[1, 2, 3, 4], 4), Some(4));
    assert_eq!(cache.evict_oldest(), Some(4));
    cache.insert(vec![1, 2, 3, 4, 5, 6], 6, 4).unwrap();
    assert_eq!(cache.lookup(&[1, 2, 3, 4, 5, 6], 5), None);
    assert_eq!(cache.lookup(&[1, 2, 3, 4, 5, 6], 6), Some(6));
    cache.insert(vec![1, 2], 2, 4).unwrap();
    assert_eq!(cache.lookup(&[1, 2, 7, 8], 4), Some(2));
}

#[test]
fn publication_transfers_every_retirement_to_the_cleanup_owner() {
    let drops = Rc::new(Cell::new(0));
    let mut cache = PrefixCache::new(b"batch", 2, 12, 3).unwrap();
    for tokens in [[1, 2], [3, 4], [5, 6]] {
        publish(&mut cache, &tokens, 0, &drops).unwrap();
    }
    let retired = cache
        .publish(PrefixSnapshot {
            tokens: vec![7, 8],
            value: OwnedSnapshot {
                layout: 1,
                drops: drops.clone(),
            },
            bytes: 12,
        })
        .unwrap();
    assert_eq!(retired.len(), 3);
    assert!(!retired.is_empty());
    assert_eq!((cache.len(), cache.bytes(), drops.get()), (1, 12, 0));
    let mut cleanup = retired.into_iter();
    drop(cleanup.next());
    assert_eq!(drops.get(), 1);
    drop(cleanup);
    assert_eq!(drops.get(), 3);
    drop(cache);
    assert_eq!(drops.get(), 4);
}
