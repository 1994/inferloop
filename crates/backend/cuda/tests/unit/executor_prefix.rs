use super::*;
use std::convert::Infallible;
use std::result::Result as StdResult;

fn entry(tokens: &[u32], bytes: u64) -> CachedPrefix {
    CachedPrefix {
        tokens: tokens.to_vec(),
        value: Snapshot {
            target: ProgramSnapshot::default(),
            draft: None,
        },
        bytes,
    }
}
fn cache(bytes: u64) -> Result<PrefixCache> {
    // Small fixture boundaries; production uses PREFIX_GRANULARITY.
    PrefixCache::with_matcher(b"cuda-fixture", 1, bytes, usize::MAX, SnapshotScan)
}
struct ReadTokens;
impl PrefixReuse<Snapshot> for ReadTokens {
    type Output = Vec<u32>;
    type Error = Infallible;
    fn compatible(&self, _: &[u32], _: &Snapshot) -> bool {
        true
    }
    fn reuse(&mut self, tokens: &[u32], _: &Snapshot) -> StdResult<Vec<u32>, Infallible> {
        Ok(tokens.to_vec())
    }
}
#[test]
fn restores_only_complete_snapshots_within_the_cap() {
    let mut cache = cache(1024).unwrap();
    cache.publish(entry(&[1, 2, 3, 4], 10)).unwrap();
    cache.publish(entry(&[1, 2, 9], 10)).unwrap();
    assert_eq!(cache.matched_tokens(&[1, 2, 3, 5], 4), 0);
    assert_eq!(cache.matched_tokens(&[1, 2, 3, 4, 5], 4), 4);
    assert_eq!(cache.matched_tokens(&[1, 2, 3, 4, 5], 2), 0);
    assert!(
        cache
            .reuse(&[1, 2, 3, 5], 4, &mut ReadTokens)
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.matched_tokens(&[7, 7], 4), 0);
}
#[test]
fn successful_reuse_counts_after_the_hook_and_keeps_the_snapshot_published() {
    let mut cache = cache(1024).unwrap();
    cache.publish(entry(&[4, 5, 6], 10)).unwrap();
    assert_eq!(
        cache.reuse(&[4, 5, 6, 7], 4, &mut ReadTokens).unwrap(),
        Some(vec![4, 5, 6])
    );
    assert_eq!((cache.lookups, cache.hits, cache.reused_tokens), (1, 1, 3));
    assert_eq!(
        (cache.len(), cache.bytes(), cache.cached_tokens()),
        (1, 10, 3)
    );
    assert_eq!(cache.matched_tokens(&[4, 5, 6, 7], 4), 3);
}
#[test]
fn insert_deduplicates_and_evicts_by_capacity() {
    let mut cache = cache(20).unwrap();
    cache.publish(entry(&[1], 10)).unwrap();
    assert_eq!(cache.publish(entry(&[1], 10)).unwrap().len(), 1);
    assert_eq!(cache.len(), 1);
    cache.publish(entry(&[2], 10)).unwrap();
    cache.publish(entry(&[3], 10)).unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.bytes(), 20);
    assert_eq!(cache.matched_tokens(&[1], 1), 0);
    assert_eq!(cache.publish(entry(&[9], 21)).unwrap().len(), 1);
    assert_eq!((cache.len(), cache.cached_tokens()), (2, 2));
}
