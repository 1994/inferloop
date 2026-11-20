use super::{PrefixCache, PrefixId, PrefixRecency};
#[test]
fn clock_rollover_keeps_eviction_order_independent_of_hash_iteration() {
    let mut cache = PrefixCache::new(b"rollover", 2, 100, 16).unwrap();
    for value in 0..16_u32 {
        cache.insert(vec![value, value], value, 4).unwrap();
    }
    assert_eq!(cache.lookup(&[0, 0], 2), Some(0));
    cache.clock = u64::MAX;
    assert_eq!(cache.lookup(&[2, 2], 2), Some(2));
    let expected = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2];
    for value in expected {
        assert_eq!(cache.evict_oldest(), Some(value));
    }
    assert!(cache.is_empty());
    assert_eq!(cache.bytes(), 0);
}

#[test]
fn forced_fingerprint_collision_cannot_restore_different_tokens() {
    let mut cache = PrefixCache::new(b"collision", 2, 100, 16).unwrap();
    cache.insert(vec![1, 2], 10, 4).unwrap();
    let original = cache.key(&[1, 2]);
    let collision = cache.key(&[9, 8]);
    // Put a real entry under the other tokens' fingerprint to force the index collision.
    let entry = cache.entries.remove(&original).unwrap();
    cache.lru.remove(PrefixId(original), entry.touched);
    cache.lru.insert(PrefixId(collision), entry.touched);
    cache.entries.insert(collision, entry);
    assert!(!cache.contains(&[9, 8]));
    assert!(!cache.contains_where(&[9, 8], |_| panic!("token mismatch must short-circuit")));
    assert_eq!(cache.matched_tokens(&[9, 8, 7], 3), 0);
    assert_eq!(
        cache.lookup_where(&[9, 8], 2, |_| panic!("token mismatch must short-circuit")),
        None
    );
    assert_eq!(cache.hits, 0);
    assert_eq!(cache.bytes(), 4);
    assert_eq!(cache.insert(vec![9, 8], 20, 8).unwrap(), vec![10]);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.bytes(), 8);
    assert_eq!(cache.lookup(&[9, 8], 2), Some(20));
    assert_eq!(cache.evict_oldest(), Some(20));
    assert!(cache.is_empty());
    assert_eq!(cache.bytes(), 0);
}

#[test]
fn incremental_boundaries_match_publication_for_partial_and_long_prefixes() {
    let tokens: Vec<u32> = (0..257).map(|i| i % 17).collect();
    for block_size in [1, 2, 16, 64] {
        let cache = PrefixCache::<u32>::new(b"boundaries", block_size, 100, 16).unwrap();
        for (length, key) in cache.keys(&tokens) {
            assert_eq!(key, cache.key(&tokens[..length]));
        }
    }
    let mut cache = PrefixCache::new(b"long", 16, 100, 16).unwrap();
    for length in [16, 32, 128, 256] {
        cache.insert(tokens[..length].to_vec(), length, 4).unwrap();
    }
    assert_eq!(cache.lookup(&tokens, 257), Some(256));
    assert_eq!(cache.lookup(&tokens, 255), Some(128));
    assert_eq!(cache.lookup(&tokens, 15), None);
    assert_eq!(
        cache.lookup_where(&tokens, 257, |length| *length <= 32),
        Some(32)
    );
    let mut changed = tokens;
    changed[16] = 99;
    assert_eq!(cache.lookup(&changed, 257), Some(16));
}

#[test]
fn snapshot_queue_keeps_lru_order_across_clock_rollover() {
    use super::SnapshotScan;
    let mut cache = PrefixCache::with_policy(b"rollover-scan", 2, 100, 16, SnapshotScan).unwrap();
    for value in 0..16_u32 {
        cache.insert(vec![value, value], value, 4).unwrap();
    }
    assert_eq!(cache.lookup(&[0, 0], 2), Some(0));
    cache.clock = u64::MAX;
    assert_eq!(cache.lookup(&[2, 2], 2), Some(2));
    for value in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 2] {
        assert_eq!(cache.evict_oldest(), Some(value));
    }
    assert!(cache.is_empty());
}

#[test]
fn exhausted_snapshot_ids_preserve_live_entries_and_allow_replacement() {
    use super::SnapshotScan;
    let mut cache = PrefixCache::with_policy(b"ids", 2, 8, 2, SnapshotScan).unwrap();
    cache.insert(vec![1, 2], 10, 4).unwrap();
    cache.next_id = [u64::MAX; 2];
    assert!(cache.insert(vec![3, 4], 20, 4).is_err());
    assert_eq!(
        (cache.len(), cache.bytes(), cache.cached_tokens()),
        (1, 4, 2)
    );
    assert_eq!(cache.lookup(&[1, 2], 2), Some(10));
    assert_eq!(cache.insert(vec![1, 2], 30, 4).unwrap(), vec![10]);
    assert_eq!(cache.evict_oldest(), Some(30));
}
