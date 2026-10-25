use super::PrefixCache;
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
