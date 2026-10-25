use infer_core::*;
use infer_ir::*;
use infer_state::{SequenceStateManager, blocks::BlockPool, cache::PrefixCache};
use std::iter::empty;
use std::iter::once;

#[test]
fn pool_capacity_stale_leases_and_cow_have_exact_ownership() {
    let mut p = BlockPool::new(3).unwrap();
    let mut parent = p.allocate(2).unwrap();
    assert!(p.allocate(2).is_err());
    assert_eq!(p.free_blocks(), 1);
    p.retain(&parent).unwrap();
    let mut child = parent.clone();
    let old = child[1];
    let (source, destination) = p.writable_tail(&mut child, 3, 2).unwrap().unwrap();
    assert_eq!(source, old);
    assert_ne!(destination, old);
    assert_eq!(parent[1], old);
    assert_eq!(p.references(old).unwrap(), 1);
    p.check_owners([parent.as_slice(), child.as_slice()].into_iter())
        .unwrap();
    assert_eq!(p.cow_copies, 1);
    assert!(p.writable_tail(&mut parent, 4, 2).unwrap().is_none());
    p.release(&parent).unwrap();
    p.release(&child).unwrap();
    assert_eq!(p.free_blocks(), 3);
    let new = p.allocate(3).unwrap();
    assert!(p.retain(&[old]).is_err());
    p.check_owners(once(new.as_slice())).unwrap();
    p.release(&new).unwrap();
}
#[test]
fn hash_chain_matches_longest_complete_prefix_and_isolates_namespaces() {
    let mut c = PrefixCache::new(b"model-revision-layout", 2, 100, 10).unwrap();
    c.insert(vec![1, 2], 10, 4).unwrap();
    c.insert(vec![1, 2, 3, 4], 20, 8).unwrap();
    assert_eq!(c.lookup(&[1, 2, 3, 4, 5], 4), Some(20));
    assert_eq!(c.lookup(&[1, 2, 8, 9, 5], 4), Some(10));
    assert_eq!(c.lookup(&[9, 8, 3, 4, 5], 4), None);
    assert_eq!(c.lookup(&[1, 2, 3, 4], 3), Some(10));
    let isolated: PrefixCache<u32> = PrefixCache::new(b"other-weights", 2, 100, 10).unwrap();
    assert_eq!(isolated.matched_tokens(&[1, 2, 3, 4], 4), 0);
    assert!(c.insert(vec![1], 30, 4).is_err());
}
#[test]
fn lru_touches_and_byte_eviction_are_bounded() {
    let mut c = PrefixCache::new(b"binding", 2, 8, 10).unwrap();
    c.insert(vec![1, 2], 1, 4).unwrap();
    c.insert(vec![3, 4], 2, 4).unwrap();
    assert_eq!(c.lookup(&[1, 2, 9], 2), Some(1));
    assert_eq!(c.insert(vec![5, 6], 3, 4).unwrap(), vec![2]);
    assert!(c.contains(&[1, 2]));
    assert!(!c.contains(&[3, 4]));
    assert_eq!(c.bytes(), 8);
    assert_eq!(c.evictions, 1);
    assert_eq!(c.insert(vec![7, 8], 4, 9).unwrap(), vec![4]);
    assert_eq!(c.bytes(), 8);
}
#[test]
fn cache_pressure_never_recycles_pages_pinned_by_an_active_sequence() {
    let mut pool = BlockPool::new(4).unwrap();
    let active = pool.allocate(2).unwrap();
    pool.retain(&active).unwrap();
    let mut cache = PrefixCache::new(b"binding", 2, 4, 4).unwrap();
    for evicted in cache.insert(vec![1, 2, 3, 4], active.clone(), 4).unwrap() {
        pool.release(&evicted).unwrap();
    }
    for id in 5..1005u32 {
        let pages = pool.allocate(1).unwrap();
        for evicted in cache.insert(vec![id, id], pages, 4).unwrap() {
            pool.release(&evicted).unwrap();
        }
        pool.check_owners(once(active.as_slice()).chain(cache.values().map(Vec::as_slice)))
            .unwrap();
        assert!(pool.references(active[0]).is_ok());
    }
    while let Some(v) = cache.evict_oldest() {
        pool.release(&v).unwrap();
    }
    pool.release(&active).unwrap();
    assert_eq!(pool.free_blocks(), 4);
}
#[test]
fn logical_pages_grow_on_demand_and_failed_growth_is_atomic() {
    let mut s = SequenceStateManager::new(2, 2).unwrap();
    let id = s
        .reserve_incremental(RequestId::new(1).unwrap(), StateKind::AttentionKv, 8)
        .unwrap();
    assert_eq!(s.snapshot().allocated_pages, 0);
    s.ensure_tokens(id, 3).unwrap();
    s.commit(id, 3).unwrap();
    assert!(s.ensure_tokens(id, 5).is_err());
    assert_eq!(s.get(id).unwrap().pages.len(), 2);
    s.check_invariants().unwrap();
    s.reset(id).unwrap();
    assert_eq!(s.snapshot().free_pages, 2);
    s.ensure_tokens(id, 1).unwrap();
    s.release(id).unwrap();
    assert_eq!(s.snapshot().free_pages, 2);
}

#[test]
fn fixed_page_output_rejects_capacity_and_duplicate_tables_before_reference_changes() -> Result<()>
{
    let mut pool = BlockPool::new(4)?;
    let mut leases = Vec::with_capacity(2);
    let pointer = leases.as_ptr();
    assert_eq!(
        pool.allocate_into(3, &mut leases)
            .err()
            .map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    assert_eq!(pool.free_blocks(), 4);
    assert_eq!(leases, []);
    pool.allocate_into(2, &mut leases)?;
    assert_eq!(leases.as_ptr(), pointer);
    let duplicates = [leases[0], leases[0]];
    assert!(pool.retain(&duplicates).is_err());
    assert!(pool.release(&duplicates).is_err());
    assert_eq!(pool.references(leases[0])?, 1);
    pool.retain(&leases)?;
    assert_eq!(pool.references(leases[0])?, 2);
    pool.release(&leases)?;
    pool.release(&leases)?;
    pool.check_owners(empty())?;
    assert_eq!(pool.free_blocks(), 4);
    Ok(())
}

#[test]
fn independent_physical_pools_reject_matching_foreign_page_generations() -> Result<()> {
    let mut first = BlockPool::new(1)?;
    let mut second = BlockPool::new(1)?;
    let left = first.allocate(1)?;
    let right = second.allocate(1)?;
    assert_eq!(
        (left[0].index, left[0].generation),
        (right[0].index, right[0].generation)
    );
    assert_ne!(left[0].owner, right[0].owner);
    assert!(second.references(left[0]).is_err());
    assert!(second.retain(&left).is_err());
    assert!(second.release(&left).is_err());
    assert_eq!(second.references(right[0])?, 1);
    first.release(&left)?;
    second.release(&right)?;
    Ok(())
}

#[test]
fn cache_churn_matches_lru_oracle_across_hash_iteration_orders() {
    use std::collections::VecDeque;
    let mut cache = PrefixCache::new(b"churn", 2, 28, 7).unwrap();
    let mut order = VecDeque::<(u32, u64)>::new();
    let mut random = 42_u64;
    for i in 0..10_000_u64 {
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let key = u32::try_from(random % 19).unwrap();
        let tokens = [key, key];
        let position = order.iter().position(|(k, _)| *k == key);
        if random & 128 == 0 {
            let expected = position.map(|p| {
                let entry = order.remove(p).unwrap();
                order.push_back(entry);
                entry.1
            });
            assert_eq!(cache.lookup(&tokens, 2), expected);
        } else {
            let mut expected = Vec::new();
            if let Some(p) = position {
                expected.push(order.remove(p).unwrap().1);
            }
            if order.len() == 7 {
                expected.push(order.pop_front().unwrap().1);
            }
            order.push_back((key, i));
            assert_eq!(cache.insert(tokens.to_vec(), i, 4).unwrap(), expected);
        }
        assert_eq!(cache.len(), order.len());
        assert_eq!(cache.bytes(), order.len() as u64 * 4);
        for key in 0..19 {
            assert_eq!(
                cache.contains(&[key, key]),
                order.iter().any(|(k, _)| *k == key)
            );
        }
        let mut values: Vec<_> = cache.values().copied().collect();
        let mut expected: Vec<_> = order.iter().map(|(_, v)| *v).collect();
        values.sort_unstable();
        expected.sort_unstable();
        assert_eq!(values, expected);
    }
    while let Some((_, value)) = order.pop_front() {
        assert_eq!(cache.evict_oldest(), Some(value));
    }
    assert!(cache.evict_oldest().is_none());
}
