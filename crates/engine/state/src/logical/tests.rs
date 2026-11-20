use infer_core::ErrorCode;

use super::*;
fn req(id: u64) -> RequestId {
    RequestId::new(id).unwrap()
}
#[test]
fn cache_miss_does_not_claim_uninitialized_tokens() {
    let mut state = SequenceStateManager::new(4, 4).unwrap();
    let id = state
        .reserve(req(1), StateKind::AttentionKv, 8, Some(&prefix()))
        .unwrap();
    assert_eq!(state.get(id).unwrap().committed_tokens, 0);
    state.check_invariants().unwrap();
}
fn prefix() -> PrefixKey {
    PrefixKey {
        model: ModelId::new(1).unwrap(),
        revision: "a".into(),
        precision: "bf16".into(),
        layout: "kv-v1".into(),
        tokens: vec![1, 2, 3, 4],
    }
}
#[test]
fn failed_reservation_is_atomic_and_release_is_exact() {
    let mut m = SequenceStateManager::new(2, 4).unwrap();
    let s = m.reserve(req(1), StateKind::AttentionKv, 8, None).unwrap();
    assert_eq!(
        m.reserve(req(2), StateKind::LinearAttention, 1, None)
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    m.check_invariants().unwrap();
    assert_eq!(m.snapshot().sequence_count, 1);
    m.release(s).unwrap();
    assert!(m.release(s).is_err());
    assert_eq!(m.snapshot().free_pages, 2);
}
#[test]
fn shared_prefix_survives_owner_and_cache_release() {
    let mut m = SequenceStateManager::new(3, 4).unwrap();
    let s = m.reserve(req(1), StateKind::AttentionKv, 8, None).unwrap();
    m.commit(s, 4).unwrap();
    m.cache_prefix(s, prefix()).unwrap();
    let t = m
        .reserve(req(2), StateKind::AttentionKv, 8, Some(&prefix()))
        .unwrap();
    m.check_invariants().unwrap();
    m.release(s).unwrap();
    assert!(m.evict_prefix(&prefix()));
    m.check_invariants().unwrap();
    assert_eq!(m.get(t).unwrap().committed_tokens, 4);
    m.release(t).unwrap();
    m.check_invariants().unwrap();
    assert_eq!(m.snapshot().allocated_pages, 0);
}
#[test]
fn long_reserve_cancel_sequence_has_no_leak() {
    let mut m = SequenceStateManager::new(256, 8).unwrap();
    let mut live = Vec::new();
    for id in 1..1000 {
        if id % 3 == 0 && !live.is_empty() {
            m.release(live.remove(0)).unwrap();
        }
        if let Ok(s) = m.reserve(
            req(id),
            StateKind::Conv,
            (usize::try_from(id % 19).unwrap_or_default()) + 1,
            None,
        ) {
            live.push(s);
        }
        m.check_invariants().unwrap();
    }
    for s in live {
        m.release(s).unwrap();
    }
    assert_eq!(m.snapshot().allocated_pages, 0);
}
#[test]
fn batch_growth_validates_all_states_before_committing_pages_or_ids() {
    let mut state = SequenceStateManager::new(3, 4).unwrap();
    let first = state
        .reserve_incremental(req(1), StateKind::AttentionKv, 12)
        .unwrap();
    let second = state
        .reserve_incremental(req(2), StateKind::AttentionKv, 12)
        .unwrap();
    let before = serde_json::to_value(&state).unwrap();
    assert_eq!(
        state
            .ensure_batch(&[(first, 8), (second, 8)])
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    assert_eq!(serde_json::to_value(&state).unwrap(), before);
    assert!(state.ensure_batch(&[(first, 4), (second, 13)]).is_err());
    assert_eq!(serde_json::to_value(&state).unwrap(), before);
    assert!(state.ensure_batch(&[(first, 4), (first, 4)]).is_err());
    assert_eq!(serde_json::to_value(&state).unwrap(), before);
    state.ensure_batch(&[(first, 8), (second, 4)]).unwrap();
    assert_eq!(state.get(first).unwrap().pages.len(), 2);
    assert_eq!(state.get(second).unwrap().pages.len(), 1);
    state.check_invariants().unwrap();
    state.release(first).unwrap();
    state.release(second).unwrap();
    assert_eq!(state.snapshot().allocated_pages, 0);
}
