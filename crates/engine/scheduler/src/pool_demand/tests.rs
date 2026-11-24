use super::*;

const PAGE: usize = 32;

fn id(value: u64) -> StateId {
    StateId::new(value).unwrap()
}

fn demand(tokens: usize, live: LiveStates, pool: Option<PoolFacts>) -> PoolDemand {
    PoolDemand {
        state: id(1),
        tokens,
        page_tokens: PAGE,
        requires_pool: false,
        live,
        pool,
        idle: true,
    }
}

const POOL: PoolFacts = PoolFacts {
    capacity: 8192,
    width: 4,
    pages: 64,
};

#[test]
fn a_first_solo_request_uses_its_private_program() {
    let mut retry = None;
    let action = decide_pool(&demand(100, LiveStates::Empty, None), &mut retry);
    assert_eq!(action, PoolAction::Keep);
    assert_eq!(retry, None);
}

#[test]
fn a_concurrent_request_waits_and_then_funds_the_longest_demand() {
    let mut retry = None;
    let waiting = demand(100, LiveStates::Solo { capacity: 500 }, None);
    assert_eq!(decide_pool(&waiting, &mut retry), PoolAction::Wait);
    assert_eq!(retry, Some((id(1), 500)));
    let drained = demand(100, LiveStates::Empty, None);
    assert_eq!(
        decide_pool(&drained, &mut retry),
        PoolAction::Fund { demand: 500 }
    );
    assert_eq!(retry, None);
}

#[test]
fn another_request_arriving_after_the_drain_forgets_the_retry() {
    let mut retry = Some((id(9), 500));
    let action = decide_pool(&demand(100, LiveStates::Empty, None), &mut retry);
    assert_eq!(action, PoolAction::Keep);
    assert_eq!(retry, None);
}

#[test]
fn incompatible_private_states_keep_their_programs() {
    let mut retry = None;
    let action = decide_pool(&demand(100, LiveStates::Other, None), &mut retry);
    assert_eq!(action, PoolAction::Keep);
}

#[test]
fn a_separate_draft_funds_the_pool_for_its_first_request() {
    let mut retry = None;
    let mut first = demand(100, LiveStates::Empty, None);
    first.requires_pool = true;
    assert_eq!(
        decide_pool(&first, &mut retry),
        PoolAction::Fund { demand: 100 }
    );
}

#[test]
fn a_pool_with_enough_pages_is_reused() {
    let mut retry = None;
    let fits = demand(512, LiveStates::Other, Some(POOL));
    assert_eq!(decide_pool(&fits, &mut retry), PoolAction::Keep);
}

#[test]
fn a_pool_short_of_pages_or_capacity_is_replaced_only_when_idle() {
    let mut retry = None;
    let short = demand(1024, LiveStates::Empty, Some(POOL));
    assert_eq!(
        decide_pool(&short, &mut retry),
        PoolAction::Fund { demand: 1024 }
    );
    let beyond = demand(9000, LiveStates::Empty, Some(POOL));
    assert_eq!(
        decide_pool(&beyond, &mut retry),
        PoolAction::Fund { demand: 9000 }
    );
    let mut busy = short;
    busy.idle = false;
    assert_eq!(decide_pool(&busy, &mut retry), PoolAction::Keep);
}
