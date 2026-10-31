//! Whether a reservation keeps the serving pool, waits for it, or funds a new one.
//!
//! The decision belongs to the engine; a backend supplies physical facts and carries out the
//! answer. Pricing a pool stays with the backend, because that is a device fact.
use infer_core::StateId;

/// The physical pool a backend currently holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolFacts {
    /// Tokens one slot reaches.
    pub capacity: usize,
    /// Concurrent slots.
    pub width: usize,
    /// KV pages the pool owns.
    pub pages: usize,
}

/// Live sequences as a pool decision sees them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveStates {
    /// No sequence is live.
    Empty,
    /// One private sequence that a pool could serve, with its reserved capacity.
    Solo { capacity: usize },
    /// Anything else: these keep their private programs.
    Other,
}

/// Everything the decision depends on, after the backend has reached an idle boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolDemand {
    pub state: StateId,
    pub tokens: usize,
    /// Tokens per page of the backend's KV layout.
    pub page_tokens: usize,
    /// A separate draft model has no private proposal graph, so it always needs the pool.
    pub requires_pool: bool,
    pub live: LiveStates,
    pub pool: Option<PoolFacts>,
    /// The pool has no leased slot and no live state, so it may be replaced.
    pub idle: bool,
}

/// What the backend should do for this reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolAction {
    /// Serve with what exists: the current pool, or the request's private program.
    Keep,
    /// Refuse for now; the request retries once the solo sequence drains.
    Wait,
    /// Replace the pool with one funded for this many tokens.
    Fund { demand: usize },
}

/// Longest request observed waiting for a pool, kept across reservations.
pub type PoolRetry = Option<(StateId, usize)>;

/// Decide one reservation; `retry` carries the waiting request between reservations.
#[must_use]
pub fn decide_pool(demand: &PoolDemand, retry: &mut PoolRetry) -> PoolAction {
    let tokens = demand.tokens;
    let needed =
        if demand.pool.is_some() || (demand.requires_pool && demand.live == LiveStates::Empty) {
            tokens
        } else {
            match demand.live {
                LiveStates::Empty => match retry.take() {
                    Some((waiting, longest)) if waiting == demand.state => longest.max(tokens),
                    _ => return PoolAction::Keep,
                },
                LiveStates::Solo { capacity } => {
                    let longest = capacity.max(tokens);
                    let waiting = retry.get_or_insert((demand.state, longest));
                    waiting.1 = waiting.1.max(longest);
                    return PoolAction::Wait;
                }
                LiveStates::Other => return PoolAction::Keep,
            }
        };
    if !demand.idle {
        return PoolAction::Keep;
    }
    // A shorter request does not justify rebuilding a working pool.
    if demand.pool.is_some_and(|pool| {
        tokens <= pool.capacity
            && demand.page_tokens > 0
            && pool
                .width
                .saturating_mul(tokens.div_ceil(demand.page_tokens))
                <= pool.pages
    }) {
        return PoolAction::Keep;
    }
    PoolAction::Fund { demand: needed }
}

#[cfg(test)]
#[path = "../tests/unit/pool_demand.rs"]
mod tests;
