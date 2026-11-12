//! Whether a new reservation fits beside what a backend holds, and what to give up if not.
//!
//! The backend reports its accounts and physical memory, then carries out each step; the
//! order in which retained work is given up is engine policy.

/// A backend's accounts at the moment a reservation asks for room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoomFacts {
    /// Bytes the new reservation charges to the admission budget.
    pub needed: u64,
    /// Device scratch the reservation allocates on top, outside the budget.
    pub scratch: u64,
    /// Bytes live sequences hold.
    pub active: u64,
    /// Bytes idle sequences retained for reuse hold.
    pub cached: u64,
    /// Bytes retained prefix snapshots hold.
    pub prefix: u64,
    /// The admission budget all of the above share.
    pub budget: u64,
    /// Device memory free right now, which the reservation and its scratch must fit.
    pub physical: u64,
    /// Live plus retained sequences.
    pub sequences: usize,
    /// Sequences the backend can hold at once.
    pub maximum_sequences: usize,
    /// Whether any idle sequence is retained.
    pub has_cached: bool,
    /// Whether any prefix snapshot is retained.
    pub has_prefix: bool,
}

/// The next step towards room for a reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoomStep {
    /// The reservation fits.
    Fits,
    /// Drop the oldest retained idle sequence, then ask again.
    EvictSequence,
    /// Drop the oldest prefix snapshot, then ask again.
    EvictPrefix,
    /// Nothing retained is left to give up: refuse for capacity.
    Refuse,
}

/// Decide one step. Idle sequences go before prefix snapshots: a snapshot saves a whole
/// prompt, a retained sequence only its graph capture.
#[must_use]
pub fn next_room_step(facts: &RoomFacts) -> RoomStep {
    let within_budget = facts
        .active
        .checked_add(facts.cached)
        .and_then(|charged| charged.checked_add(facts.prefix))
        .and_then(|charged| charged.checked_add(facts.needed))
        .is_some_and(|charged| charged <= facts.budget);
    if within_budget
        && facts.sequences < facts.maximum_sequences
        && facts.needed.saturating_add(facts.scratch) <= facts.physical
    {
        RoomStep::Fits
    } else if facts.has_cached {
        RoomStep::EvictSequence
    } else if facts.has_prefix {
        RoomStep::EvictPrefix
    } else {
        RoomStep::Refuse
    }
}

#[cfg(test)]
#[path = "../tests/unit/room.rs"]
mod tests;
