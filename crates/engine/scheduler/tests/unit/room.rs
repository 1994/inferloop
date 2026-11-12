use super::*;

const GIB: u64 = 1 << 30;

fn facts() -> RoomFacts {
    RoomFacts {
        needed: GIB,
        scratch: 0,
        active: 2 * GIB,
        cached: GIB,
        prefix: GIB,
        budget: 8 * GIB,
        physical: 4 * GIB,
        sequences: 3,
        maximum_sequences: 16,
        has_cached: true,
        has_prefix: true,
    }
}

#[test]
fn a_reservation_within_every_limit_fits() {
    assert_eq!(next_room_step(&facts()), RoomStep::Fits);
}

#[test]
fn each_limit_alone_asks_for_room() {
    let over_budget = RoomFacts {
        needed: 5 * GIB,
        physical: 8 * GIB,
        ..facts()
    };
    let at_sequence_limit = RoomFacts {
        sequences: 16,
        ..facts()
    };
    let scratch_over_physical = RoomFacts {
        scratch: 3 * GIB + 1,
        ..facts()
    };
    for case in [over_budget, at_sequence_limit, scratch_over_physical] {
        assert_eq!(next_room_step(&case), RoomStep::EvictSequence, "{case:?}");
    }
}

#[test]
fn idle_sequences_go_before_prefix_snapshots_and_then_admission_refuses() {
    let full = RoomFacts {
        sequences: 16,
        ..facts()
    };
    assert_eq!(next_room_step(&full), RoomStep::EvictSequence);
    let no_cached = RoomFacts {
        has_cached: false,
        ..full
    };
    assert_eq!(next_room_step(&no_cached), RoomStep::EvictPrefix);
    let nothing_left = RoomFacts {
        has_prefix: false,
        ..no_cached
    };
    assert_eq!(next_room_step(&nothing_left), RoomStep::Refuse);
}

#[test]
fn an_overflowing_account_never_fits() {
    let overflow = RoomFacts {
        active: u64::MAX,
        has_cached: false,
        has_prefix: false,
        ..facts()
    };
    assert_eq!(next_room_step(&overflow), RoomStep::Refuse);
}
