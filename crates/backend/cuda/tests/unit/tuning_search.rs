use super::*;
use crate::tuning::MIN_REPS;
use std::cell::RefCell;
use std::env::temp_dir;
use std::fs::{remove_file, write};
use std::process::id;

/// Paired times whose candidate runs `speedup` times faster than the baseline.
fn times(speedup: f32) -> PairedTimes {
    let candidate = 1.0 / speedup;
    PairedTimes {
        baseline_ms: vec![1.0; MIN_REPS],
        candidate_ms: vec![candidate; MIN_REPS],
        baseline_median_ms: 1.0,
        candidate_median_ms: candidate,
        speedup: 1.0 / candidate,
    }
}

fn measured(choice: u8, order: MeasurementOrder, speedup: f32) -> Trial<u8> {
    Trial {
        choice,
        order,
        cold: times(speedup),
        warm: times(speedup),
    }
}

/// Baseline 0 against candidates 1, 2 and 3.
fn protocol() -> Protocol<u8> {
    Protocol {
        baseline: 0,
        candidates: vec![1, 2, 3],
    }
}

/// Run the protocol with fixed speedups per choice, baseline-first and candidate-first; a
/// missing speedup is a rejected candidate.
fn search(
    baseline_first: &[(u8, f32)],
    candidate_first: &[(u8, f32)],
) -> (Option<Evidence<u8>>, Vec<u8>) {
    let rejects = RefCell::new(Vec::new());
    let evidence = run(
        &protocol(),
        || true,
        |choice, order| {
            let table = if order == MeasurementOrder::BaselineFirst {
                baseline_first
            } else {
                candidate_first
            };
            table
                .iter()
                .find(|(listed, _)| *listed == choice)
                .map(|&(_, speedup)| measured(choice, order, speedup))
                .ok_or_else(|| Error::invalid("output mismatch"))
        },
        |choice, _| {
            rejects.borrow_mut().push(choice);
            Ok(())
        },
    )
    .unwrap();
    (evidence, rejects.into_inner())
}

#[test]
fn the_best_candidate_confirmed_in_both_orders_wins() {
    let (evidence, rejects) = search(&[(1, 1.1), (2, 1.3), (3, 1.01)], &[(2, 1.25)]);
    let evidence = evidence.unwrap();
    assert_eq!(evidence.selected, 2);
    assert_eq!(
        evidence.confirmation.order,
        MeasurementOrder::CandidateFirst
    );
    assert_eq!(rejects, Vec::<u8>::new());
    assert!(evidence.valid(&protocol(), |_| true));
}

#[test]
fn a_winner_that_does_not_confirm_falls_back_to_the_baseline() {
    let (evidence, _) = search(&[(1, 1.5), (2, 1.0), (3, 1.0)], &[(1, 1.0), (0, 1.0)]);
    let evidence = evidence.unwrap();
    assert_eq!(evidence.selected, 0);
    assert_eq!(evidence.confirmation.choice, 0);
    assert!(evidence.valid(&protocol(), |_| true));
}

#[test]
fn rejected_candidates_are_handled_and_recorded_once() {
    let (evidence, rejects) = search(&[(1, 1.2), (3, 1.0)], &[(1, 1.2)]);
    let evidence = evidence.unwrap();
    assert_eq!(rejects, [2]);
    let reason = Error::invalid("output mismatch").to_string();
    assert_eq!(evidence.rejected, [(2, reason)]);
    assert_eq!(evidence.selected, 1);
    assert!(evidence.valid(&protocol(), |_| true));
}

#[test]
fn a_stopped_search_records_nothing() {
    let trials = RefCell::new(0);
    let evidence = run(
        &protocol(),
        || *trials.borrow() < 2,
        |choice, order| {
            *trials.borrow_mut() += 1;
            Ok(measured(choice, order, 1.5))
        },
        |_, _| Ok(()),
    )
    .unwrap();
    assert!(evidence.is_none());
}

#[test]
fn incomplete_or_disallowed_evidence_is_invalid() {
    let (evidence, _) = search(&[(1, 1.1), (2, 1.3), (3, 1.01)], &[(2, 1.25)]);
    let complete = evidence.unwrap();
    assert!(!complete.valid(&protocol(), |choice| choice != 2));
    let mut missing = complete.clone();
    missing.trials.pop();
    assert!(!missing.valid(&protocol(), |_| true));
    let mut unordered = complete.clone();
    unordered.trials[0].order = MeasurementOrder::CandidateFirst;
    assert!(!unordered.valid(&protocol(), |_| true));
    let mut unconfirmed = complete;
    unconfirmed.confirmation = measured(2, MeasurementOrder::CandidateFirst, 1.0);
    assert!(!unconfirmed.valid(&protocol(), |_| true));
}

#[test]
fn stored_records_replace_their_key_and_keep_what_this_build_cannot_read() {
    let path = temp_dir().join(format!("infer-tuning-search-{}.json", id()));
    let (evidence, _) = search(&[(1, 1.1), (2, 1.3), (3, 1.01)], &[(2, 1.25)]);
    let record = Record {
        key: "shape-a".to_owned(),
        evidence: evidence.unwrap(),
    };
    let mut table = TilingTable::default();
    table
        .records
        .entry("family".into())
        .or_default()
        .push(serde_json::json!({ "written": "by a newer build" }));
    write(&path, serde_json::to_string(&table).unwrap()).unwrap();
    assert!(store(Some(&path), "family", &record));
    let mut replaced = record.clone();
    replaced.evidence.selected = 0;
    assert!(store(Some(&path), "family", &replaced));
    let table = TilingTable::load(&path).unwrap();
    remove_file(&path).unwrap();
    assert_eq!(table.records["family"].len(), 2);
    let found = lookup::<String, u8>(&table, "family", &record.key, |_| true).unwrap();
    assert_eq!(found.evidence.selected, 0);
    assert!(lookup::<String, u8>(&table, "family", &"shape-b".into(), |_| true).is_none());
    assert!(lookup::<String, u8>(&table, "family", &record.key, |_| false).is_none());
}
