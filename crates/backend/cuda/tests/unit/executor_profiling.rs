use super::{capture_outcome_record, memory_increment};

#[test]
fn physical_stage_deltas_reset_at_each_unpublished_capture() {
    let before = memory_increment("before_pool", None, 1_000);
    assert_eq!(before, 0);
    assert_eq!(memory_increment("target_pool", Some(1_000), 600), 400);
    assert_eq!(memory_increment("draft_pool", Some(600), 550), 50);
    assert_eq!(memory_increment("worker", Some(550), 700), 0);
    assert_eq!(memory_increment("before_pool", Some(700), 900), 0);
    assert_eq!(memory_increment("target_pool", Some(900), 850), 50);
}

#[test]
fn rejected_capture_preserves_unquoted_cost_and_replan_funding() {
    let record = capture_outcome_record(1, "rejected", 100, Some(125), 110, Some(80), None);
    assert_eq!(record["capture_attempt"], 1);
    assert_eq!(record["pool_capture_outcome"], "rejected");
    assert_eq!(record["quoted_bytes"], 100);
    assert_eq!(record["allocated_bytes"], 125);
    assert_eq!(record["unquoted_bytes"], 25);
    assert_eq!(record["allocated_over_envelope_bytes"], 15);
    assert_eq!(record["next_quote_budget"], 80);
}

#[test]
fn failed_capture_without_an_allocation_measurement_keeps_the_error() {
    let record = capture_outcome_record(0, "rejected", 100, None, 110, None, Some("OOM"));
    assert_eq!(record["allocated_bytes"], serde_json::Value::Null);
    assert_eq!(record["unquoted_bytes"], serde_json::Value::Null);
    assert_eq!(record["next_quote_budget"], serde_json::Value::Null);
    assert_eq!(record["error"], "OOM");
}
