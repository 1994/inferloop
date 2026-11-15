use super::*;
use infer_core::event::event_ring;
use infer_core::{RequestId, Result, event::EventKind, event::ObjectKind, event::SemanticEvent};
use serde_json::json;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;

const fn event(kind: EventKind, id: u64, clock: u64, arg0: u64, arg1: u64) -> SemanticEvent {
    SemanticEvent {
        timestamp_us: clock,
        kind,
        object_kind: ObjectKind::Request,
        reserved: 0,
        object_id: id,
        correlation_id: 0,
        arg0,
        arg1,
    }
}

#[test]
fn histograms_track_boundaries_overflow_and_nearest_rank() {
    let mut histogram = LatencyHistogram::default();
    for value in [0, 10, 11, 50, 5_000_001] {
        histogram.record(value);
    }
    assert_eq!(histogram.buckets[0], 2);
    assert_eq!(histogram.buckets[1], 2);
    assert_eq!(histogram.buckets[16], 1);
    assert_eq!(histogram.percentile_upper_bound(50), Some(50));
    assert_eq!(histogram.percentile_upper_bound(99), Some(5_000_001));
    assert_eq!(histogram.percentile_upper_bound(0), None);
    assert_eq!(LatencyHistogram::default().percentile_upper_bound(99), None);
    histogram.count = u64::MAX;
    histogram.buckets = [0; 17];
    histogram.buckets[16] = u64::MAX;
    assert_eq!(histogram.percentile_upper_bound(100), Some(5_000_001));
}
#[test]
fn filtered_cursor_discloses_eviction_and_remains_non_destructive() {
    let mut store = ObservationStore::new(2).unwrap();
    store.record(event(EventKind::Accepted, 1, 0, 0, 0));
    store.record(event(EventKind::Accepted, 2, 1, 0, 0));
    store.record(event(EventKind::Finished, 1, 2, 0, 2));
    let first = store.query(0, 1, None, 3).unwrap();
    assert!(first.cursor_gap);
    assert_eq!(first.oldest_cursor, 2);
    assert_eq!(first.history_evicted, 1);
    assert_eq!(first.ring_dropped, 3);
    let filtered = store
        .query(first.next_cursor, 2, Some(RequestId::new(1).unwrap()), 3)
        .unwrap();
    assert_eq!(filtered.events.len(), 1);
    assert_eq!(filtered.next_cursor, 3);
    assert!(store.query(3, 1, None, 3).unwrap().events.is_empty());
    assert_eq!(store.query(0, 2, None, 3).unwrap().events.len(), 2);
    assert!(store.query(0, 4097, None, 0).is_err());
    let mut cursor = 0;
    assert_eq!(store.legacy_events(&mut cursor).len(), 2);
    assert_eq!(store.legacy_events(&mut cursor), [] as [SemanticEvent; 0]);
}
#[test]
fn metric_collection_survives_trace_ring_overflow() {
    let mut metrics = RuntimeMetrics::default();
    let (mut writer, mut reader) = event_ring(1);
    for record in [
        event(EventKind::Accepted, 1, 0, 0, 0),
        event(EventKind::FirstToken, 1, 5, 5, 0),
        event(EventKind::Finished, 1, 8, 0, 8),
    ] {
        metrics.record(&record);
        writer.emit(record);
    }
    assert_eq!(writer.dropped(), 2);
    assert_eq!(metrics.successful_requests, 1);
    assert_eq!(metrics.e2e.sum_us, 8);
    assert_eq!(metrics.ttft.sum_us, 5);
    assert_eq!(reader.pop().unwrap().kind, EventKind::Accepted);
    assert!(reader.pop().is_err());
    let mut deferred = event(EventKind::Deferred, 2, 9, 1, 0);
    deferred.reserved = infer_ir::DeferReason::PreemptionFocus {
        request: RequestId::new(1).unwrap(),
    }
    .code();
    metrics.record(&deferred);
    assert_eq!(metrics.deferred_reasons[11], 1);
    deferred.reserved = u32::MAX;
    metrics.record(&deferred);
    assert_eq!(metrics.deferred_reasons[11], 1);
}
#[test]
fn traceparent_validates_version_identifiers_and_flags() {
    let valid = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    assert!(trace::TraceContext::parse(valid).unwrap().sampled);
    for invalid in [
        valid.replace("00-", "ff-"),
        valid.to_uppercase(),
        valid.replace("4bf92f3577b34da6a3ce929d0e0e4736", &"0".repeat(32)),
        valid.replace("00f067aa0ba902b7", &"0".repeat(16)),
        format!("{valid}-extra"),
    ] {
        assert!(trace::TraceContext::parse(&invalid).is_err());
    }
}
#[test]
fn otlp_contains_complete_spans_with_parent_and_unix_nanosecond_strings() {
    let mut store = ObservationStore::new(16).unwrap();
    for record in [
        event(EventKind::Accepted, 1, 7, 0, 0),
        event(EventKind::Finished, 1, 12, 5, 5),
        event(EventKind::Accepted, 2, 13, 0, 0),
    ] {
        store.record(record);
    }
    let parent =
        trace::TraceContext::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00")
            .unwrap();
    let result = trace::otlp(
        &store.timeline(),
        &BTreeMap::from([(1, parent.clone())]),
        9,
        1_000_000,
    );
    assert_eq!(
        result.as_object().unwrap().keys().collect::<Vec<_>>(),
        vec!["resourceSpans"]
    );
    let spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span["traceId"], parent.trace_id);
    assert_eq!(span["parentSpanId"], parent.parent_span_id);
    assert_eq!(span["flags"], 0);
    assert_eq!(span["status"]["code"], 2);
    assert_eq!(span["startTimeUnixNano"], "1000007000");
    assert_eq!(span["endTimeUnixNano"], "1000012000");
    assert_eq!(span["spanId"].as_str().unwrap().len(), 16);
    assert_ne!(
        span["spanId"],
        trace::otlp(&store.timeline(), &BTreeMap::new(), 10, 1_000_000)["resourceSpans"][0]["scopeSpans"]
            [0]["spans"][0]["spanId"]
    );
    assert_eq!(
        trace::timeline(&store.timeline(), 2, 3)["clock"],
        json!("runtime_logical")
    );
}
#[test]
fn prometheus_histograms_are_cumulative_and_labels_have_bounded_cardinality() {
    let mut metrics = RuntimeMetrics::default();
    metrics.e2e.record(10);
    metrics.e2e.record(20);
    let gauges = export::ResourceGauges {
        backend: "metal".into(),
        ready: true,
        active_requests: 0,
        inflight_steps: 0,
        resource_waiters: 0,
        output_waiters: 0,
        output_batches: 0,
        logical_pages: 0,
        free_logical_pages: 2,
        active_kv_blocks: 0,
        cached_kv_blocks: 1,
        available_kv_blocks: 2,
        kv_pool_bytes: 128,
        event_ring_dropped: 4,
        collector_metadata_dropped: 0,
    };
    let output = export::prometheus(&metrics, &gauges, &ObservationStore::new(2).unwrap());
    assert!(
        output.contains("infer_request_e2e_seconds_bucket{backend=\"metal\",le=\"0.00001\"} 1\n")
    );
    assert!(output.contains("infer_request_e2e_seconds_bucket{backend=\"metal\",le=\"+Inf\"} 2\n"));
    assert!(output.contains("infer_request_e2e_seconds_count{backend=\"metal\"} 2\n"));
    assert!(output.ends_with('\n'));
    assert!(!output.contains("request_id="));
    assert!(!output.contains("tenant="));
}

/// The finish event carries its cancel source or error code, so the per-cause counters and their
/// exposition agree with the terminal log line; an unknown cause is counted nowhere.
#[test]
fn finish_causes_count_by_cancel_source_and_error_code() {
    use infer_core::{CancelSource, ErrorCode, FinishReason};
    let mut metrics = RuntimeMetrics::default();
    let finished = |reason: &FinishReason, cancel, error| {
        let mut record = event(EventKind::Finished, 1, 0, finish_code(reason), 0);
        record.reserved = finish_cause(reason, cancel, error);
        record
    };
    metrics.record(&finished(
        &FinishReason::Cancelled,
        Some(CancelSource::Shutdown),
        None,
    ));
    metrics.record(&finished(
        &FinishReason::Failed("backend".into()),
        None,
        Some(ErrorCode::Backend),
    ));
    metrics.record(&finished(&FinishReason::Cancelled, None, None));
    // A code on a successful finish is not a cause.
    metrics.record(&finished(
        &FinishReason::Eos,
        None,
        Some(ErrorCode::Backend),
    ));
    assert_eq!(metrics.cancelled_by[CancelSource::Shutdown.index()], 1);
    assert_eq!(metrics.cancelled_by.iter().sum::<u64>(), 1);
    assert_eq!(metrics.failed_by[ErrorCode::Backend.index()], 1);
    assert_eq!(metrics.failed_by.iter().sum::<u64>(), 1);
    let gauges = export::ResourceGauges {
        backend: "metal".into(),
        ready: true,
        active_requests: 0,
        inflight_steps: 0,
        resource_waiters: 0,
        output_waiters: 0,
        output_batches: 0,
        logical_pages: 0,
        free_logical_pages: 0,
        active_kv_blocks: 0,
        cached_kv_blocks: 0,
        available_kv_blocks: 0,
        kv_pool_bytes: 0,
        event_ring_dropped: 0,
        collector_metadata_dropped: 0,
    };
    let output = export::prometheus(&metrics, &gauges, &ObservationStore::new(2).unwrap());
    assert!(
        output.contains("infer_requests_cancelled_total{backend=\"metal\",cause=\"shutdown\"} 1\n")
    );
    assert!(output.contains(
        "infer_requests_cancelled_total{backend=\"metal\",cause=\"client_disconnected\"} 0\n"
    ));
    assert!(
        output.contains("infer_request_failures_total{backend=\"metal\",code=\"backend\"} 1\n")
    );
}

#[test]
fn shared_window_survives_tail_mutation_and_multiple_page_evictions() -> Result<()> {
    for capacity in [1, 2, 127, 128, 129, 257] {
        let mut store = ObservationStore::new(capacity)?;
        let mut oracle = VecDeque::new();
        for sequence in 1..=512_u64 {
            let record = event(EventKind::Accepted, sequence % 7 + 1, sequence, 0, 0);
            store.record(record);
            oracle.push_back(record);
            if oracle.len() > capacity {
                oracle.pop_front();
            }
        }
        let window = store.snapshot(true);
        let shared = store.snapshot(true);
        assert!(Arc::ptr_eq(
            &window.events.pages[0],
            &shared.events.pages[0]
        ));
        let expected: Vec<_> = oracle.iter().copied().collect();
        for sequence in 513..=2049_u64 {
            let record = event(EventKind::Finished, sequence % 7 + 1, sequence, 0, 0);
            store.record(record);
            oracle.push_back(record);
            if oracle.len() > capacity {
                oracle.pop_front();
            }
            if sequence % 73 != 0 {
                continue;
            }
            for request in 1..=7 {
                assert_eq!(
                    store.contains_request(request),
                    oracle.iter().any(|event| event.object_id == request)
                );
            }
            assert_eq!(
                store
                    .timeline()
                    .iter()
                    .map(|event| event.event)
                    .collect::<Vec<_>>(),
                oracle.iter().copied().collect::<Vec<_>>()
            );
            assert_eq!(
                window
                    .timeline()
                    .iter()
                    .map(|event| event.event)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        let live = store.query(0, 4096, None, 3)?;
        assert_eq!(live.history_evicted, 2049 - capacity as u64);
        assert_eq!(live.oldest_cursor, 2050 - capacity as u64);
        assert_eq!(live.next_cursor, 2049);
        let old = window.query(0, 4096, None, 3)?;
        assert_eq!(old.history_evicted, 512 - capacity as u64);
        assert_eq!(old.next_cursor, 512);
        assert!(!store.contains_request(99));
    }
    Ok(())
}

#[test]
fn collector_snapshot_excludes_future_events_from_shared_tail_and_query_cursors() -> Result<()> {
    let mut store = ObservationStore::new(8)?;
    for id in 1..=7 {
        store.record(event(EventKind::Progress, id, id, 0, 0));
    }
    let frozen = store.snapshot(true).as_of(4, true);
    assert_eq!(frozen.retained(), 4);
    assert_eq!(frozen.timeline().len(), 4);
    let query = frozen.query(0, 8, None, 0)?;
    assert_eq!(query.next_cursor, 4);
    assert!(query.events.iter().all(|event| event.sequence <= 4));
    assert_eq!(frozen.query(4, 8, None, 0)?.next_cursor, 4);
    store.record(event(EventKind::Progress, 8, 8, 0, 0));
    assert_eq!(frozen.timeline().len(), 4);
    Ok(())
}
