use super::*;
use infer_ir::{OpClass, TimedGraph, TimedInterval, TimedPhase, TimedSegment};

fn sample(rows: u32) -> DeviceTimingSample {
    DeviceTimingSample {
        rows,
        intervals: vec![
            TimedInterval {
                phase: TimedPhase::Draft,
                start_ns: 0,
                duration_ns: 4_000,
            },
            TimedInterval {
                phase: TimedPhase::Verify,
                start_ns: 4_000,
                duration_ns: 30_000,
            },
        ],
        segments: vec![TimedSegment {
            phase: TimedPhase::Verify,
            graph: TimedGraph::Target,
            node: Some(9),
            op: OpClass::Attention,
            layer: Some(3),
            variant: Some(0),
            start_ns: 5_000,
            duration_ns: 2_000,
        }],
        host: Vec::new(),
    }
}

#[test]
fn records_keep_the_backend_timeline_and_its_runtime_placement() {
    let mut timelines = DeviceTimelines::new(vec!["fa4".into()]);
    timelines.push(7, 1, 1_234, sample(16));
    let query = timelines.query(0, 4).unwrap();
    assert_eq!(query.variants, ["fa4"]);
    assert_eq!(query.evicted, 0);
    let [record] = query.records.as_slice() else {
        panic!("one record");
    };
    assert_eq!(
        (record.step, record.program, record.completed_us),
        (7, 1, 1_234)
    );
    assert_eq!(record.sample, sample(16));
    assert_eq!(record.sample.total_ns(), 34_000);
    assert_eq!(query.next, record.sequence);
}

#[test]
fn a_cursor_pages_through_retained_records_without_consuming_them() {
    let mut timelines = DeviceTimelines::new(Vec::new());
    for step in 0..5 {
        timelines.push(step, 1, step * 10, sample(1));
    }
    let first = timelines.query(0, 2).unwrap();
    assert_eq!(
        first.records.iter().map(|r| r.step).collect::<Vec<_>>(),
        [0, 1]
    );
    let second = timelines.query(first.next, 2).unwrap();
    assert_eq!(
        second.records.iter().map(|r| r.step).collect::<Vec<_>>(),
        [2, 3]
    );
    let again = timelines.query(0, 2).unwrap();
    assert_eq!(again, first);
    let end = timelines.query(second.next, 4).unwrap();
    assert_eq!(end.records.len(), 1);
    assert_eq!(timelines.query(end.next, 4).unwrap().records.len(), 0);
}

#[test]
fn retention_is_bounded_and_counts_evictions() {
    let mut timelines = DeviceTimelines::new(Vec::new());
    let total = DEVICE_TIMELINE_CAPACITY + 3;
    for step in 0..total {
        timelines.push(u64::try_from(step).unwrap(), 1, 0, sample(1));
    }
    let query = timelines.query(0, MAX_DEVICE_TIMELINE_QUERY).unwrap();
    assert_eq!(query.records.len(), DEVICE_TIMELINE_CAPACITY);
    assert_eq!(query.evicted, 3);
    assert_eq!(query.records[0].step, 3);
    assert!(timelines.query(0, 0).is_err());
    assert!(timelines.query(0, MAX_DEVICE_TIMELINE_QUERY + 1).is_err());
}
