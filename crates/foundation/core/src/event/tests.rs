use super::*;
#[test]
fn bounded_ring_reports_loss() {
    assert_eq!(size_of::<SemanticEvent>(), 48);
    let (mut writer, mut reader) = event_ring(1);
    let e = SemanticEvent {
        timestamp_us: 0,
        kind: EventKind::Accepted,
        object_kind: ObjectKind::Request,
        reserved: 0,
        object_id: 1,
        correlation_id: 0,
        arg0: 0,
        arg1: 0,
    };
    writer.emit(e);
    writer.emit(e);
    assert_eq!(writer.dropped(), 1);
    assert_eq!(reader.pop().unwrap(), e);
}
