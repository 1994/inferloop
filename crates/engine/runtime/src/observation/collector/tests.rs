use super::*;
use infer_core::event::{EventKind, ObjectKind, SemanticEvent, event_ring};
fn event(id: u64) -> SemanticEvent {
    SemanticEvent {
        timestamp_us: id,
        kind: EventKind::Progress,
        object_kind: ObjectKind::Request,
        reserved: 0,
        object_id: id,
        correlation_id: 0,
        arg0: 0,
        arg1: 0,
    }
}
fn parent() -> Result<TraceContext> {
    TraceContext::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
}
#[test]
fn active_parent_survives_eviction_and_retires_without_metadata_leaks() -> Result<()> {
    let (mut writer, reader) = event_ring(8);
    let mut collector = Collector::new(reader, ObservationStore::new(2)?, 0, 2)?;
    writer.emit(event(1));
    assert!(collector.parent(1, parent()?));
    let snapshot = collector
        .snapshot(writer.published(), true, true)?
        .finish()?;
    assert_eq!(snapshot.parents.get(&1), Some(&parent()?));
    writer.emit(event(2));
    writer.emit(event(3));
    assert!(
        collector
            .snapshot(writer.published(), true, true)?
            .finish()?
            .parents
            .contains_key(&1)
    );
    collector.retire(1, writer.published());
    assert!(
        !collector
            .snapshot(writer.published(), true, true)?
            .finish()?
            .parents
            .contains_key(&1)
    );
    assert_eq!(collector.shared.parents.load(Ordering::Acquire), 0);
    assert!(collector.parent(4, parent()?));
    assert!(collector.parent(5, parent()?));
    assert!(!collector.parent(6, parent()?));
    let snapshot = collector
        .snapshot(writer.published(), false, true)?
        .finish()?;
    assert_eq!(snapshot.parents.len(), 2);
    assert_eq!(snapshot.dropped_metadata, 1);
    assert_eq!(snapshot.window.timeline().len(), 0);
    assert_eq!(snapshot.window.retained(), 2);
    Ok(())
}
#[test]
fn snapshot_barrier_counts_successful_publications_without_waiting_for_dropped_events() -> Result<()>
{
    let (mut writer, reader) = event_ring(1);
    writer.emit(event(1));
    writer.emit(event(2));
    assert_eq!(writer.published(), 1);
    assert_eq!(writer.dropped(), 1);
    let collector = Collector::new(reader, ObservationStore::new(8)?, 0, 2)?;
    let snapshot = collector
        .snapshot(writer.published(), true, false)?
        .finish()?;
    assert_eq!(snapshot.window.timeline().len(), 1);
    assert_eq!(snapshot.window.timeline()[0].event.object_id, 1);
    Ok(())
}
