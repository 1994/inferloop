//! The in-process backends against the shared suites, and their bounded-capacity rules.
use infer_core::{ErrorCode, Result};
use infer_spi::{
    Namespace, Precondition, RecordKey, RecordState, RecordStore, RecordWrite, ResponseStore,
    Retention,
};
use infer_storage::{
    MemoryRecordStore, MemoryRecordStoreConfig, RecordResponseStore, ResponseRetention,
};
use infer_storage_conformance::{records, responses};
use std::sync::Arc;
use std::time::Duration;

const VALUE_BYTES: usize = 64 * 1024;
const SHORT_TTL: Duration = Duration::from_millis(50);

fn bounded(max_records: usize, max_bytes: usize) -> Result<MemoryRecordStore> {
    MemoryRecordStore::new(MemoryRecordStoreConfig {
        max_records,
        max_bytes,
        max_value_bytes: max_bytes.min(VALUE_BYTES),
    })
}

fn responses_over(
    records: MemoryRecordStore,
    ttl: Option<Duration>,
) -> Result<RecordResponseStore> {
    RecordResponseStore::new(
        Arc::new(records),
        ResponseRetention {
            max_record_bytes: 1024,
            max_events: responses::EVENTS,
            ttl,
        },
    )
}

fn key(id: &str) -> Result<RecordKey> {
    RecordKey::new(Namespace::new("memory_test")?, id)
}

fn write(value: &[u8], retention: Retention) -> Result<RecordWrite<'_>> {
    Ok(RecordWrite {
        value,
        state: RecordState::new(0)?,
        retention,
        condition: Precondition::Any,
    })
}

const DISPOSABLE: Retention = Retention::Disposable(None);

#[tokio::test]
async fn memory_records_conform() -> Result<()> {
    records::suite(&MemoryRecordStore::default()).await
}

#[tokio::test]
async fn memory_responses_conform() -> Result<()> {
    let store = responses_over(MemoryRecordStore::default(), None)?;
    responses::suite(&store).await?;
    responses::expiry(
        &responses_over(MemoryRecordStore::default(), Some(SHORT_TTL))?,
        SHORT_TTL,
    )
    .await
}

#[tokio::test]
async fn disposable_records_go_least_recently_used_first() -> Result<()> {
    let store = bounded(2, 1024)?;
    store.put(&key("a")?, write(b"a", DISPOSABLE)?).await?;
    store.put(&key("b")?, write(b"b", DISPOSABLE)?).await?;
    store.get(&key("a")?).await?;
    store.put(&key("c")?, write(b"c", DISPOSABLE)?).await?;
    assert!(
        store.get(&key("a")?).await?.is_some(),
        "the recently read record stays"
    );
    assert!(
        store.get(&key("b")?).await?.is_none(),
        "the least recently used record goes"
    );
    assert!(store.get(&key("c")?).await?.is_some());
    Ok(())
}

#[tokio::test]
async fn pinned_records_are_never_evicted() -> Result<()> {
    let store = bounded(2, 1024)?;
    store
        .put(&key("p1")?, write(b"1", Retention::Pinned)?)
        .await?;
    store
        .put(&key("p2")?, write(b"2", Retention::Pinned)?)
        .await?;
    let refused = store.put(&key("p3")?, write(b"3", DISPOSABLE)?).await;
    assert_eq!(
        refused.err().map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    assert!(store.get(&key("p1")?).await?.is_some() && store.get(&key("p2")?).await?.is_some());
    Ok(())
}

#[tokio::test]
async fn the_byte_bound_counts_values_and_logs_once() -> Result<()> {
    let store = bounded(8, 16)?;
    store.put(&key("x")?, write(&[0; 8], DISPOSABLE)?).await?;
    // Rewriting the same record does not count its old bytes.
    store.put(&key("x")?, write(&[1; 8], DISPOSABLE)?).await?;
    store.append(&key("x")?, 1, &[2; 4], 4).await?;
    store
        .put(&key("y")?, write(&[3; 4], Retention::Pinned)?)
        .await?;
    assert!(
        store.get(&key("x")?).await?.is_some(),
        "16 bytes fit exactly"
    );
    // Four more bytes push out the disposable record together with its log.
    store
        .put(&key("z")?, write(&[4; 4], Retention::Pinned)?)
        .await?;
    assert!(store.get(&key("x")?).await?.is_none());
    assert_eq!(store.log(&key("x")?, None).await?, Vec::new());
    Ok(())
}

#[tokio::test]
async fn a_bounded_store_drops_finished_responses_and_keeps_running_ones() -> Result<()> {
    let store = responses_over(bounded(2, 64 * 1024)?, None)?;
    let response = |id: &str, status| -> Result<infer_spi::StoredResponse> {
        Ok(infer_spi::StoredResponse {
            id: infer_spi::ResponseId::new(id)?,
            request_id: infer_core::RequestId::new(1)?,
            model: "m".to_owned(),
            created_at: 1,
            completed_at: matches!(status, infer_spi::ResponseStatus::Completed).then_some(2),
            status,
            incomplete: None,
            error: None,
            usage: None,
            previous_response_id: None,
            request_input: b"[]".to_vec(),
            response: b"{}".to_vec(),
            payload_version: infer_spi::RESPONSES_PAYLOAD_VERSION,
        })
    };
    let always = infer_spi::WriteCondition::Always;
    store
        .put(
            response("resp_done", infer_spi::ResponseStatus::Completed)?,
            always,
        )
        .await?;
    store
        .put(
            response("resp_running", infer_spi::ResponseStatus::InProgress)?,
            always,
        )
        .await?;
    store
        .put(
            response("resp_new", infer_spi::ResponseStatus::InProgress)?,
            always,
        )
        .await?;
    assert!(
        store
            .get(&infer_spi::ResponseId::new("resp_done")?)
            .await?
            .is_none()
    );
    assert!(
        store
            .get(&infer_spi::ResponseId::new("resp_running")?)
            .await?
            .is_some()
    );
    let full = store
        .put(
            response("resp_third", infer_spi::ResponseStatus::InProgress)?,
            always,
        )
        .await;
    assert_eq!(
        full.err().map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    Ok(())
}

#[test]
fn bounds_must_be_nonzero_and_consistent() {
    let code = |config| MemoryRecordStore::new(config).err().map(|error| error.code);
    let valid = MemoryRecordStoreConfig::default();
    assert_eq!(
        code(MemoryRecordStoreConfig {
            max_records: 0,
            ..valid
        }),
        Some(ErrorCode::InvalidInput)
    );
    assert_eq!(
        code(MemoryRecordStoreConfig {
            max_value_bytes: valid.max_bytes + 1,
            ..valid
        }),
        Some(ErrorCode::InvalidInput)
    );
    assert!(code(valid).is_none());
}

#[test]
fn responses_need_a_record_store_that_fits_them() -> Result<()> {
    let small = bounded(4, 512)?;
    let retention = ResponseRetention {
        max_record_bytes: 1024,
        ..ResponseRetention::default()
    };
    let refused = RecordResponseStore::new(Arc::new(small), retention)
        .err()
        .map(|error| error.code);
    assert_eq!(refused, Some(ErrorCode::InvalidInput));
    Ok(())
}
