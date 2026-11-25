use super::*;
use infer_core::RequestId;
use infer_spi::RESPONSES_PAYLOAD_VERSION;

fn record(id: &str, status: ResponseStatus) -> StoredResponse {
    StoredResponse {
        id: ResponseId::new(id).unwrap(),
        request_id: RequestId::new(1).unwrap(),
        model: "1".to_string(),
        created_at: 100,
        completed_at: (status == ResponseStatus::Completed).then_some(101),
        status,
        incomplete: None,
        error: None,
        usage: None,
        previous_response_id: None,
        request_input: b"[]".to_vec(),
        response: b"{}".to_vec(),
        payload_version: RESPONSES_PAYLOAD_VERSION,
    }
}

fn store(max_entries: usize) -> InMemoryResponseStore {
    InMemoryResponseStore::new(InMemoryResponseStoreConfig {
        max_entries,
        max_bytes: 1024 * 1024,
        max_record_bytes: 1024,
        max_events: 4,
        ttl: None,
    })
    .unwrap()
}

#[tokio::test]
async fn put_get_and_delete_round_trip() {
    let store = store(4);
    let id = ResponseId::new("resp_1").unwrap();
    assert!(store.get(&id).await.unwrap().is_none());
    store
        .put(
            record("resp_1", ResponseStatus::InProgress),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    assert_eq!(
        store.get(&id).await.unwrap().unwrap().status,
        ResponseStatus::InProgress
    );
    store.delete(&id).await.unwrap();
    assert!(store.get(&id).await.unwrap().is_none());
}

#[tokio::test]
async fn terminal_condition_keeps_any_completed_record() {
    let store = store(4);
    store
        .put(
            record("resp_1", ResponseStatus::Completed),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    store
        .put(
            record("resp_1", ResponseStatus::Cancelled),
            WriteCondition::UnlessTerminal,
        )
        .await
        .unwrap();
    assert_eq!(
        store
            .get(&ResponseId::new("resp_1").unwrap())
            .await
            .unwrap()
            .unwrap()
            .status,
        ResponseStatus::Completed
    );
}

#[tokio::test]
async fn cancellation_condition_keeps_a_cancelled_record() {
    let store = store(4);
    let mut cancelled = record("resp_1", ResponseStatus::Cancelled);
    cancelled.completed_at = None;
    store.put(cancelled, WriteCondition::Always).await.unwrap();
    store
        .put(
            record("resp_1", ResponseStatus::Completed),
            WriteCondition::UnlessStatus(ResponseStatus::Cancelled),
        )
        .await
        .unwrap();
    assert_eq!(
        store
            .get(&ResponseId::new("resp_1").unwrap())
            .await
            .unwrap()
            .unwrap()
            .status,
        ResponseStatus::Cancelled
    );
}

#[tokio::test]
async fn transitions_are_checked_and_atomic() {
    let store = store(4);
    let id = ResponseId::new("resp_1").unwrap();
    store
        .put(
            record("resp_1", ResponseStatus::InProgress),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    let updated = store
        .transition(
            &id,
            &[ResponseStatus::InProgress],
            ResponseStatus::Completed,
        )
        .await
        .unwrap();
    assert_eq!(updated.status, ResponseStatus::Completed);
    assert_eq!(
        store
            .transition(
                &id,
                &[ResponseStatus::InProgress],
                ResponseStatus::Cancelled
            )
            .await
            .unwrap_err()
            .code,
        ErrorCode::Conflict
    );
    assert_eq!(
        store
            .transition(
                &ResponseId::new("resp_missing").unwrap(),
                &[ResponseStatus::InProgress],
                ResponseStatus::Cancelled
            )
            .await
            .unwrap_err()
            .code,
        ErrorCode::NotFound
    );
}

#[tokio::test]
async fn least_recently_used_terminal_records_are_evicted() {
    let store = store(2);
    store
        .put(
            record("resp_1", ResponseStatus::Completed),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    store
        .put(
            record("resp_2", ResponseStatus::Completed),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    // Touch resp_1 so resp_2 becomes the oldest.
    store
        .get(&ResponseId::new("resp_1").unwrap())
        .await
        .unwrap();
    store
        .put(
            record("resp_3", ResponseStatus::Completed),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    assert!(
        store
            .get(&ResponseId::new("resp_1").unwrap())
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .get(&ResponseId::new("resp_2").unwrap())
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .get(&ResponseId::new("resp_3").unwrap())
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn active_records_are_not_evicted_for_capacity() {
    let store = store(1);
    store
        .put(
            record("resp_1", ResponseStatus::InProgress),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    let error = store
        .put(
            record("resp_2", ResponseStatus::Completed),
            WriteCondition::Always,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Capacity);
}

#[tokio::test]
async fn record_and_total_byte_limits_are_enforced() {
    let store = InMemoryResponseStore::new(InMemoryResponseStoreConfig {
        max_entries: 4,
        max_bytes: 16,
        max_record_bytes: 16,
        max_events: 1,
        ttl: None,
    })
    .unwrap();
    let mut large = record("resp_1", ResponseStatus::InProgress);
    large.request_input = vec![b'x'; 20];
    assert_eq!(
        store
            .put(large, WriteCondition::Always)
            .await
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
}

#[tokio::test]
async fn terminal_records_expire() {
    let store = InMemoryResponseStore::new(InMemoryResponseStoreConfig {
        max_entries: 4,
        max_bytes: 1024,
        max_record_bytes: 1024,
        max_events: 1,
        ttl: Some(Duration::ZERO),
    })
    .unwrap();
    store
        .put(
            record("resp_1", ResponseStatus::Completed),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    assert!(
        store
            .get(&ResponseId::new("resp_1").unwrap())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn events_are_ordered_filtered_and_bounded() {
    let store = store(4);
    let id = ResponseId::new("resp_1").unwrap();
    store
        .put(
            record("resp_1", ResponseStatus::InProgress),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    for sequence in 1..=4 {
        store
            .append_event(
                &id,
                ResponseEvent {
                    sequence,
                    payload: b"{}".to_vec(),
                },
            )
            .await
            .unwrap();
    }
    assert_eq!(store.events(&id, None).await.unwrap().len(), 4);
    assert_eq!(store.events(&id, Some(2)).await.unwrap().len(), 2);
    assert_eq!(
        store
            .append_event(
                &id,
                ResponseEvent {
                    sequence: 5,
                    payload: b"{}".to_vec(),
                },
            )
            .await
            .unwrap_err()
            .code,
        ErrorCode::Capacity
    );
    assert_eq!(
        store
            .append_event(
                &id,
                ResponseEvent {
                    sequence: 4,
                    payload: b"{}".to_vec(),
                },
            )
            .await
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
}

#[tokio::test]
async fn invalid_records_are_rejected() {
    let store = store(4);
    let mut invalid = record("resp_1", ResponseStatus::InProgress);
    invalid.request_input.clear();
    assert_eq!(
        store
            .put(invalid, WriteCondition::Always)
            .await
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
}

#[tokio::test]
async fn total_byte_budget_evicts_terminal_records() {
    let store = InMemoryResponseStore::new(InMemoryResponseStoreConfig {
        max_entries: 4,
        max_bytes: 8,
        max_record_bytes: 8,
        max_events: 1,
        ttl: None,
    })
    .unwrap();
    for id in ["resp_1", "resp_2", "resp_3"] {
        store
            .put(
                record(id, ResponseStatus::Completed),
                WriteCondition::Always,
            )
            .await
            .unwrap();
    }
    assert!(
        store
            .get(&ResponseId::new("resp_1").unwrap())
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .get(&ResponseId::new("resp_2").unwrap())
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .get(&ResponseId::new("resp_3").unwrap())
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn replacing_a_record_does_not_count_its_old_bytes_twice() {
    let store = store(1);
    store
        .put(
            record("resp_1", ResponseStatus::InProgress),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    store
        .put(
            record("resp_1", ResponseStatus::Completed),
            WriteCondition::Always,
        )
        .await
        .unwrap();
    assert_eq!(
        store
            .get(&ResponseId::new("resp_1").unwrap())
            .await
            .unwrap()
            .unwrap()
            .status,
        ResponseStatus::Completed
    );
}
