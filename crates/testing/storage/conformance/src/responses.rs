//! Behavior every [`ResponseStore`] must show.
use crate::expect::{assert_code, check, fresh_id};
use infer_core::{ErrorCode, RequestId, Result};
use infer_spi::{
    IncompleteReason, RESPONSES_PAYLOAD_VERSION, ResponseError, ResponseEvent, ResponseId,
    ResponseStatus, ResponseStore, ResponseUsage, StoredResponse, WriteCondition,
};
use std::time::Duration;
use tokio::time::sleep;

/// Events the store under test must keep per response; the suite fills exactly this many.
pub const EVENTS: usize = 4;

fn id(label: &str) -> Result<ResponseId> {
    ResponseId::new(format!("resp_{}", fresh_id(label)?))
}

fn record(id: &ResponseId, status: ResponseStatus) -> Result<StoredResponse> {
    Ok(StoredResponse {
        id: id.clone(),
        request_id: RequestId::new(7)?,
        model: "model".to_owned(),
        created_at: 100,
        completed_at: matches!(status, ResponseStatus::Completed).then_some(101),
        status,
        incomplete: None,
        error: None,
        usage: None,
        previous_response_id: None,
        request_input: b"[]".to_vec(),
        response: b"{}".to_vec(),
        payload_version: RESPONSES_PAYLOAD_VERSION,
    })
}

fn failed(id: &ResponseId, message: &str) -> Result<StoredResponse> {
    let mut response = record(id, ResponseStatus::Failed)?;
    response.error = Some(ResponseError::new("server_error", message)?);
    Ok(response)
}

fn event(sequence: u64) -> ResponseEvent {
    ResponseEvent {
        sequence,
        payload: format!("{{\"n\":{sequence}}}").into_bytes(),
    }
}

async fn status(store: &dyn ResponseStore, id: &ResponseId) -> Result<Option<ResponseStatus>> {
    Ok(store.get(id).await?.map(|response| response.status))
}

/// Run every case. The store must keep [`EVENTS`] events per response and accept records of
/// its advertised size.
///
/// # Errors
/// Returns the first failed expectation or store error.
pub async fn suite(store: &dyn ResponseStore) -> Result<()> {
    round_trip(store).await?;
    conditional_writes(store).await?;
    transitions(store).await?;
    events(store).await?;
    invalid_and_oversized(store).await
}

async fn round_trip(store: &dyn ResponseStore) -> Result<()> {
    let id = id("round_trip")?;
    let mut full = record(&id, ResponseStatus::Incomplete)?;
    full.incomplete = Some(IncompleteReason::MaxOutputTokens);
    full.usage = Some(ResponseUsage {
        input_tokens: 10,
        output_tokens: 20,
        reasoning_tokens: 3,
        cached_tokens: 4,
        cache_write_tokens: 5,
    });
    full.previous_response_id = Some(self::id("previous")?);
    full.request_input = b"[{\"role\":\"user\",\"content\":\"\\u0000 bytes\"}]".to_vec();
    store.put(full.clone(), WriteCondition::Always).await?;
    check(
        store.get(&id).await? == Some(full),
        "every field reads back unchanged",
    )?;
    let failed_id = self::id("failed")?;
    let failure = failed(&failed_id, "boom")?;
    store.put(failure.clone(), WriteCondition::Always).await?;
    check(
        store.get(&failed_id).await? == Some(failure),
        "an error reads back unchanged",
    )?;
    store.delete(&id).await?;
    check(
        store.get(&id).await?.is_none(),
        "a deleted response is absent",
    )?;
    store.delete(&id).await
}

async fn conditional_writes(store: &dyn ResponseStore) -> Result<()> {
    let id = id("conditions")?;
    store
        .put(
            record(&id, ResponseStatus::InProgress)?,
            WriteCondition::Always,
        )
        .await?;
    store
        .put(
            record(&id, ResponseStatus::Completed)?,
            WriteCondition::UnlessTerminal,
        )
        .await?;
    store
        .put(failed(&id, "late")?, WriteCondition::UnlessTerminal)
        .await?;
    check(
        status(store, &id).await? == Some(ResponseStatus::Completed),
        "a terminal response is not overwritten by a later terminal snapshot",
    )?;
    let cancelled = self::id("cancelled")?;
    store
        .put(
            record(&cancelled, ResponseStatus::Cancelled)?,
            WriteCondition::Always,
        )
        .await?;
    let late = record(&cancelled, ResponseStatus::Completed)?;
    store
        .put(
            late.clone(),
            WriteCondition::UnlessStatus(ResponseStatus::Cancelled),
        )
        .await?;
    check(
        status(store, &cancelled).await? == Some(ResponseStatus::Cancelled),
        "a cancelled response keeps its status",
    )?;
    store.put(late, WriteCondition::Always).await?;
    check(
        status(store, &cancelled).await? == Some(ResponseStatus::Completed),
        "an unconditional write replaces any status",
    )
}

async fn transitions(store: &dyn ResponseStore) -> Result<()> {
    let id = id("transitions")?;
    store
        .put(record(&id, ResponseStatus::Queued)?, WriteCondition::Always)
        .await?;
    let active = [ResponseStatus::Queued, ResponseStatus::InProgress];
    let moved = store
        .transition(&id, &active, ResponseStatus::InProgress)
        .await?;
    check(
        moved.status == ResponseStatus::InProgress,
        "a transition returns the new status",
    )?;
    check(
        status(store, &id).await? == Some(ResponseStatus::InProgress),
        "a transition is stored",
    )?;
    let queued_only = [ResponseStatus::Queued];
    assert_code(
        store
            .transition(&id, &queued_only, ResponseStatus::Cancelled)
            .await,
        ErrorCode::Conflict,
        "a transition from a status outside the allowed list",
    )?;
    store
        .transition(&id, &active, ResponseStatus::Cancelled)
        .await?;
    assert_code(
        store
            .transition(
                &id,
                &[ResponseStatus::Cancelled],
                ResponseStatus::InProgress,
            )
            .await,
        ErrorCode::Conflict,
        "a transition out of a terminal status",
    )?;
    assert_code(
        store
            .transition(&self::id("missing")?, &active, ResponseStatus::Cancelled)
            .await,
        ErrorCode::NotFound,
        "a transition of an unknown response",
    )
}

async fn events(store: &dyn ResponseStore) -> Result<()> {
    let id = id("events")?;
    assert_code(
        store.append_event(&id, event(1)).await,
        ErrorCode::NotFound,
        "an event for an unknown response",
    )?;
    store
        .put(
            record(&id, ResponseStatus::InProgress)?,
            WriteCondition::Always,
        )
        .await?;
    let full = u64::try_from(EVENTS).map_err(|_| infer_core::Error::invariant("event count"))?;
    for sequence in 1..=2 {
        store.append_event(&id, event(sequence)).await?;
    }
    // The terminal snapshot of a background response must keep the events streamed before it.
    store
        .put(
            record(&id, ResponseStatus::Completed)?,
            WriteCondition::UnlessTerminal,
        )
        .await?;
    for sequence in 3..=full {
        store.append_event(&id, event(sequence)).await?;
    }
    check(
        store.events(&id, None).await? == (1..=full).map(event).collect::<Vec<_>>(),
        "events keep their order across the terminal snapshot",
    )?;
    check(
        store.events(&id, Some(2)).await? == (3..=full).map(event).collect::<Vec<_>>(),
        "events after a sequence skip earlier ones",
    )?;
    assert_code(
        store.append_event(&id, event(full)).await,
        ErrorCode::InvalidInput,
        "a repeated event sequence",
    )?;
    assert_code(
        store.append_event(&id, event(full + 1)).await,
        ErrorCode::Capacity,
        "an event past the limit",
    )?;
    store.delete(&id).await?;
    check(
        store.events(&id, None).await?.is_empty(),
        "deleting a response deletes its events",
    )
}

async fn invalid_and_oversized(store: &dyn ResponseStore) -> Result<()> {
    let id = id("invalid")?;
    let mut invalid = record(&id, ResponseStatus::InProgress)?;
    invalid.completed_at = Some(1);
    assert_code(
        store.put(invalid, WriteCondition::Always).await,
        ErrorCode::InvalidInput,
        "an inconsistent record",
    )?;
    let mut large = record(&id, ResponseStatus::InProgress)?;
    large.response = vec![b' '; store.capabilities().max_record_bytes];
    assert_code(
        store.put(large, WriteCondition::Always).await,
        ErrorCode::Capacity,
        "a record over the size limit",
    )?;
    check(
        store.get(&id).await?.is_none(),
        "refused records are not stored",
    )
}

/// Terminal responses disappear after `ttl` while active ones stay. Run against a store
/// configured with that short retention.
///
/// # Errors
/// Returns the first failed expectation or store error.
pub async fn expiry(store: &dyn ResponseStore, ttl: Duration) -> Result<()> {
    let done = id("expired")?;
    let active = id("active")?;
    store
        .put(
            record(&done, ResponseStatus::Completed)?,
            WriteCondition::Always,
        )
        .await?;
    store
        .put(
            record(&active, ResponseStatus::InProgress)?,
            WriteCondition::Always,
        )
        .await?;
    sleep(ttl * 3).await;
    check(
        store.get(&done).await?.is_none(),
        "a terminal response expires",
    )?;
    check(
        store.get(&active).await?.is_some(),
        "an active response never expires",
    )
}
