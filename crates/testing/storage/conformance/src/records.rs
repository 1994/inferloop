//! Behavior every [`RecordStore`] must show.
use crate::expect::{assert_code, check, fresh_id};
use infer_core::{ErrorCode, Result};
use infer_spi::{
    Namespace, Precondition, PutOutcome, RecordKey, RecordState, RecordStore, RecordWrite,
    Retention, StateSet, UnixMillis, put_by_version,
};

const NAMESPACE: &str = "conformance";
const LOG_ENTRIES: usize = 3;

fn key(label: &str) -> Result<RecordKey> {
    RecordKey::new(Namespace::new(NAMESPACE)?, &fresh_id(label)?)
}

fn state(label: u8) -> Result<RecordState> {
    RecordState::new(label)
}

fn write(value: &[u8], label: u8, condition: Precondition) -> Result<RecordWrite<'_>> {
    Ok(RecordWrite {
        value,
        state: state(label)?,
        retention: Retention::Pinned,
        condition,
    })
}

async fn written(store: &dyn RecordStore, key: &RecordKey, write: RecordWrite<'_>) -> Result<u64> {
    match store.put(key, write).await? {
        PutOutcome::Written(version) => Ok(version.get()),
        PutOutcome::Rejected(current) => Err(infer_core::Error::invariant(format!(
            "write to {key} was rejected (current {current:?})"
        ))),
    }
}

/// Run every case that needs only one store handle.
///
/// # Errors
/// Returns the first failed expectation or store error.
pub async fn suite(store: &dyn RecordStore) -> Result<()> {
    round_trip(store).await?;
    versions(store).await?;
    conditions(store).await?;
    state_predicates_match_the_fallback(store).await?;
    logs(store).await?;
    expiry(store).await?;
    limits(store).await
}

async fn round_trip(store: &dyn RecordStore) -> Result<()> {
    let key = key("round_trip")?;
    check(store.get(&key).await?.is_none(), "a new key is absent")?;
    let retention = Retention::Disposable(Some(UnixMillis::new(u64::MAX / 2)));
    let value = b"value \x00\xff bytes";
    let version = written(
        store,
        &key,
        RecordWrite {
            retention,
            ..write(value, 3, Precondition::Any)?
        },
    )
    .await?;
    let record = store.get(&key).await?;
    check(
        record.as_ref().is_some_and(|record| {
            record.version.get() == version
                && record.state.get() == 3
                && record.retention == retention
                && record.value == value
        }),
        "a record reads back as written",
    )?;
    store.delete(&key).await?;
    check(
        store.get(&key).await?.is_none(),
        "a deleted record is absent",
    )?;
    store.delete(&key).await
}

async fn versions(store: &dyn RecordStore) -> Result<()> {
    let key = key("versions")?;
    let first = written(store, &key, write(b"1", 0, Precondition::Any)?).await?;
    let second = written(store, &key, write(b"2", 0, Precondition::Any)?).await?;
    check(second > first, "a rewrite gets a newer version")?;
    store.delete(&key).await?;
    let recreated = written(store, &key, write(b"3", 0, Precondition::Any)?).await?;
    check(
        recreated > second,
        "a recreated record never reuses a version",
    )
}

async fn conditions(store: &dyn RecordStore) -> Result<()> {
    let key = key("conditions")?;
    let version = written(store, &key, write(b"a", 1, Precondition::Absent)?).await?;
    let again = store
        .put(&key, write(b"b", 1, Precondition::Absent)?)
        .await?;
    check(
        matches!(again, PutOutcome::Rejected(Some((current, label))) if current.get() == version && label.get() == 1),
        "absent rejects an existing record and reports it",
    )?;
    let stale = infer_spi::Version::new(version.wrapping_sub(1));
    let rejected = store
        .put(&key, write(b"c", 1, Precondition::Version(stale))?)
        .await?;
    check(
        matches!(rejected, PutOutcome::Rejected(Some(_))),
        "a stale version is rejected",
    )?;
    let exact = infer_spi::Version::new(version);
    written(store, &key, write(b"d", 2, Precondition::Version(exact))?).await?;
    let missing = key_absent(store, "conditions_missing").await?;
    let on_missing = store
        .put(&missing, write(b"e", 0, Precondition::Version(exact))?)
        .await?;
    check(
        on_missing == PutOutcome::Rejected(None),
        "a version write to an absent key is rejected",
    )?;
    let value = store.get(&key).await?.map(|record| record.value);
    check(
        value.as_deref() == Some(b"d".as_slice()),
        "only accepted writes take effect",
    )
}

async fn key_absent(store: &dyn RecordStore, label: &str) -> Result<RecordKey> {
    let key = key(label)?;
    check(store.get(&key).await?.is_none(), "a fresh key is absent")?;
    Ok(key)
}

async fn put_state(
    store: &dyn RecordStore,
    key: &RecordKey,
    excluded: StateSet,
    fallback: bool,
    value: &[u8],
    label: u8,
) -> Result<PutOutcome> {
    let write = write(value, label, Precondition::StateNotIn(excluded))?;
    if fallback {
        put_by_version(store, key, write).await
    } else {
        store.put(key, write).await
    }
}

/// `StateNotIn` gives the same outcomes evaluated by the store and by [`put_by_version`].
async fn state_predicates_match_the_fallback(store: &dyn RecordStore) -> Result<()> {
    let excluded = StateSet::of(&[state(4)?, state(5)?]);
    for fallback in [false, true] {
        let key = key(if fallback { "fallback" } else { "native" })?;
        check(
            matches!(
                put_state(store, &key, excluded, fallback, b"new", 1).await?,
                PutOutcome::Written(_)
            ),
            "an absent record passes",
        )?;
        check(
            matches!(
                put_state(store, &key, excluded, fallback, b"next", 2).await?,
                PutOutcome::Written(_)
            ),
            "an allowed state passes",
        )?;
        check(
            matches!(
                put_state(store, &key, excluded, fallback, b"done", 4).await?,
                PutOutcome::Written(_)
            ),
            "writing an excluded state passes",
        )?;
        let blocked = put_state(store, &key, excluded, fallback, b"late", 1).await?;
        check(
            matches!(blocked, PutOutcome::Rejected(Some((_, label))) if label.get() == 4),
            "an excluded current state blocks the write",
        )?;
        let value = store.get(&key).await?.map(|record| record.value);
        check(
            value.as_deref() == Some(b"done".as_slice()),
            "the blocked write left no trace",
        )?;
    }
    Ok(())
}

async fn logs(store: &dyn RecordStore) -> Result<()> {
    let key = key("logs")?;
    assert_code(
        store.append(&key, 1, b"x", LOG_ENTRIES).await,
        ErrorCode::NotFound,
        "append to an absent record",
    )?;
    check(
        store.log(&key, None).await?.is_empty(),
        "an absent record has an empty log",
    )?;
    written(store, &key, write(b"r", 0, Precondition::Any)?).await?;
    store.append(&key, 1, b"one", LOG_ENTRIES).await?;
    store.append(&key, 2, b"two", LOG_ENTRIES).await?;
    // A rewrite keeps the log: a terminal snapshot must not erase the events before it.
    written(store, &key, write(b"r2", 1, Precondition::Any)?).await?;
    store.append(&key, 5, b"five", LOG_ENTRIES).await?;
    let sequences: Vec<u64> = store
        .log(&key, None)
        .await?
        .iter()
        .map(|entry| entry.sequence)
        .collect();
    check(
        sequences == [1, 2, 5],
        "the log keeps order across a rewrite",
    )?;
    let after: Vec<u64> = store
        .log(&key, Some(2))
        .await?
        .iter()
        .map(|entry| entry.sequence)
        .collect();
    check(
        after == [5],
        "reading after a sequence skips earlier entries",
    )?;
    assert_code(
        store.append(&key, 5, b"dup", LOG_ENTRIES + 1).await,
        ErrorCode::InvalidInput,
        "a sequence that does not increase",
    )?;
    assert_code(
        store.append(&key, 6, b"six", LOG_ENTRIES).await,
        ErrorCode::Capacity,
        "a full log",
    )?;
    let values: Vec<Vec<u8>> = store
        .log(&key, None)
        .await?
        .into_iter()
        .map(|entry| entry.value)
        .collect();
    check(
        values == [b"one".to_vec(), b"two".to_vec(), b"five".to_vec()],
        "failed appends add nothing",
    )?;
    store.delete(&key).await?;
    written(store, &key, write(b"r3", 0, Precondition::Any)?).await?;
    check(
        store.log(&key, None).await?.is_empty(),
        "deleting a record deletes its log",
    )
}

async fn expiry(store: &dyn RecordStore) -> Result<()> {
    let key = key("expired")?;
    let past = RecordWrite {
        retention: Retention::Disposable(Some(UnixMillis::new(1))),
        ..write(b"old", 0, Precondition::Any)?
    };
    written(store, &key, past).await?;
    check(
        store.get(&key).await?.is_none(),
        "an expired record is absent",
    )?;
    assert_code(
        store.append(&key, 1, b"x", LOG_ENTRIES).await,
        ErrorCode::NotFound,
        "append to an expired record",
    )?;
    let fresh = store
        .put(&key, write(b"new", 0, Precondition::Absent)?)
        .await?;
    check(
        matches!(fresh, PutOutcome::Written(_)),
        "an expired record counts as absent",
    )
}

async fn limits(store: &dyn RecordStore) -> Result<()> {
    let key = key("limits")?;
    let limit = store.capabilities().max_value_bytes;
    let oversized = vec![0_u8; limit + 1];
    assert_code(
        store
            .put(&key, write(&oversized, 0, Precondition::Any)?)
            .await,
        ErrorCode::Capacity,
        "a value over the limit",
    )?;
    check(
        store.get(&key).await?.is_none(),
        "a refused value is not stored",
    )
}

/// Two handles on one shared store: concurrent version writes let exactly one through.
///
/// # Errors
/// Returns the first failed expectation or store error.
pub async fn concurrent_versions(first: &dyn RecordStore, second: &dyn RecordStore) -> Result<()> {
    let key = key("race")?;
    let version = written(first, &key, write(b"base", 0, Precondition::Any)?).await?;
    let condition = Precondition::Version(infer_spi::Version::new(version));
    let (left, right) = tokio::join!(
        first.put(&key, write(b"left", 1, condition)?),
        second.put(&key, write(b"right", 2, condition)?),
    );
    let outcomes = [left?, right?];
    let wins = outcomes
        .iter()
        .filter(|outcome| matches!(outcome, PutOutcome::Written(_)))
        .count();
    check(wins == 1, "exactly one concurrent version write wins")
}
