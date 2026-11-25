//! One behavioral suite for every Responses store: the in-process reference, SQLite and
//! PostgreSQL must agree on retention, conditional writes, transitions and events.
//!
//! The PostgreSQL cases need a server: run them with
//! `INFER_TEST_POSTGRES_URL=postgres://... cargo test -p infer-response-store -- --ignored`.
//! They fail, rather than pass empty, when the variable is missing.
use infer_agentic::{InMemoryResponseStore, InMemoryResponseStoreConfig};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_response_store::{PersistentStoreConfig, PostgresResponseStore, SqliteResponseStore};
use infer_spi::{
    IncompleteReason, RESPONSES_PAYLOAD_VERSION, ResponseError, ResponseEvent, ResponseId,
    ResponseStatus, ResponseStore, ResponseUsage, StoredResponse, WriteCondition,
};
use rusqlite::Connection;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;
use tokio_postgres::Config as PostgresConfig;

const RECORD_BYTES: usize = 1024;
const EVENTS: usize = 4;
const SHORT_TTL: Duration = Duration::from_millis(50);
const POSTGRES_URL: &str = "INFER_TEST_POSTGRES_URL";

static UNIQUE: AtomicU64 = AtomicU64::new(0);

/// A response id no other case or run uses, so cases can share one database.
fn unique_id(label: &str) -> Result<ResponseId> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::invariant("clock before epoch"))?
        .as_nanos();
    let count = UNIQUE.fetch_add(1, Ordering::Relaxed);
    ResponseId::new(format!("resp_{label}_{nanos}_{count}"))
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

fn event(sequence: u64) -> ResponseEvent {
    ResponseEvent {
        sequence,
        payload: format!("{{\"n\":{sequence}}}").into_bytes(),
    }
}

fn code<T>(result: Result<T>) -> Option<ErrorCode> {
    result.err().map(|error| error.code)
}

const fn limits(ttl: Option<Duration>) -> PersistentStoreConfig {
    PersistentStoreConfig {
        max_record_bytes: RECORD_BYTES,
        max_events: EVENTS,
        ttl,
    }
}

/// A database file removed when the case ends.
struct TempFile(PathBuf);

impl TempFile {
    fn new(label: &str) -> Result<Self> {
        let name = format!("{}.sqlite", unique_id(label)?);
        Ok(Self(env::temp_dir().join(name)))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let mut path = self.0.clone().into_os_string();
            path.push(suffix);
            let _missing = fs::remove_file(path);
        }
    }
}

async fn round_trip(store: &dyn ResponseStore) -> Result<()> {
    let id = unique_id("round_trip")?;
    let mut full = record(&id, ResponseStatus::Incomplete)?;
    full.incomplete = Some(IncompleteReason::MaxOutputTokens);
    full.usage = Some(ResponseUsage {
        input_tokens: 10,
        output_tokens: 20,
        reasoning_tokens: 3,
        cached_tokens: 4,
        cache_write_tokens: 5,
    });
    full.previous_response_id = Some(unique_id("previous")?);
    store.put(full.clone(), WriteCondition::Always).await?;
    assert_eq!(store.get(&id).await?, Some(full));
    let failed_id = unique_id("failed")?;
    let mut failed = record(&failed_id, ResponseStatus::Failed)?;
    failed.error = Some(ResponseError::new("server_error", "boom")?);
    store.put(failed.clone(), WriteCondition::Always).await?;
    assert_eq!(store.get(&failed_id).await?, Some(failed));
    store.delete(&id).await?;
    assert_eq!(store.get(&id).await?, None);
    store.delete(&id).await?;
    Ok(())
}

async fn conditional_writes(store: &dyn ResponseStore) -> Result<()> {
    let id = unique_id("conditions")?;
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
    let failed = record(&id, ResponseStatus::Failed);
    let mut failed = failed?;
    failed.error = Some(ResponseError::new("server_error", "late")?);
    store
        .put(failed.clone(), WriteCondition::UnlessTerminal)
        .await?;
    let kept = store.get(&id).await?.map(|response| response.status);
    assert_eq!(kept, Some(ResponseStatus::Completed));

    let cancelled_id = unique_id("cancelled")?;
    store
        .put(
            record(&cancelled_id, ResponseStatus::Cancelled)?,
            WriteCondition::Always,
        )
        .await?;
    let late = record(&cancelled_id, ResponseStatus::Completed)?;
    store
        .put(
            late.clone(),
            WriteCondition::UnlessStatus(ResponseStatus::Cancelled),
        )
        .await?;
    let kept = store
        .get(&cancelled_id)
        .await?
        .map(|response| response.status);
    assert_eq!(kept, Some(ResponseStatus::Cancelled));
    store.put(late, WriteCondition::Always).await?;
    let replaced = store
        .get(&cancelled_id)
        .await?
        .map(|response| response.status);
    assert_eq!(replaced, Some(ResponseStatus::Completed));
    Ok(())
}

async fn transitions(store: &dyn ResponseStore) -> Result<()> {
    let id = unique_id("transitions")?;
    store
        .put(record(&id, ResponseStatus::Queued)?, WriteCondition::Always)
        .await?;
    let active = [ResponseStatus::Queued, ResponseStatus::InProgress];
    let moved = store
        .transition(&id, &active, ResponseStatus::InProgress)
        .await?;
    assert_eq!(moved.status, ResponseStatus::InProgress);
    let stored = store.get(&id).await?.map(|response| response.status);
    assert_eq!(stored, Some(ResponseStatus::InProgress));
    let only_queued = [ResponseStatus::Queued];
    let conflict = store
        .transition(&id, &only_queued, ResponseStatus::Cancelled)
        .await;
    assert_eq!(code(conflict), Some(ErrorCode::Conflict));
    store
        .transition(&id, &active, ResponseStatus::Cancelled)
        .await?;
    let backwards = [ResponseStatus::Cancelled];
    let reopened = store
        .transition(&id, &backwards, ResponseStatus::InProgress)
        .await;
    assert_eq!(code(reopened), Some(ErrorCode::Conflict));
    let missing = unique_id("missing")?;
    let unknown = store
        .transition(&missing, &active, ResponseStatus::Cancelled)
        .await;
    assert_eq!(code(unknown), Some(ErrorCode::NotFound));
    Ok(())
}

async fn events(store: &dyn ResponseStore) -> Result<()> {
    let id = unique_id("events")?;
    assert_eq!(
        code(store.append_event(&id, event(1)).await),
        Some(ErrorCode::NotFound)
    );
    store
        .put(
            record(&id, ResponseStatus::InProgress)?,
            WriteCondition::Always,
        )
        .await?;
    for sequence in 1..=2 {
        store.append_event(&id, event(sequence)).await?;
    }
    // Writing the terminal snapshot must keep the events a background stream already produced.
    store
        .put(
            record(&id, ResponseStatus::Completed)?,
            WriteCondition::UnlessTerminal,
        )
        .await?;
    for sequence in 3..=4 {
        store.append_event(&id, event(sequence)).await?;
    }
    let all = store.events(&id, None).await?;
    assert_eq!(all, (1..=4).map(event).collect::<Vec<_>>());
    let later = store.events(&id, Some(2)).await?;
    assert_eq!(later, vec![event(3), event(4)]);
    assert_eq!(
        code(store.append_event(&id, event(4)).await),
        Some(ErrorCode::InvalidInput)
    );
    assert_eq!(
        code(store.append_event(&id, event(5)).await),
        Some(ErrorCode::Capacity)
    );
    store.delete(&id).await?;
    assert_eq!(store.events(&id, None).await?, Vec::new());
    Ok(())
}

async fn invalid_and_oversized(store: &dyn ResponseStore) -> Result<()> {
    let id = unique_id("invalid")?;
    let mut invalid = record(&id, ResponseStatus::InProgress)?;
    invalid.completed_at = Some(1);
    assert_eq!(
        code(store.put(invalid, WriteCondition::Always).await),
        Some(ErrorCode::InvalidInput)
    );
    let mut large = record(&id, ResponseStatus::InProgress)?;
    large.response = vec![b' '; RECORD_BYTES];
    assert_eq!(
        code(store.put(large, WriteCondition::Always).await),
        Some(ErrorCode::Capacity)
    );
    assert_eq!(store.get(&id).await?, None);
    Ok(())
}

/// Run against a store whose terminal records live [`SHORT_TTL`].
async fn expiry(store: &dyn ResponseStore) -> Result<()> {
    let done = unique_id("expired")?;
    let active = unique_id("active")?;
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
    sleep(SHORT_TTL * 3).await;
    assert_eq!(store.get(&done).await?, None);
    assert!(
        store.get(&active).await?.is_some(),
        "active responses never expire"
    );
    Ok(())
}

async fn suite(store: &dyn ResponseStore) -> Result<()> {
    round_trip(store).await?;
    conditional_writes(store).await?;
    transitions(store).await?;
    events(store).await?;
    invalid_and_oversized(store).await
}

fn memory(ttl: Option<Duration>) -> Result<InMemoryResponseStore> {
    InMemoryResponseStore::new(InMemoryResponseStoreConfig {
        max_entries: 64,
        max_bytes: 64 * RECORD_BYTES,
        max_record_bytes: RECORD_BYTES,
        max_events: EVENTS,
        ttl,
    })
}

#[tokio::test]
async fn memory_store_conforms() -> Result<()> {
    suite(&memory(None)?).await?;
    expiry(&memory(Some(SHORT_TTL))?).await
}

#[tokio::test]
async fn sqlite_store_conforms() -> Result<()> {
    let file = TempFile::new("suite")?;
    suite(&SqliteResponseStore::open(&file.0, limits(None))?).await?;
    let expiring = TempFile::new("expiry")?;
    expiry(&SqliteResponseStore::open(
        &expiring.0,
        limits(Some(SHORT_TTL)),
    )?)
    .await
}

#[tokio::test]
async fn sqlite_state_survives_reopening() -> Result<()> {
    let file = TempFile::new("reopen")?;
    let id = unique_id("reopen")?;
    let response = record(&id, ResponseStatus::Completed)?;
    {
        let store = SqliteResponseStore::open(&file.0, limits(None))?;
        store.put(response.clone(), WriteCondition::Always).await?;
        store.append_event(&id, event(1)).await?;
    }
    let store = SqliteResponseStore::open(&file.0, limits(None))?;
    assert_eq!(store.get(&id).await?, Some(response));
    assert_eq!(store.events(&id, None).await?, vec![event(1)]);
    Ok(())
}

#[tokio::test]
async fn sqlite_refuses_another_schema_version() -> Result<()> {
    let file = TempFile::new("schema")?;
    drop(SqliteResponseStore::open(&file.0, limits(None))?);
    bump_schema(&file.0)?;
    let reopened = SqliteResponseStore::open(&file.0, limits(None));
    assert_eq!(code(reopened), Some(ErrorCode::Unsupported));
    Ok(())
}

fn bump_schema(path: &Path) -> Result<()> {
    let connection = Connection::open(path).map_err(|error| Error::invalid(error.to_string()))?;
    connection
        .execute("UPDATE infer_response_store_meta SET value = value + 1", [])
        .map_err(|error| Error::invalid(error.to_string()))?;
    Ok(())
}

#[test]
fn persistent_limits_must_be_nonzero() {
    let file = env::temp_dir().join("unused.sqlite");
    let zero = PersistentStoreConfig {
        max_events: 0,
        ..limits(None)
    };
    assert_eq!(
        code(SqliteResponseStore::open(&file, zero)),
        Some(ErrorCode::InvalidInput)
    );
    let instant = limits(Some(Duration::ZERO));
    assert_eq!(
        code(SqliteResponseStore::open(&file, instant)),
        Some(ErrorCode::InvalidInput)
    );
}

fn postgres_url() -> Result<String> {
    env::var(POSTGRES_URL)
        .map_err(|_| Error::invalid(format!("set {POSTGRES_URL} to run the PostgreSQL cases")))
}

#[tokio::test]
#[ignore = "needs a PostgreSQL server named by INFER_TEST_POSTGRES_URL"]
async fn postgres_store_conforms() -> Result<()> {
    let url = postgres_url()?;
    suite(&PostgresResponseStore::connect(&url, limits(None)).await?).await?;
    expiry(&PostgresResponseStore::connect(&url, limits(Some(SHORT_TTL))).await?).await
}

#[tokio::test]
#[ignore = "needs a PostgreSQL server named by INFER_TEST_POSTGRES_URL"]
async fn postgres_transitions_are_atomic_across_connections() -> Result<()> {
    let url = postgres_url()?;
    let first = PostgresResponseStore::connect(&url, limits(None)).await?;
    let second = PostgresResponseStore::connect(&url, limits(None)).await?;
    let id = unique_id("race")?;
    first
        .put(
            record(&id, ResponseStatus::InProgress)?,
            WriteCondition::Always,
        )
        .await?;
    let active = [ResponseStatus::InProgress];
    let (left, right) = tokio::join!(
        first.transition(&id, &active, ResponseStatus::Cancelled),
        second.transition(&id, &active, ResponseStatus::Cancelled),
    );
    let mut codes = [code(left), code(right)];
    codes.sort_by_key(Option::is_some);
    assert_eq!(codes, [None, Some(ErrorCode::Conflict)]);
    Ok(())
}

#[tokio::test]
#[ignore = "needs a PostgreSQL server named by INFER_TEST_POSTGRES_URL"]
async fn postgres_debug_hides_the_password() -> Result<()> {
    let url = postgres_url()?;
    let store = PostgresResponseStore::connect(&url, limits(None)).await?;
    let shown = format!("{store:?}");
    let config =
        PostgresConfig::from_str(&url).map_err(|error| Error::invalid(error.to_string()))?;
    let password = config.get_password().unwrap_or_default();
    assert!(!password.is_empty(), "the test URL needs a password");
    assert!(
        !shown
            .as_bytes()
            .windows(password.len())
            .any(|window| window == password)
    );
    Ok(())
}

#[tokio::test]
async fn postgres_refuses_urls_that_require_tls() {
    let url = "postgres://user:secret@127.0.0.1:1/db?sslmode=require";
    let refused = PostgresResponseStore::connect(url, limits(None)).await;
    assert_eq!(code(refused), Some(ErrorCode::Unsupported));
}
