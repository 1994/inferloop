//! Responses state in a local SQLite database file.
use crate::record::{
    PersistentStoreConfig, ReadRow, Row, SCHEMA_VERSION, USAGE_COLUMNS, events_full, from_sql,
    not_found, now_millis, sources, to_sql, transition_conflict, unordered_event,
};
use crate::sql::{Statements, TRANSITION_SLOTS, schema};
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{
    RESPONSES_SPI_VERSION, ResponseEvent, ResponseFuture, ResponseId, ResponseStatus,
    ResponseStore, ResponseStoreCapabilities, StoredResponse, WriteCondition,
};
use rusqlite::{Connection, OptionalExtension, ToSql, Transaction, params};
use std::array;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task;

/// How long a write waits for another process holding the database lock.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

struct Shared {
    connection: Mutex<Connection>,
    statements: Statements,
    config: PersistentStoreConfig,
}

/// Responses state in one SQLite database file.
///
/// The state survives a restart but belongs to one host. One connection in write-ahead-log mode
/// serves every call; each call is one transaction on Tokio's blocking pool, so a dropped future
/// may still commit, but never half a write.
pub struct SqliteResponseStore {
    shared: Arc<Shared>,
    path: PathBuf,
}

impl SqliteResponseStore {
    /// Open or create the database at `path` and its tables.
    ///
    /// # Errors
    /// Returns an invalid-input error for zero limits, an unsupported error for a database
    /// created by another schema version, and a backend error when the file cannot be opened.
    pub fn open(path: &Path, config: PersistentStoreConfig) -> Result<Self> {
        config.validate()?;
        let mut connection = Connection::open(path).map_err(backend)?;
        connection.busy_timeout(BUSY_TIMEOUT).map_err(backend)?;
        connection
            .execute_batch(
                "PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; \
                 PRAGMA foreign_keys = ON;",
            )
            .map_err(backend)?;
        let transaction = connection.transaction().map_err(backend)?;
        transaction
            .execute_batch(&schema("INTEGER", "BLOB"))
            .map_err(backend)?;
        let version: Option<i64> = transaction
            .query_row(
                "SELECT value FROM infer_response_store_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(backend)?;
        match version {
            None => {
                transaction
                    .execute(
                        "INSERT INTO infer_response_store_meta (key, value) \
                         VALUES ('schema_version', ?1)",
                        [SCHEMA_VERSION],
                    )
                    .map_err(backend)?;
            }
            Some(SCHEMA_VERSION) => {}
            Some(other) => {
                return Err(Error::unsupported(format!(
                    "response store schema version {other} is not {SCHEMA_VERSION}"
                )));
            }
        }
        transaction.commit().map_err(backend)?;
        Ok(Self {
            shared: Arc::new(Shared {
                connection: Mutex::new(connection),
                statements: Statements::new(|index| format!("?{index}")),
                config,
            }),
            path: path.to_owned(),
        })
    }

    fn run<T, F>(&self, work: F) -> ResponseFuture<'_, T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection, &Statements, &PersistentStoreConfig) -> Result<T>
            + Send
            + 'static,
    {
        let shared = Arc::clone(&self.shared);
        Box::pin(async move {
            task::spawn_blocking(move || {
                let mut connection = shared
                    .connection
                    .lock()
                    .map_err(|_| Error::invariant("sqlite response store lock poisoned"))?;
                work(&mut connection, &shared.statements, &shared.config)
            })
            .await
            .map_err(|_| Error::new(ErrorCode::Backend, "sqlite response store task failed"))?
        })
    }
}

impl fmt::Debug for SqliteResponseStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteResponseStore")
            .field("path", &self.path)
            .field("config", &self.shared.config)
            .finish_non_exhaustive()
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "a `map_err` adapter receives the error by value"
)]
fn backend(error: rusqlite::Error) -> Error {
    Error::new(
        ErrorCode::Backend,
        format!("sqlite response store: {error}"),
    )
}

fn read_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ReadRow> {
    Ok(ReadRow {
        id: row.get("id")?,
        request_id: row.get("request_id")?,
        model: row.get("model")?,
        created_at: row.get("created_at")?,
        completed_at: row.get("completed_at")?,
        status: row.get("status")?,
        incomplete: row.get("incomplete")?,
        error_code: row.get("error_code")?,
        error_message: row.get("error_message")?,
        usage: [
            row.get("input_tokens")?,
            row.get("output_tokens")?,
            row.get("reasoning_tokens")?,
            row.get("cached_tokens")?,
            row.get("cache_write_tokens")?,
        ],
        previous_response_id: row.get("previous_response_id")?,
        request_input: row.get("request_input")?,
        response: row.get("response")?,
        payload_version: row.get("payload_version")?,
    })
}

fn purge(transaction: &Transaction<'_>, statements: &Statements, now: i64) -> Result<()> {
    transaction
        .execute(&statements.purge, [now])
        .map_err(backend)?;
    Ok(())
}

fn get(
    connection: &Connection,
    statements: &Statements,
    id: &ResponseId,
) -> Result<Option<StoredResponse>> {
    let row = connection
        .query_row(
            &statements.get,
            params![id.as_str(), now_millis()?],
            read_row,
        )
        .optional()
        .map_err(backend)?;
    row.map(ReadRow::into_response).transpose()
}

fn put(
    connection: &mut Connection,
    statements: &Statements,
    config: &PersistentStoreConfig,
    response: StoredResponse,
    condition: WriteCondition,
) -> Result<()> {
    response.validate()?;
    config.check_size(&response)?;
    let expires_at = config.expires_at()?;
    let row = Row::new(response)?;
    let usage = |index: usize| row.usage.map(|usage| usage[index]);
    let (statement, status) = match condition {
        WriteCondition::Always => (&statements.put, None),
        WriteCondition::UnlessStatus(status) => {
            (&statements.put_unless_status, Some(status.as_str()))
        }
        WriteCondition::UnlessTerminal => (&statements.put_unless_terminal, None),
    };
    let transaction = connection.transaction().map_err(backend)?;
    purge(&transaction, statements, now_millis()?)?;
    let usage: [Option<i64>; USAGE_COLUMNS] = array::from_fn(usage);
    let mut values: Vec<&dyn ToSql> = vec![
        &row.id,
        &row.request_id,
        &row.model,
        &row.created_at,
        &row.completed_at,
        &row.status,
        &row.incomplete,
        &row.error_code,
        &row.error_message,
        &usage[0],
        &usage[1],
        &usage[2],
        &usage[3],
        &usage[4],
        &row.previous_response_id,
        &row.request_input,
        &row.response,
        &row.payload_version,
        &expires_at,
    ];
    if let Some(status) = &status {
        values.push(status);
    }
    transaction
        .execute(statement, values.as_slice())
        .map_err(backend)?;
    transaction.commit().map_err(backend)
}

fn transition(
    connection: &mut Connection,
    statements: &Statements,
    config: &PersistentStoreConfig,
    id: &ResponseId,
    allowed: &[ResponseStatus],
    next: ResponseStatus,
) -> Result<StoredResponse> {
    let expires_at = config.expires_at()?;
    let sources = sources(allowed, next);
    let slots: [Option<&str>; TRANSITION_SLOTS] =
        array::from_fn(|index| sources.get(index).copied());
    let next_text = next.as_str();
    let id_text = id.as_str();
    let mut values: Vec<&dyn ToSql> = vec![&next_text, &expires_at, &id_text];
    values.extend(slots.iter().map(|slot| slot as &dyn ToSql));
    let transaction = connection.transaction().map_err(backend)?;
    purge(&transaction, statements, now_millis()?)?;
    let updated = transaction
        .query_row(&statements.transition, values.as_slice(), read_row)
        .optional()
        .map_err(backend)?;
    let Some(row) = updated else {
        let current: Option<String> = transaction
            .query_row(&statements.status, [id_text], |row| row.get(0))
            .optional()
            .map_err(backend)?;
        return Err(current.map_or_else(not_found, |current| transition_conflict(&current, next)));
    };
    let response = row.into_response()?;
    transaction.commit().map_err(backend)?;
    Ok(response)
}

fn append_event(
    connection: &mut Connection,
    statements: &Statements,
    config: &PersistentStoreConfig,
    id: &ResponseId,
    event: &ResponseEvent,
) -> Result<()> {
    let sequence = to_sql(event.sequence, "event sequence")?;
    let transaction = connection.transaction().map_err(backend)?;
    purge(&transaction, statements, now_millis()?)?;
    let exists = transaction
        .query_row(&statements.status, [id.as_str()], |_| Ok(()))
        .optional()
        .map_err(backend)?
        .is_some();
    if !exists {
        return Err(not_found());
    }
    let (last, count): (Option<i64>, i64) = transaction
        .query_row(&statements.event_tail, [id.as_str()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .map_err(backend)?;
    if last.is_some_and(|last| last >= sequence) {
        return Err(unordered_event());
    }
    if usize::try_from(from_sql(count, "event count")?).unwrap_or(usize::MAX) >= config.max_events {
        return Err(events_full());
    }
    transaction
        .execute(
            &statements.append_event,
            params![id.as_str(), sequence, event.payload],
        )
        .map_err(backend)?;
    transaction.commit().map_err(backend)
}

fn events(
    connection: &Connection,
    statements: &Statements,
    id: &ResponseId,
    after: Option<u64>,
) -> Result<Vec<ResponseEvent>> {
    let after = after.map_or(Ok(-1), |after| to_sql(after, "event sequence"))?;
    let mut statement = connection
        .prepare_cached(&statements.events)
        .map_err(backend)?;
    let rows = statement
        .query_map(params![id.as_str(), after, now_millis()?], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .map_err(backend)?;
    let mut events = Vec::new();
    for row in rows {
        let (sequence, payload) = row.map_err(backend)?;
        events.push(ResponseEvent {
            sequence: from_sql(sequence, "event sequence")?,
            payload,
        });
    }
    Ok(events)
}

impl ResponseStore for SqliteResponseStore {
    fn capabilities(&self) -> ResponseStoreCapabilities {
        ResponseStoreCapabilities {
            retrieve: true,
            previous_response_id: true,
            background: true,
            cancel: true,
            event_replay: true,
            persistent: true,
            shared: false,
            max_record_bytes: self.shared.config.max_record_bytes,
        }
    }

    fn spi_version(&self) -> u32 {
        RESPONSES_SPI_VERSION
    }

    fn get<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, Option<StoredResponse>> {
        let id = id.clone();
        self.run(move |connection, statements, _| get(connection, statements, &id))
    }

    fn put(&self, response: StoredResponse, condition: WriteCondition) -> ResponseFuture<'_, ()> {
        self.run(move |connection, statements, config| {
            put(connection, statements, config, response, condition)
        })
    }

    fn delete<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, ()> {
        let id = id.clone();
        self.run(move |connection, statements, _| {
            connection
                .execute(&statements.delete, [id.as_str()])
                .map_err(backend)?;
            Ok(())
        })
    }

    fn transition<'a>(
        &'a self,
        id: &'a ResponseId,
        allowed: &'a [ResponseStatus],
        next: ResponseStatus,
    ) -> ResponseFuture<'a, StoredResponse> {
        let id = id.clone();
        let allowed = allowed.to_vec();
        self.run(move |connection, statements, config| {
            transition(connection, statements, config, &id, &allowed, next)
        })
    }

    fn append_event<'a>(
        &'a self,
        id: &'a ResponseId,
        event: ResponseEvent,
    ) -> ResponseFuture<'a, ()> {
        let id = id.clone();
        self.run(move |connection, statements, config| {
            append_event(connection, statements, config, &id, &event)
        })
    }

    fn events<'a>(
        &'a self,
        id: &'a ResponseId,
        after: Option<u64>,
    ) -> ResponseFuture<'a, Vec<ResponseEvent>> {
        let id = id.clone();
        self.run(move |connection, statements, _| events(connection, statements, &id, after))
    }
}
