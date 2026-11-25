//! Responses state in a PostgreSQL database, shared by every replica that connects to it.
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
use std::array;
use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_postgres::config::SslMode;
use tokio_postgres::types::ToSql;
use tokio_postgres::{Client, Config, NoTls, Transaction};

/// Connections one store keeps open; calls beyond this wait for a free connection.
const CONNECTIONS: usize = 4;
/// Key of the advisory lock that serializes table creation across replicas.
const SCHEMA_LOCK: i64 = 0x696e_6665_725f_7273;

type Parameter<'a> = &'a (dyn ToSql + Sync);

/// Responses state in one PostgreSQL database.
///
/// The state survives restarts and is shared by every replica using the same database; status
/// transitions and conditional writes are atomic across them. At most four connections are
/// open; each call is one transaction, so a dropped future rolls back or commits whole.
/// Connections are unencrypted: a URL that requires TLS is refused.
pub struct PostgresResponseStore {
    config: Config,
    statements: Statements,
    limits: PersistentStoreConfig,
    idle: Mutex<Vec<Client>>,
    permits: Arc<Semaphore>,
}

/// One connection borrowed from the store; returned to it when dropped while still open.
struct Lease<'a> {
    store: &'a PostgresResponseStore,
    client: Option<Client>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        // A closed connection, or a pool whose lock is poisoned, drops the client instead.
        if let Some(client) = self.client.take()
            && !client.is_closed()
            && let Ok(mut idle) = self.store.idle.lock()
        {
            idle.push(client);
        }
    }
}

impl Lease<'_> {
    fn client(&mut self) -> Result<&mut Client> {
        self.client
            .as_mut()
            .ok_or_else(|| Error::invariant("postgres connection lease is empty"))
    }
}

impl PostgresResponseStore {
    /// Connect to the database named by a `postgres://` URL or key-value connection string and
    /// create the tables when missing.
    ///
    /// # Errors
    /// Returns an invalid-input error for a malformed URL or zero limits, an unsupported error for
    /// a URL that requires TLS or a database created by another schema version, and a backend
    /// error when the database cannot be reached.
    pub async fn connect(url: &str, limits: PersistentStoreConfig) -> Result<Self> {
        limits.validate()?;
        let config = Config::from_str(url)
            .map_err(|error| Error::invalid(format!("invalid postgres URL: {error}")))?;
        if !matches!(config.get_ssl_mode(), SslMode::Disable | SslMode::Prefer) {
            return Err(Error::unsupported(
                "postgres response store does not support TLS yet; use sslmode=disable or prefer",
            ));
        }
        let store = Self {
            config,
            statements: Statements::new(|index| format!("${index}")),
            limits,
            idle: Mutex::new(Vec::new()),
            permits: Arc::new(Semaphore::new(CONNECTIONS)),
        };
        let mut lease = store.lease().await?;
        let transaction = lease.client()?.transaction().await.map_err(backend)?;
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&SCHEMA_LOCK])
            .await
            .map_err(backend)?;
        transaction
            .batch_execute(&schema("BIGINT", "BYTEA"))
            .await
            .map_err(backend)?;
        let version = transaction
            .query_opt(
                "SELECT value FROM infer_response_store_meta WHERE key = 'schema_version'",
                &[],
            )
            .await
            .map_err(backend)?
            .map(|row| row.try_get::<_, i64>(0))
            .transpose()
            .map_err(backend)?;
        match version {
            None => {
                transaction
                    .execute(
                        "INSERT INTO infer_response_store_meta (key, value) \
                         VALUES ('schema_version', $1)",
                        &[&SCHEMA_VERSION],
                    )
                    .await
                    .map_err(backend)?;
            }
            Some(SCHEMA_VERSION) => {}
            Some(other) => {
                return Err(Error::unsupported(format!(
                    "response store schema version {other} is not {SCHEMA_VERSION}"
                )));
            }
        }
        transaction.commit().await.map_err(backend)?;
        drop(lease);
        Ok(store)
    }

    async fn lease(&self) -> Result<Lease<'_>> {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| Error::invariant("postgres connection pool closed"))?;
        let idle = self
            .idle
            .lock()
            .map_err(|_| Error::invariant("postgres connection pool lock poisoned"))?
            .pop();
        let client = match idle {
            Some(client) if !client.is_closed() => client,
            _ => self.open().await?,
        };
        Ok(Lease {
            store: self,
            client: Some(client),
            _permit: permit,
        })
    }

    async fn open(&self) -> Result<Client> {
        let (client, connection) = self.config.connect(NoTls).await.map_err(backend)?;
        // The connection future drives the socket until its client is dropped; the permit bound
        // keeps the number of these tasks at the pool size.
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                tracing::warn!(target: "infer::response_store", %error, "postgres connection closed");
            }
        });
        Ok(client)
    }

    async fn get_response(&self, id: &ResponseId) -> Result<Option<StoredResponse>> {
        let mut lease = self.lease().await?;
        let now = now_millis()?;
        let row = lease
            .client()?
            .query_opt(&self.statements.get, &[&id.as_str(), &now])
            .await
            .map_err(backend)?;
        drop(lease);
        row.map(|row| read_row(&row)?.into_response()).transpose()
    }

    async fn put_response(
        &self,
        response: StoredResponse,
        condition: WriteCondition,
    ) -> Result<()> {
        response.validate()?;
        self.limits.check_size(&response)?;
        let expires_at = self.limits.expires_at()?;
        let row = Row::new(response)?;
        let usage: [Option<i64>; USAGE_COLUMNS] =
            array::from_fn(|index| row.usage.map(|usage| usage[index]));
        let (statement, status) = match condition {
            WriteCondition::Always => (&self.statements.put, None),
            WriteCondition::UnlessStatus(status) => {
                (&self.statements.put_unless_status, Some(status.as_str()))
            }
            WriteCondition::UnlessTerminal => (&self.statements.put_unless_terminal, None),
        };
        let mut values: Vec<Parameter<'_>> = vec![
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
        let mut lease = self.lease().await?;
        let transaction = lease.client()?.transaction().await.map_err(backend)?;
        self.purge(&transaction).await?;
        transaction
            .execute(statement, &values)
            .await
            .map_err(backend)?;
        transaction.commit().await.map_err(backend)?;
        drop(lease);
        Ok(())
    }

    async fn purge(&self, transaction: &Transaction<'_>) -> Result<()> {
        let now = now_millis()?;
        transaction
            .execute(&self.statements.purge, &[&now])
            .await
            .map_err(backend)?;
        Ok(())
    }

    async fn transition_response(
        &self,
        id: &ResponseId,
        allowed: &[ResponseStatus],
        next: ResponseStatus,
    ) -> Result<StoredResponse> {
        let expires_at = self.limits.expires_at()?;
        let sources = sources(allowed, next);
        let slots: [Option<&str>; TRANSITION_SLOTS] =
            array::from_fn(|index| sources.get(index).copied());
        let next_text = next.as_str();
        let id_text = id.as_str();
        let mut values: Vec<Parameter<'_>> = vec![&next_text, &expires_at, &id_text];
        values.extend(slots.iter().map(|slot| slot as Parameter<'_>));
        let mut lease = self.lease().await?;
        let transaction = lease.client()?.transaction().await.map_err(backend)?;
        self.purge(&transaction).await?;
        let updated = transaction
            .query_opt(&self.statements.transition, &values)
            .await
            .map_err(backend)?;
        let Some(row) = updated else {
            let current = transaction
                .query_opt(&self.statements.status, &[&id_text])
                .await
                .map_err(backend)?
                .map(|row| row.try_get::<_, String>(0))
                .transpose()
                .map_err(backend)?;
            return Err(
                current.map_or_else(not_found, |current| transition_conflict(&current, next))
            );
        };
        transaction.commit().await.map_err(backend)?;
        drop(lease);
        read_row(&row)?.into_response()
    }

    async fn append(&self, id: &ResponseId, event: ResponseEvent) -> Result<()> {
        let sequence = to_sql(event.sequence, "event sequence")?;
        let id_text = id.as_str();
        let mut lease = self.lease().await?;
        let transaction = lease.client()?.transaction().await.map_err(backend)?;
        self.purge(&transaction).await?;
        self.check_append(&transaction, id_text, sequence).await?;
        transaction
            .execute(
                &self.statements.append_event,
                &[&id_text, &sequence, &event.payload],
            )
            .await
            .map_err(backend)?;
        transaction.commit().await.map_err(backend)?;
        drop(lease);
        Ok(())
    }

    /// Lock the response row, which serializes appends to it across connections and replicas,
    /// then check that `sequence` may follow its events.
    async fn check_append(
        &self,
        transaction: &Transaction<'_>,
        id: &str,
        sequence: i64,
    ) -> Result<()> {
        if transaction
            .query_opt(&self.statements.status_locked, &[&id])
            .await
            .map_err(backend)?
            .is_none()
        {
            return Err(not_found());
        }
        let tail = transaction
            .query_one(&self.statements.event_tail, &[&id])
            .await
            .map_err(backend)?;
        let last: Option<i64> = tail.try_get(0).map_err(backend)?;
        let count: i64 = tail.try_get(1).map_err(backend)?;
        if last.is_some_and(|last| last >= sequence) {
            return Err(unordered_event());
        }
        if usize::try_from(from_sql(count, "event count")?).unwrap_or(usize::MAX)
            >= self.limits.max_events
        {
            return Err(events_full());
        }
        Ok(())
    }

    async fn read_events(&self, id: &ResponseId, after: Option<u64>) -> Result<Vec<ResponseEvent>> {
        let after = after.map_or(Ok(-1), |after| to_sql(after, "event sequence"))?;
        let now = now_millis()?;
        let mut lease = self.lease().await?;
        let rows = lease
            .client()?
            .query(&self.statements.events, &[&id.as_str(), &after, &now])
            .await
            .map_err(backend)?;
        drop(lease);
        rows.iter()
            .map(|row| {
                Ok(ResponseEvent {
                    sequence: from_sql(row.try_get(0).map_err(backend)?, "event sequence")?,
                    payload: row.try_get(1).map_err(backend)?,
                })
            })
            .collect()
    }

    async fn delete_response(&self, id: &ResponseId) -> Result<()> {
        let mut lease = self.lease().await?;
        lease
            .client()?
            .execute(&self.statements.delete, &[&id.as_str()])
            .await
            .map_err(backend)?;
        drop(lease);
        Ok(())
    }
}

impl fmt::Debug for PostgresResponseStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The connection config holds the password; show only where the store points.
        formatter
            .debug_struct("PostgresResponseStore")
            .field("hosts", &self.config.get_hosts())
            .field("dbname", &self.config.get_dbname())
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "a `map_err` adapter receives the error by value"
)]
fn backend(error: tokio_postgres::Error) -> Error {
    Error::new(
        ErrorCode::Backend,
        format!("postgres response store: {error}"),
    )
}

fn read_row(row: &tokio_postgres::Row) -> Result<ReadRow> {
    let column = |name: &str| row.try_get::<_, Option<i64>>(name).map_err(backend);
    Ok(ReadRow {
        id: row.try_get("id").map_err(backend)?,
        request_id: row.try_get("request_id").map_err(backend)?,
        model: row.try_get("model").map_err(backend)?,
        created_at: row.try_get("created_at").map_err(backend)?,
        completed_at: row.try_get("completed_at").map_err(backend)?,
        status: row.try_get("status").map_err(backend)?,
        incomplete: row.try_get("incomplete").map_err(backend)?,
        error_code: row.try_get("error_code").map_err(backend)?,
        error_message: row.try_get("error_message").map_err(backend)?,
        usage: [
            column("input_tokens")?,
            column("output_tokens")?,
            column("reasoning_tokens")?,
            column("cached_tokens")?,
            column("cache_write_tokens")?,
        ],
        previous_response_id: row.try_get("previous_response_id").map_err(backend)?,
        request_input: row.try_get("request_input").map_err(backend)?,
        response: row.try_get("response").map_err(backend)?,
        payload_version: row.try_get("payload_version").map_err(backend)?,
    })
}

impl ResponseStore for PostgresResponseStore {
    fn capabilities(&self) -> ResponseStoreCapabilities {
        ResponseStoreCapabilities {
            retrieve: true,
            previous_response_id: true,
            background: true,
            cancel: true,
            event_replay: true,
            persistent: true,
            shared: true,
            max_record_bytes: self.limits.max_record_bytes,
        }
    }

    fn spi_version(&self) -> u32 {
        RESPONSES_SPI_VERSION
    }

    fn get<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, Option<StoredResponse>> {
        Box::pin(self.get_response(id))
    }

    fn put(&self, response: StoredResponse, condition: WriteCondition) -> ResponseFuture<'_, ()> {
        Box::pin(self.put_response(response, condition))
    }

    fn delete<'a>(&'a self, id: &'a ResponseId) -> ResponseFuture<'a, ()> {
        Box::pin(self.delete_response(id))
    }

    fn transition<'a>(
        &'a self,
        id: &'a ResponseId,
        allowed: &'a [ResponseStatus],
        next: ResponseStatus,
    ) -> ResponseFuture<'a, StoredResponse> {
        Box::pin(self.transition_response(id, allowed, next))
    }

    fn append_event<'a>(
        &'a self,
        id: &'a ResponseId,
        event: ResponseEvent,
    ) -> ResponseFuture<'a, ()> {
        Box::pin(self.append(id, event))
    }

    fn events<'a>(
        &'a self,
        id: &'a ResponseId,
        after: Option<u64>,
    ) -> ResponseFuture<'a, Vec<ResponseEvent>> {
        Box::pin(self.read_events(id, after))
    }
}
