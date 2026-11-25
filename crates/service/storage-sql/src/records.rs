//! [`RecordStore`] over any [`SqlDatabase`]: the one place the SQL record semantics live.
use crate::database::{SqlDatabase, SqlRow, SqlTransaction, SqlValue};
use crate::schema::{Statements, meta_table, migrations};
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{
    LogRecord, Precondition, PutOutcome, Record, RecordCapabilities, RecordFuture, RecordKey,
    RecordState, RecordStore, RecordWrite, Retention, UnixMillis, Version,
};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default largest value: a 4 MiB response record and its metadata.
const DEFAULT_MAX_VALUE_BYTES: usize = 4 * 1024 * 1024 + 64 * 1024;
/// Writes between sweeps of expired records of all keys. Each write already removes its own
/// key's expired record, so the sweep only reclaims space and can be infrequent.
const SWEEP_EVERY: u64 = 1024;
/// Columns of the record read, in the order `Statements::get` selects them.
const VERSION: usize = 0;
const STATE: usize = 1;
const DISPOSABLE: usize = 2;
const EXPIRES_AT: usize = 3;
const VALUE: usize = 4;
/// Key of the schema version in `infer_storage_meta`.
const SCHEMA_VERSION_KEY: &str = "schema_version";

/// Limits of a [`SqlRecordStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqlRecordStoreConfig {
    pub max_value_bytes: usize,
}

impl Default for SqlRecordStoreConfig {
    fn default() -> Self {
        Self {
            max_value_bytes: DEFAULT_MAX_VALUE_BYTES,
        }
    }
}

/// Records in a SQL database: durable, and shared when the database is.
///
/// Every operation is one transaction that touches one key's rows, so concurrent writers of
/// different keys never wait for each other. Expired records are invisible at once and removed
/// with their key's next write, and in a periodic sweep.
pub struct SqlRecordStore<D> {
    database: D,
    statements: Statements,
    config: SqlRecordStoreConfig,
    writes: AtomicU64,
}

impl<D: SqlDatabase + fmt::Debug> fmt::Debug for SqlRecordStore<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqlRecordStore")
            .field("database", &self.database)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Finish a transaction: commit what succeeded, roll back what failed.
async fn finish<T, X: SqlTransaction>(transaction: X, result: Result<T>) -> Result<T> {
    match result {
        Ok(value) => {
            transaction.commit().await?;
            Ok(value)
        }
        Err(error) => {
            if let Err(rollback) = transaction.rollback().await {
                tracing::warn!(target: "infer::storage", error = %rollback, "record transaction rollback failed");
            }
            Err(error)
        }
    }
}

fn now() -> Result<i64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::invariant("system clock is before the Unix epoch"))?;
    i64::try_from(elapsed.as_millis()).map_err(|_| Error::invariant("system clock overflow"))
}

fn signed(value: u64, what: &str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::invalid(format!("{what} {value} is too large to store")))
}

fn unsigned(value: Option<i64>, what: &str) -> Result<u64> {
    value
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| Error::invariant(format!("stored {what} is missing or negative")))
}

/// The `(disposable, expires_at)` columns of a retention class.
fn retention_columns(retention: Retention) -> Result<(i64, Option<i64>)> {
    match retention {
        Retention::Pinned => Ok((0, None)),
        Retention::Disposable(expires) => Ok((
            1,
            expires.map(|at| signed(at.get(), "deadline")).transpose()?,
        )),
    }
}

fn retention_of(disposable: Option<i64>, expires: Option<i64>) -> Result<Retention> {
    match (disposable, expires) {
        (Some(0), None) => Ok(Retention::Pinned),
        (Some(1), expires) => Ok(Retention::Disposable(
            expires
                .map(|at| unsigned(Some(at), "deadline").map(UnixMillis::new))
                .transpose()?,
        )),
        _ => Err(Error::invariant("stored record has an invalid retention")),
    }
}

fn state_of(value: Option<i64>) -> Result<RecordState> {
    let label = u8::try_from(unsigned(value, "state")?)
        .map_err(|_| Error::invariant("stored record state is out of range"))?;
    RecordState::new(label)
}

fn version_and_state(mut row: SqlRow) -> Result<(Version, RecordState)> {
    Ok((
        Version::new(unsigned(row.integer(0)?, "version")?),
        state_of(row.integer(1)?)?,
    ))
}

impl<D: SqlDatabase> SqlRecordStore<D> {
    /// Use `database`, creating or migrating its tables.
    ///
    /// # Errors
    /// Returns an invalid-input error for a zero value limit, an unsupported error for a
    /// database migrated by a newer version of this store, and the database's errors.
    pub async fn open(database: D, config: SqlRecordStoreConfig) -> Result<Self> {
        if config.max_value_bytes == 0 {
            return Err(Error::invalid("record store value limit must be nonzero"));
        }
        let statements = Statements::new(&database.dialect());
        let store = Self {
            database,
            statements,
            config,
            writes: AtomicU64::new(0),
        };
        let mut transaction = store.database.begin().await?;
        let result = store.migrate(&mut transaction).await;
        finish(transaction, result).await?;
        Ok(store)
    }

    async fn migrate(&self, transaction: &mut D::Transaction<'_>) -> Result<()> {
        let dialect = self.database.dialect();
        if let Some(lock) = dialect.migration_lock {
            transaction.execute(lock, &[]).await?;
        }
        transaction.batch(&meta_table(&dialect)).await?;
        let rows = transaction
            .query(
                &self.statements.schema_version,
                &[SqlValue::Text(SCHEMA_VERSION_KEY)],
            )
            .await?;
        let applied = match rows.into_iter().next() {
            Some(mut row) => usize::try_from(unsigned(row.integer(0)?, "schema version")?)
                .map_err(|_| Error::invariant("stored schema version overflow"))?,
            None => 0,
        };
        let migrations = migrations(&dialect);
        if applied > migrations.len() {
            return Err(Error::unsupported(format!(
                "record store schema version {applied} is newer than {}",
                migrations.len()
            )));
        }
        for migration in &migrations[applied..] {
            transaction.batch(migration).await?;
        }
        if applied < migrations.len() {
            let latest = i64::try_from(migrations.len())
                .map_err(|_| Error::invariant("schema version overflow"))?;
            transaction
                .execute(
                    &self.statements.set_schema_version,
                    &[
                        SqlValue::Text(SCHEMA_VERSION_KEY),
                        SqlValue::Integer(latest),
                    ],
                )
                .await?;
        }
        Ok(())
    }

    async fn current(
        &self,
        transaction: &mut D::Transaction<'_>,
        key: &RecordKey,
    ) -> Result<Option<(Version, RecordState)>> {
        let rows = transaction
            .query(&self.statements.current, &key_values(key))
            .await?;
        rows.into_iter().next().map(version_and_state).transpose()
    }

    /// Remove `key`'s record if it has expired, so the write sees what a read would.
    async fn purge_key(
        &self,
        transaction: &mut D::Transaction<'_>,
        key: &RecordKey,
        now: i64,
    ) -> Result<()> {
        let [namespace, id] = key_values(key);
        transaction
            .execute(
                &self.statements.purge_key,
                &[namespace, id, SqlValue::Integer(now)],
            )
            .await
            .map(drop)
    }

    async fn get_in(
        &self,
        transaction: &mut D::Transaction<'_>,
        key: &RecordKey,
    ) -> Result<Option<Record>> {
        let [namespace, id] = key_values(key);
        let rows = transaction
            .query(
                &self.statements.get,
                &[namespace, id, SqlValue::Integer(now()?)],
            )
            .await?;
        let Some(mut row) = rows.into_iter().next() else {
            return Ok(None);
        };
        Ok(Some(Record {
            version: Version::new(unsigned(row.integer(VERSION)?, "version")?),
            state: state_of(row.integer(STATE)?)?,
            retention: retention_of(row.integer(DISPOSABLE)?, row.integer(EXPIRES_AT)?)?,
            value: row
                .bytes(VALUE)?
                .ok_or_else(|| Error::invariant("stored record has no value"))?,
        }))
    }

    async fn put_in(
        &self,
        transaction: &mut D::Transaction<'_>,
        key: &RecordKey,
        write: RecordWrite<'_>,
    ) -> Result<PutOutcome> {
        self.purge_key(transaction, key, now()?).await?;
        let mut versions = transaction.query(self.statements.next_version, &[]).await?;
        let version = versions
            .first_mut()
            .map(|row| row.integer(0))
            .transpose()?
            .flatten()
            .ok_or_else(|| Error::invariant("the version source returned no version"))?;
        let (disposable, expires) = retention_columns(write.retention)?;
        let [namespace, id] = key_values(key);
        let columns = [
            SqlValue::Integer(version),
            SqlValue::Integer(i64::from(write.state.get())),
            SqlValue::Integer(signed(write.state.bit(), "state bit")?),
            SqlValue::Integer(disposable),
            expires.into(),
            SqlValue::Bytes(write.value),
        ];
        let changed = match write.condition {
            Precondition::Any => {
                let values = [namespace, id]
                    .into_iter()
                    .chain(columns)
                    .collect::<Vec<_>>();
                transaction
                    .execute(&self.statements.upsert, &values)
                    .await?
            }
            Precondition::Absent => {
                let values = [namespace, id]
                    .into_iter()
                    .chain(columns)
                    .collect::<Vec<_>>();
                transaction
                    .execute(&self.statements.insert_absent, &values)
                    .await?
            }
            Precondition::StateNotIn(excluded) => {
                let mask = SqlValue::Integer(signed(excluded.bits(), "state set")?);
                let values = [namespace, id]
                    .into_iter()
                    .chain(columns)
                    .chain([mask])
                    .collect::<Vec<_>>();
                transaction
                    .execute(&self.statements.upsert_unless_state, &values)
                    .await?
            }
            Precondition::Version(expected) => {
                let expected = SqlValue::Integer(signed(expected.get(), "version")?);
                let values = columns
                    .into_iter()
                    .chain([namespace, id, expected])
                    .collect::<Vec<_>>();
                transaction
                    .execute(&self.statements.update_version, &values)
                    .await?
            }
        };
        if changed == 0 {
            return Ok(PutOutcome::Rejected(self.current(transaction, key).await?));
        }
        let version = u64::try_from(version)
            .map_err(|_| Error::invariant("the version source returned a negative version"))?;
        Ok(PutOutcome::Written(Version::new(version)))
    }

    async fn append_in(
        &self,
        transaction: &mut D::Transaction<'_>,
        key: &RecordKey,
        sequence: u64,
        value: &[u8],
        max_entries: usize,
    ) -> Result<()> {
        let sequence = signed(sequence, "log sequence")?;
        self.purge_key(transaction, key, now()?).await?;
        // Reading the record locks it where the dialect needs a lock, which orders appends.
        if self.current(transaction, key).await?.is_none() {
            return Err(Error::new(
                ErrorCode::NotFound,
                format!("record {key} not found"),
            ));
        }
        let mut tail = transaction
            .query(&self.statements.log_tail, &key_values(key))
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invariant("log tail query returned no row"))?;
        let last = tail.integer(0)?;
        let count = unsigned(tail.integer(1)?, "log length")?;
        if last.is_some_and(|last| last >= sequence) {
            return Err(Error::invalid(format!(
                "log sequence {sequence} of {key} does not follow the last entry"
            )));
        }
        if usize::try_from(count).unwrap_or(usize::MAX) >= max_entries {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!("log of {key} is full"),
            ));
        }
        let [namespace, id] = key_values(key);
        transaction
            .execute(
                &self.statements.append,
                &[
                    namespace,
                    id,
                    SqlValue::Integer(sequence),
                    SqlValue::Bytes(value),
                ],
            )
            .await
            .map(drop)
    }

    async fn log_in(
        &self,
        transaction: &mut D::Transaction<'_>,
        key: &RecordKey,
        after: Option<u64>,
    ) -> Result<Vec<LogRecord>> {
        let after = after.map_or(Ok(-1), |after| signed(after, "log sequence"))?;
        let [namespace, id] = key_values(key);
        let rows = transaction
            .query(
                &self.statements.log,
                &[
                    namespace,
                    id,
                    SqlValue::Integer(after),
                    SqlValue::Integer(now()?),
                ],
            )
            .await?;
        rows.into_iter()
            .map(|mut row| {
                Ok(LogRecord {
                    sequence: unsigned(row.integer(0)?, "log sequence")?,
                    value: row
                        .bytes(1)?
                        .ok_or_else(|| Error::invariant("stored log entry has no value"))?,
                })
            })
            .collect()
    }

    /// Every [`SWEEP_EVERY`] writes, remove expired records of all keys in a separate
    /// transaction. Best effort: a failure only leaves the space for the next sweep.
    async fn sweep_if_due(&self) {
        if self.writes.fetch_add(1, Ordering::Relaxed) % SWEEP_EVERY != SWEEP_EVERY - 1 {
            return;
        }
        let result = async {
            let mut transaction = self.database.begin().await?;
            let now = now()?;
            let result = transaction
                .execute(&self.statements.purge, &[SqlValue::Integer(now)])
                .await;
            finish(transaction, result).await
        };
        if let Err(error) = result.await {
            tracing::warn!(target: "infer::storage", %error, "expired record sweep failed");
        }
    }

    async fn transact<T>(
        &self,
        work: impl AsyncFnOnce(&Self, &mut D::Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut transaction = self.database.begin().await?;
        let result = work(self, &mut transaction).await;
        finish(transaction, result).await
    }
}

fn key_values(key: &RecordKey) -> [SqlValue<'_>; 2] {
    [
        SqlValue::Text(key.namespace().as_str()),
        SqlValue::Text(key.id()),
    ]
}

impl<D: SqlDatabase> RecordStore for SqlRecordStore<D> {
    fn capabilities(&self) -> RecordCapabilities {
        RecordCapabilities {
            durable: true,
            shared: self.database.shared(),
            evicts_disposable: false,
            state_predicates: true,
            max_value_bytes: self.config.max_value_bytes,
        }
    }

    fn get<'a>(&'a self, key: &'a RecordKey) -> RecordFuture<'a, Option<Record>> {
        Box::pin(
            self.transact(async move |store, transaction| store.get_in(transaction, key).await),
        )
    }

    fn put<'a>(
        &'a self,
        key: &'a RecordKey,
        write: RecordWrite<'a>,
    ) -> RecordFuture<'a, PutOutcome> {
        Box::pin(async move {
            if write.value.len() > self.config.max_value_bytes {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "record value exceeds the store's value limit",
                ));
            }
            let outcome = self
                .transact(async |store, transaction| store.put_in(transaction, key, write).await)
                .await?;
            self.sweep_if_due().await;
            Ok(outcome)
        })
    }

    fn delete<'a>(&'a self, key: &'a RecordKey) -> RecordFuture<'a, ()> {
        Box::pin(self.transact(async move |store, transaction| {
            transaction
                .execute(&store.statements.delete, &key_values(key))
                .await
                .map(drop)
        }))
    }

    fn append<'a>(
        &'a self,
        key: &'a RecordKey,
        sequence: u64,
        value: &'a [u8],
        max_entries: usize,
    ) -> RecordFuture<'a, ()> {
        Box::pin(self.transact(async move |store, transaction| {
            store
                .append_in(transaction, key, sequence, value, max_entries)
                .await
        }))
    }

    fn log<'a>(
        &'a self,
        key: &'a RecordKey,
        after: Option<u64>,
    ) -> RecordFuture<'a, Vec<LogRecord>> {
        Box::pin(
            self.transact(async move |store, transaction| {
                store.log_in(transaction, key, after).await
            }),
        )
    }
}
