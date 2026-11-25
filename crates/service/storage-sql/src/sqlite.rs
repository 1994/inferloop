//! A local SQLite database file, served by one connection on its own thread.
use crate::database::{Dialect, SqlCell, SqlDatabase, SqlRow, SqlTransaction, SqlValue};
use infer_core::{Error, ErrorCode, Result};
use rusqlite::types::{Value, ValueRef};
use rusqlite::{Connection, params_from_iter};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::from_utf8;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread;
use std::time::Duration;
use tokio::sync::{Semaphore, SemaphorePermit, oneshot};

/// How long a write waits for another process holding the database lock.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Commands queued for the connection thread. One transaction runs at a time and waits for
/// each reply, so the queue holds at most its next command and a rollback.
const COMMAND_QUEUE: usize = 4;

type Reply<T> = oneshot::Sender<Result<T>>;

enum Command {
    Begin(Reply<()>),
    Execute(String, Vec<Value>, Reply<u64>),
    Query(String, Vec<Value>, Reply<Vec<SqlRow>>),
    Batch(String, Reply<()>),
    Commit(Reply<()>),
    Rollback(Option<Reply<()>>),
}

/// A SQLite database file, created when missing.
///
/// One connection in write-ahead-log mode lives on a dedicated thread, so no SQLite call blocks
/// an async service thread; transactions take turns on it. The file is durable but belongs to
/// one host.
pub struct Sqlite {
    commands: SyncSender<Command>,
    turn: Semaphore,
    path: PathBuf,
}

impl fmt::Debug for Sqlite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Sqlite")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl Sqlite {
    /// Open or create the database at `path` and start its connection thread.
    ///
    /// # Errors
    /// Returns a backend error when the file cannot be opened or configured.
    pub fn open(path: &Path) -> Result<Self> {
        let connection = Connection::open(path).map_err(backend)?;
        connection.busy_timeout(BUSY_TIMEOUT).map_err(backend)?;
        connection
            .execute_batch(
                "PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;",
            )
            .map_err(backend)?;
        let (commands, receiver) = sync_channel(COMMAND_QUEUE);
        thread::Builder::new()
            .name("infer-sqlite".to_owned())
            .spawn(move || serve(&connection, &receiver))
            .map_err(|error| Error::new(ErrorCode::Backend, format!("sqlite thread: {error}")))?;
        Ok(Self {
            commands,
            turn: Semaphore::new(1),
            path: path.to_owned(),
        })
    }

    fn send(&self, command: Command) -> Result<()> {
        self.commands
            .try_send(command)
            .map_err(|error| match error {
                TrySendError::Full(_) => Error::invariant("sqlite command queue is full"),
                TrySendError::Disconnected(_) => {
                    Error::new(ErrorCode::Backend, "sqlite connection thread stopped")
                }
            })
    }
}

/// Run commands until every handle is dropped. The thread then closes the connection.
fn serve(connection: &Connection, commands: &Receiver<Command>) {
    while let Ok(command) = commands.recv() {
        // A reply nobody waits for any more belongs to a cancelled call; dropping it is right.
        match command {
            Command::Begin(reply) => drop(reply.send(batch(connection, "BEGIN IMMEDIATE"))),
            Command::Execute(statement, values, reply) => {
                drop(reply.send(execute(connection, &statement, values)));
            }
            Command::Query(statement, values, reply) => {
                drop(reply.send(query(connection, &statement, values)));
            }
            Command::Batch(statements, reply) => drop(reply.send(batch(connection, &statements))),
            Command::Commit(reply) => drop(reply.send(batch(connection, "COMMIT"))),
            Command::Rollback(reply) => {
                let result = batch(connection, "ROLLBACK");
                if let Some(reply) = reply {
                    drop(reply.send(result));
                } else if let Err(error) = result {
                    tracing::warn!(target: "infer::storage", %error, "sqlite rollback after a dropped transaction failed");
                }
            }
        }
    }
}

fn batch(connection: &Connection, statements: &str) -> Result<()> {
    connection.execute_batch(statements).map_err(backend)
}

fn execute(connection: &Connection, statement: &str, values: Vec<Value>) -> Result<u64> {
    let mut statement = connection.prepare_cached(statement).map_err(backend)?;
    let changed = statement
        .execute(params_from_iter(values))
        .map_err(backend)?;
    u64::try_from(changed).map_err(|_| Error::invariant("sqlite changed-row count overflow"))
}

fn query(connection: &Connection, statement: &str, values: Vec<Value>) -> Result<Vec<SqlRow>> {
    let mut statement = connection.prepare_cached(statement).map_err(backend)?;
    let columns = statement.column_count();
    let mut rows = statement.query(params_from_iter(values)).map_err(backend)?;
    let mut collected = Vec::new();
    while let Some(row) = rows.next().map_err(backend)? {
        let cells = (0..columns)
            .map(|index| match row.get_ref(index).map_err(backend)? {
                ValueRef::Null => Ok(SqlCell::Null),
                ValueRef::Integer(value) => Ok(SqlCell::Integer(value)),
                ValueRef::Text(text) => from_utf8(text)
                    .map(|text| SqlCell::Text(text.to_owned()))
                    .map_err(|_| Error::invariant("sqlite returned text that is not UTF-8")),
                ValueRef::Blob(bytes) => Ok(SqlCell::Bytes(bytes.to_vec())),
                ValueRef::Real(_) => {
                    Err(Error::invariant("sqlite returned a floating-point column"))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        collected.push(SqlRow(cells));
    }
    Ok(collected)
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "a `map_err` adapter receives the error by value"
)]
fn backend(error: rusqlite::Error) -> Error {
    Error::new(ErrorCode::Backend, format!("sqlite: {error}"))
}

fn owned(values: &[SqlValue<'_>]) -> Vec<Value> {
    values
        .iter()
        .map(|value| match *value {
            SqlValue::Null => Value::Null,
            SqlValue::Integer(value) => Value::Integer(value),
            SqlValue::Text(text) => Value::Text(text.to_owned()),
            SqlValue::Bytes(bytes) => Value::Blob(bytes.to_vec()),
        })
        .collect()
}

/// The connection's current transaction; it holds the turn until it ends.
pub struct SqliteTransaction<'a> {
    database: &'a Sqlite,
    _turn: SemaphorePermit<'a>,
    open: bool,
}

impl SqliteTransaction<'_> {
    async fn call<T>(&self, command: impl FnOnce(Reply<T>) -> Command) -> Result<T> {
        let (reply, response) = oneshot::channel();
        self.database.send(command(reply))?;
        response
            .await
            .map_err(|_| Error::new(ErrorCode::Backend, "sqlite connection thread stopped"))?
    }
}

impl Drop for SqliteTransaction<'_> {
    fn drop(&mut self) {
        // The next transaction queues behind this rollback, so it starts from a clean state.
        if self.open
            && let Err(error) = self.database.send(Command::Rollback(None))
        {
            tracing::warn!(target: "infer::storage", %error, "sqlite rollback of a dropped transaction was not queued");
        }
    }
}

impl SqlTransaction for SqliteTransaction<'_> {
    async fn execute(&mut self, statement: &str, parameters: &[SqlValue<'_>]) -> Result<u64> {
        let (statement, values) = (statement.to_owned(), owned(parameters));
        self.call(|reply| Command::Execute(statement, values, reply))
            .await
    }

    async fn query(&mut self, statement: &str, parameters: &[SqlValue<'_>]) -> Result<Vec<SqlRow>> {
        let (statement, values) = (statement.to_owned(), owned(parameters));
        self.call(|reply| Command::Query(statement, values, reply))
            .await
    }

    async fn batch(&mut self, statements: &str) -> Result<()> {
        let statements = statements.to_owned();
        self.call(|reply| Command::Batch(statements, reply)).await
    }

    async fn commit(mut self) -> Result<()> {
        let result = self.call(Command::Commit).await;
        self.open = result.is_err();
        result
    }

    async fn rollback(mut self) -> Result<()> {
        self.open = false;
        self.call(|reply| Command::Rollback(Some(reply))).await
    }
}

impl SqlDatabase for Sqlite {
    type Transaction<'a> = SqliteTransaction<'a>;

    fn dialect(&self) -> Dialect {
        Dialect {
            placeholder: |index| format!("?{index}"),
            integer: "INTEGER",
            bytes: "BLOB",
            key: "TEXT",
            row_lock: "",
            migration_lock: None,
            version_source: "INSERT INTO infer_storage_meta (key, value) VALUES ('last_version', 0) \
                             ON CONFLICT (key) DO NOTHING",
            next_version: "UPDATE infer_storage_meta SET value = value + 1 \
                           WHERE key = 'last_version' RETURNING value",
        }
    }

    fn shared(&self) -> bool {
        false
    }

    async fn begin(&self) -> Result<SqliteTransaction<'_>> {
        let turn = self
            .turn
            .acquire()
            .await
            .map_err(|_| Error::invariant("sqlite transaction turn closed"))?;
        // Open from the moment BEGIN is queued: if this future is dropped before the reply, the
        // drop still queues the rollback that ends whatever BEGIN started.
        let mut transaction = SqliteTransaction {
            database: self,
            _turn: turn,
            open: true,
        };
        if let Err(error) = transaction.call(Command::Begin).await {
            transaction.open = false;
            return Err(error);
        }
        Ok(transaction)
    }
}
