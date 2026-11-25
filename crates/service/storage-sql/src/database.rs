//! The execution contract a SQL database implements for [`crate::SqlRecordStore`].
//!
//! The record semantics (conditions, versions, retention, logs, migrations) live once in the
//! store; a database only opens transactions, runs the store's statements with typed values and
//! returns rows. Adding a SQL database is a [`Dialect`] plus these two traits.
use infer_core::{Error, Result};
use std::future::Future;
use std::mem;

/// One bound statement parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlValue<'a> {
    Null,
    Integer(i64),
    Text(&'a str),
    Bytes(&'a [u8]),
}

impl From<Option<i64>> for SqlValue<'_> {
    fn from(value: Option<i64>) -> Self {
        value.map_or(Self::Null, SqlValue::Integer)
    }
}

/// One column value read back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SqlCell {
    Null,
    Integer(i64),
    Text(String),
    Bytes(Vec<u8>),
}

/// One result row, in the order of the statement's select list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SqlRow(pub Vec<SqlCell>);

impl SqlRow {
    fn take(&mut self, index: usize) -> Result<SqlCell> {
        self.0
            .get_mut(index)
            .map(|cell| mem::replace(cell, SqlCell::Null))
            .ok_or_else(|| Error::invariant(format!("result row has no column {index}")))
    }

    /// Take an integer column; SQL `NULL` is `None`.
    ///
    /// # Errors
    /// Returns an invariant error for a missing column or one of another type.
    pub fn integer(&mut self, index: usize) -> Result<Option<i64>> {
        match self.take(index)? {
            SqlCell::Null => Ok(None),
            SqlCell::Integer(value) => Ok(Some(value)),
            other => Err(mismatch(index, "an integer", &other)),
        }
    }

    /// Take a byte column; SQL `NULL` is `None`.
    ///
    /// # Errors
    /// Returns an invariant error for a missing column or one of another type.
    pub fn bytes(&mut self, index: usize) -> Result<Option<Vec<u8>>> {
        match self.take(index)? {
            SqlCell::Null => Ok(None),
            SqlCell::Bytes(value) => Ok(Some(value)),
            other => Err(mismatch(index, "bytes", &other)),
        }
    }
}

fn mismatch(index: usize, expected: &str, found: &SqlCell) -> Error {
    Error::invariant(format!(
        "result column {index} is {found:?}, expected {expected}"
    ))
}

/// What differs between SQL dialects, as the store needs it.
#[derive(Debug, Clone, Copy)]
pub struct Dialect {
    /// The n-th (1-based) bound parameter, such as `?1` or `$1`.
    pub placeholder: fn(usize) -> String,
    /// Column type of a signed 64-bit integer.
    pub integer: &'static str,
    /// Column type of a byte string.
    pub bytes: &'static str,
    /// Column type of a bounded key string (namespace or record id).
    pub key: &'static str,
    /// Suffix that locks selected rows until commit; empty when a transaction already excludes
    /// other writers.
    pub row_lock: &'static str,
    /// Statement that serializes migrations across processes inside a transaction, when the
    /// transaction does not already.
    pub migration_lock: Option<&'static str>,
    /// Statements that create what [`Self::next_version`] draws from, run by the first migration.
    pub version_source: &'static str,
    /// Statement returning one row with the next store-wide record version.
    pub next_version: &'static str,
}

/// One open transaction. Dropping it without [`Self::commit`] or [`Self::rollback`] still rolls
/// it back; it only may cost the connection.
pub trait SqlTransaction: Send {
    /// Run a statement that returns no rows; returns the number of rows it changed.
    ///
    /// # Errors
    /// Returns a backend error from the database.
    fn execute(
        &mut self,
        statement: &str,
        parameters: &[SqlValue<'_>],
    ) -> impl Future<Output = Result<u64>> + Send;

    /// Run a statement and return its rows.
    ///
    /// # Errors
    /// Returns a backend error from the database, or for a column type outside [`SqlCell`].
    fn query(
        &mut self,
        statement: &str,
        parameters: &[SqlValue<'_>],
    ) -> impl Future<Output = Result<Vec<SqlRow>>> + Send;

    /// Run statements without parameters, separated by semicolons.
    ///
    /// # Errors
    /// Returns a backend error from the database.
    fn batch(&mut self, statements: &str) -> impl Future<Output = Result<()>> + Send;

    /// Make the transaction's writes visible.
    ///
    /// # Errors
    /// Returns a backend error; the writes are then discarded.
    fn commit(self) -> impl Future<Output = Result<()>> + Send;

    /// Discard the transaction's writes and keep the connection usable.
    ///
    /// # Errors
    /// Returns a backend error; the connection is then not reused.
    fn rollback(self) -> impl Future<Output = Result<()>> + Send;
}

/// A SQL database a [`crate::SqlRecordStore`] keeps records in.
pub trait SqlDatabase: Send + Sync + 'static {
    /// A transaction on one connection.
    type Transaction<'a>: SqlTransaction
    where
        Self: 'a;

    fn dialect(&self) -> Dialect;

    /// Whether processes on other hosts can use this database at the same time.
    fn shared(&self) -> bool;

    /// Open a transaction that excludes other writers of the rows it changes or locks.
    ///
    /// # Errors
    /// Returns a backend error when no connection can be opened.
    fn begin(&self) -> impl Future<Output = Result<Self::Transaction<'_>>> + Send;
}
