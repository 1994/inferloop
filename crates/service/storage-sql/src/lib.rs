//! SQL record stores: the record semantics written once, and a dialect per database.
//!
//! [`SqlRecordStore`] implements [`infer_spi::RecordStore`] over any [`SqlDatabase`]; SQLite
//! ([`Sqlite`]) and PostgreSQL ([`Postgres`]) supply only a [`Dialect`] and statement execution.
//! Another SQL database is one more implementation of [`SqlDatabase`] and [`SqlTransaction`],
//! checked by the same conformance suites.
mod database;
mod postgres;
mod records;
mod schema;
mod sqlite;
pub use database::{Dialect, SqlCell, SqlDatabase, SqlRow, SqlTransaction, SqlValue};
pub use postgres::{Postgres, PostgresTransaction};
pub use records::{SqlRecordStore, SqlRecordStoreConfig};
pub use sqlite::{Sqlite, SqliteTransaction};
