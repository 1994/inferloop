//! SQL record stores: the record semantics written once, and a dialect per database.
//!
//! [`SqlRecordStore`] implements [`infer_spi::RecordStore`] over any [`SqlDatabase`]; SQLite
//! ([`Sqlite`]) and PostgreSQL ([`Postgres`]) supply only a [`Dialect`] and statement execution.
//! Another SQL database is one more implementation of [`SqlDatabase`] and [`SqlTransaction`],
//! checked by the same conformance suites. [`register`] adds the `sqlite:` and `postgres://`
//! schemes to a deployment's [`infer_storage::RecordBackends`].
mod database;
mod openers;
mod postgres;
mod records;
mod schema;
mod sqlite;
pub use database::{Dialect, SqlCell, SqlDatabase, SqlRow, SqlTransaction, SqlValue};
pub use openers::register;
pub use postgres::{Postgres, PostgresTransaction};
pub use records::{SqlRecordStore, SqlRecordStoreConfig};
pub use sqlite::{Sqlite, SqliteTransaction};
