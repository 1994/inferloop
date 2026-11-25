//! Persistent implementations of the Responses state extension.
//!
//! [`SqliteResponseStore`] keeps state in a local database file for a single host;
//! [`PostgresResponseStore`] keeps it in a PostgreSQL database that replicas share. Both
//! implement [`infer_spi::ResponseStore`] with the same retention, conditional-write, transition
//! and event semantics as the in-process store in `infer-agentic`. A deployment with other
//! storage implements that trait itself and installs it through
//! [`infer_spi::ResponsesExtensions::with_store`]; these stores use nothing beyond that contract.
mod postgres;
mod record;
mod sql;
mod sqlite;
pub use postgres::PostgresResponseStore;
pub use record::PersistentStoreConfig;
pub use sqlite::SqliteResponseStore;
