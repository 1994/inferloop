//! Where the server keeps Responses state, chosen by `--response-store`.
use infer_core::{Error, Result};
#[cfg(any(test, native_backend))]
use infer_frontdoor::InMemoryResponseStore;
#[cfg(any(test, native_backend))]
use infer_response_store::{PersistentStoreConfig, PostgresResponseStore, SqliteResponseStore};
#[cfg(any(test, native_backend))]
use infer_spi::ResponseStore;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
#[cfg(any(test, native_backend))]
use std::sync::Arc;

const SQLITE_PREFIXES: [&str; 2] = ["sqlite://", "sqlite:"];
const POSTGRES_PREFIXES: [&str; 2] = ["postgres://", "postgresql://"];

/// A response store location: `memory`, `sqlite:PATH`, or a `postgres://` URL.
#[derive(Clone, PartialEq, Eq)]
pub enum ResponseStoreLocation {
    /// Bounded, process-local state; lost on restart.
    Memory,
    /// A local SQLite database file, created when missing.
    Sqlite(PathBuf),
    /// A PostgreSQL database shared by replicas. The URL may hold a password.
    Postgres(String),
}

impl FromStr for ResponseStoreLocation {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        if value == "memory" {
            return Ok(Self::Memory);
        }
        if let Some(path) = SQLITE_PREFIXES
            .iter()
            .find_map(|prefix| value.strip_prefix(prefix))
        {
            if path.is_empty() {
                return Err(Error::invalid("sqlite response store needs a file path"));
            }
            return Ok(Self::Sqlite(PathBuf::from(path)));
        }
        if POSTGRES_PREFIXES
            .iter()
            .any(|prefix| value.starts_with(prefix))
        {
            return Ok(Self::Postgres(value.to_owned()));
        }
        Err(Error::invalid(
            "response store must be `memory`, `sqlite:PATH` or a `postgres://` URL",
        ))
    }
}

impl fmt::Debug for ResponseStoreLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory => formatter.write_str("Memory"),
            Self::Sqlite(path) => formatter.debug_tuple("Sqlite").field(path).finish(),
            // The URL may carry credentials.
            Self::Postgres(_) => formatter.write_str("Postgres(..)"),
        }
    }
}

impl ResponseStoreLocation {
    /// Open the store, creating tables or the database file when missing.
    ///
    /// # Errors
    /// Returns the store's error when the database cannot be opened or reached.
    #[cfg(any(test, native_backend))]
    pub async fn open(&self) -> Result<Arc<dyn ResponseStore>> {
        let limits = PersistentStoreConfig::default();
        let store: Arc<dyn ResponseStore> = match self {
            Self::Memory => Arc::new(InMemoryResponseStore::default()),
            Self::Sqlite(path) => Arc::new(SqliteResponseStore::open(path, limits)?),
            Self::Postgres(url) => Arc::new(PostgresResponseStore::connect(url, limits).await?),
        };
        tracing::info!(target: "infer::serve", store = ?self, "response store ready");
        Ok(store)
    }
}

#[cfg(test)]
mod tests;
