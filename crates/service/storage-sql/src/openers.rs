//! The `sqlite:` and `postgres://` schemes for [`RecordBackends`].
use crate::postgres::Postgres;
use crate::records::{SqlRecordStore, SqlRecordStoreConfig};
use crate::sqlite::Sqlite;
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{RecordFuture, RecordStore};
use infer_storage::{RecordBackends, RecordOpener, StorageLocation};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task;

/// Schemes that name a PostgreSQL database; the whole location is its URL.
const POSTGRES_SCHEMES: [&str; 2] = ["postgres", "postgresql"];

/// Register `sqlite:PATH` (or `sqlite://PATH`) and `postgres://...` (or `postgresql://...`).
///
/// # Errors
/// Returns an invalid-input error when one of these schemes is already registered.
pub fn register(backends: &mut RecordBackends) -> Result<()> {
    backends.register("sqlite", Arc::new(SqliteOpener))?;
    for scheme in POSTGRES_SCHEMES {
        backends.register(scheme, Arc::new(PostgresOpener))?;
    }
    Ok(())
}

struct SqliteOpener;

impl RecordOpener for SqliteOpener {
    fn open<'a>(&'a self, location: &'a StorageLocation) -> RecordFuture<'a, Arc<dyn RecordStore>> {
        Box::pin(async move {
            let path = location.rest();
            let path = path.strip_prefix("//").unwrap_or(path);
            if path.is_empty() {
                return Err(Error::invalid(
                    "a sqlite location needs a file path: sqlite:PATH",
                ));
            }
            let path = PathBuf::from(path);
            // Opening the file and setting its journal mode block; keep them off async threads.
            let database = task::spawn_blocking(move || Sqlite::open(&path))
                .await
                .map_err(|_| Error::new(ErrorCode::Backend, "sqlite open task failed"))??;
            let store = SqlRecordStore::open(database, SqlRecordStoreConfig::default()).await?;
            Ok(Arc::new(store) as Arc<dyn RecordStore>)
        })
    }
}

struct PostgresOpener;

impl RecordOpener for PostgresOpener {
    fn open<'a>(&'a self, location: &'a StorageLocation) -> RecordFuture<'a, Arc<dyn RecordStore>> {
        Box::pin(async move {
            let database = Postgres::connect(location.expose())?;
            let store = SqlRecordStore::open(database, SqlRecordStoreConfig::default()).await?;
            Ok(Arc::new(store) as Arc<dyn RecordStore>)
        })
    }
}
