//! The SQL record stores against the shared suites, plus what only a database can show.
//!
//! The PostgreSQL cases need a server: run them with
//! `INFER_TEST_POSTGRES_URL=postgres://... cargo test -p infer-storage-sql -- --ignored`.
//! They fail, rather than pass empty, when the variable is missing.
use infer_core::{Error, ErrorCode, Result};
use infer_spi::{RecordStore, ResponseStore};
use infer_storage::{RecordResponseStore, ResponseRetention};
use infer_storage_conformance::{records, responses};
use infer_storage_sql::{Postgres, SqlRecordStore, SqlRecordStoreConfig, Sqlite};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio_postgres::Config as PostgresConfig;

const SHORT_TTL: Duration = Duration::from_millis(50);
const POSTGRES_URL: &str = "INFER_TEST_POSTGRES_URL";

static FILES: AtomicU64 = AtomicU64::new(0);

/// A database file removed, with its write-ahead log, when the case ends.
struct TempDatabase(PathBuf);

impl TempDatabase {
    fn new() -> Self {
        let count = FILES.fetch_add(1, Ordering::Relaxed);
        Self(env::temp_dir().join(format!(
            "infer-storage-sql-{}-{count}.sqlite",
            process::id()
        )))
    }

    async fn store(&self) -> Result<SqlRecordStore<Sqlite>> {
        SqlRecordStore::open(Sqlite::open(&self.0)?, SqlRecordStoreConfig::default()).await
    }
}

impl Drop for TempDatabase {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let mut path = self.0.clone().into_os_string();
            path.push(suffix);
            // A file the case never created is already gone.
            drop(fs::remove_file(path));
        }
    }
}

fn responses_over(
    records: Arc<dyn RecordStore>,
    ttl: Option<Duration>,
) -> Result<RecordResponseStore> {
    let retention = ResponseRetention {
        max_record_bytes: 1024,
        max_events: responses::EVENTS,
        ttl,
    };
    RecordResponseStore::new(records, retention)
}

async fn all_suites(records: Arc<dyn RecordStore>) -> Result<()> {
    records::suite(records.as_ref()).await?;
    records::concurrent_versions(records.as_ref(), records.as_ref()).await?;
    responses::suite(&responses_over(Arc::clone(&records), None)?).await?;
    responses::expiry(&responses_over(records, Some(SHORT_TTL))?, SHORT_TTL).await
}

#[tokio::test]
async fn sqlite_conforms() -> Result<()> {
    let database = TempDatabase::new();
    all_suites(Arc::new(database.store().await?)).await
}

#[tokio::test]
async fn sqlite_state_survives_reopening() -> Result<()> {
    let database = TempDatabase::new();
    let id = infer_spi::ResponseId::new("resp_reopened")?;
    let response = infer_spi::StoredResponse {
        id: id.clone(),
        request_id: infer_core::RequestId::new(3)?,
        model: "model".to_owned(),
        created_at: 1,
        completed_at: Some(2),
        status: infer_spi::ResponseStatus::Completed,
        incomplete: None,
        error: None,
        usage: None,
        previous_response_id: None,
        request_input: b"[]".to_vec(),
        response: b"{}".to_vec(),
        payload_version: infer_spi::RESPONSES_PAYLOAD_VERSION,
    };
    {
        let store = responses_over(Arc::new(database.store().await?), None)?;
        store
            .put(response.clone(), infer_spi::WriteCondition::Always)
            .await?;
        let event = infer_spi::ResponseEvent {
            sequence: 1,
            payload: b"{}".to_vec(),
        };
        store.append_event(&id, event).await?;
    }
    let reopened = responses_over(Arc::new(database.store().await?), None)?;
    assert_eq!(reopened.get(&id).await?, Some(response));
    assert_eq!(reopened.events(&id, None).await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn sqlite_refuses_a_newer_schema() -> Result<()> {
    let database = TempDatabase::new();
    drop(database.store().await?);
    let connection = rusqlite::Connection::open(&database.0)
        .map_err(|error| Error::invalid(error.to_string()))?;
    connection
        .execute(
            "UPDATE infer_storage_meta SET value = value + 1 WHERE key = 'schema_version'",
            [],
        )
        .map_err(|error| Error::invalid(error.to_string()))?;
    drop(connection);
    let refused = database.store().await.err().map(|error| error.code);
    assert_eq!(refused, Some(ErrorCode::Unsupported));
    Ok(())
}

fn postgres_url() -> Result<String> {
    env::var(POSTGRES_URL)
        .map_err(|_| Error::invalid(format!("set {POSTGRES_URL} to run the PostgreSQL cases")))
}

async fn postgres() -> Result<SqlRecordStore<Postgres>> {
    SqlRecordStore::open(
        Postgres::connect(&postgres_url()?)?,
        SqlRecordStoreConfig::default(),
    )
    .await
}

#[tokio::test]
#[ignore = "needs a PostgreSQL server named by INFER_TEST_POSTGRES_URL"]
async fn postgres_conforms() -> Result<()> {
    all_suites(Arc::new(postgres().await?)).await
}

#[tokio::test]
#[ignore = "needs a PostgreSQL server named by INFER_TEST_POSTGRES_URL"]
async fn postgres_versions_are_atomic_across_stores() -> Result<()> {
    let (first, second) = (postgres().await?, postgres().await?);
    records::concurrent_versions(&first, &second).await
}

#[tokio::test]
#[ignore = "needs a PostgreSQL server named by INFER_TEST_POSTGRES_URL"]
async fn postgres_debug_hides_the_password() -> Result<()> {
    let url = postgres_url()?;
    let shown = format!("{:?}", postgres().await?);
    let config =
        PostgresConfig::from_str(&url).map_err(|error| Error::invalid(error.to_string()))?;
    let password = config.get_password().unwrap_or_default();
    assert!(!password.is_empty(), "the test URL needs a password");
    assert!(
        !shown
            .as_bytes()
            .windows(password.len())
            .any(|window| window == password)
    );
    Ok(())
}

#[test]
fn malformed_postgres_urls_are_invalid() {
    let code = Postgres::connect("postgres://host:notaport/db")
        .err()
        .map(|error| error.code);
    assert_eq!(code, Some(ErrorCode::InvalidInput));
}
