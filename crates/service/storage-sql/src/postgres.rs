//! A PostgreSQL database that replicas share, through a bounded connection pool.
use crate::database::{Dialect, SqlCell, SqlDatabase, SqlRow, SqlTransaction, SqlValue};
use bytes::BytesMut;
use deadpool_postgres::{Manager, ManagerConfig, Object, Pool, RecyclingMethod};
use infer_core::{Error, ErrorCode, Result};
use rustls::crypto::ring::default_provider;
use rustls::{ClientConfig, RootCertStore};
use std::error::Error as StdError;
use std::fmt;
use std::result::Result as StdResult;
use std::str::FromStr;
use std::sync::Arc;
use tokio_postgres::config::SslMode;
use tokio_postgres::types::{IsNull, ToSql, Type, to_sql_checked};
use tokio_postgres::{Config, NoTls};
use tokio_postgres_rustls::MakeRustlsConnect;

/// Connections one store keeps open; calls beyond this wait for a free connection.
const CONNECTIONS: usize = 4;

/// A PostgreSQL database named by a `postgres://` URL or key-value connection string.
///
/// Up to four pooled connections. With `sslmode=prefer` (the default) or `require`, the
/// connection uses TLS and verifies the server's certificate chain and host name against the
/// system roots; `prefer` falls back to plaintext only when the server does not offer TLS.
/// `sslmode=disable` connects in plaintext.
pub struct Postgres {
    pool: Pool,
    hosts: String,
    dbname: Option<String>,
}

impl fmt::Debug for Postgres {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The connection config holds the password; show only where the database is.
        formatter
            .debug_struct("Postgres")
            .field("hosts", &self.hosts)
            .field("dbname", &self.dbname)
            .finish_non_exhaustive()
    }
}

impl Postgres {
    /// Prepare a pool for `url`. Connections open on first use.
    ///
    /// # Errors
    /// Returns an invalid-input error for a malformed URL, and a backend error when the system
    /// certificate roots cannot be loaded for a TLS connection.
    pub fn connect(url: &str) -> Result<Self> {
        let config = Config::from_str(url)
            .map_err(|error| Error::invalid(format!("invalid postgres URL: {error}")))?;
        let hosts = format!("{:?}", config.get_hosts());
        let dbname = config.get_dbname().map(str::to_owned);
        let manager_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let manager = match config.get_ssl_mode() {
            SslMode::Disable => Manager::from_config(config, NoTls, manager_config),
            _ => Manager::from_config(config, MakeRustlsConnect::new(tls()?), manager_config),
        };
        let pool = Pool::builder(manager)
            .max_size(CONNECTIONS)
            .build()
            .map_err(|error| Error::invalid(format!("postgres pool: {error}")))?;
        Ok(Self {
            pool,
            hosts,
            dbname,
        })
    }
}

fn tls() -> Result<ClientConfig> {
    let mut roots = RootCertStore::empty();
    let native = rustls_native_certs::load_native_certs();
    for certificate in native.certs {
        // One unparsable system certificate must not disable TLS for the rest.
        if let Err(error) = roots.add(certificate) {
            tracing::debug!(target: "infer::storage", %error, "skipped a system root certificate");
        }
    }
    if roots.is_empty() {
        return Err(Error::new(
            ErrorCode::Backend,
            "no system root certificates for postgres TLS; use sslmode=disable for plaintext",
        ));
    }
    Ok(
        ClientConfig::builder_with_provider(Arc::new(default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|error| Error::new(ErrorCode::Backend, format!("postgres TLS: {error}")))?
            .with_root_certificates(roots)
            .with_no_client_auth(),
    )
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "a `map_err` adapter receives the error by value"
)]
fn backend(error: tokio_postgres::Error) -> Error {
    Error::new(ErrorCode::Backend, format!("postgres: {error}"))
}

impl ToSql for SqlValue<'_> {
    fn to_sql(
        &self,
        column: &Type,
        out: &mut BytesMut,
    ) -> StdResult<IsNull, Box<dyn StdError + Sync + Send>> {
        match self {
            Self::Null => Ok(IsNull::Yes),
            Self::Integer(value) => value.to_sql(column, out),
            Self::Text(text) => text.to_sql(column, out),
            Self::Bytes(bytes) => bytes.to_sql(column, out),
        }
    }

    fn accepts(_: &Type) -> bool {
        true
    }

    to_sql_checked!();
}

fn cell(row: &tokio_postgres::Row, index: usize) -> Result<SqlCell> {
    let column = row
        .columns()
        .get(index)
        .ok_or_else(|| Error::invariant(format!("postgres row has no column {index}")))?;
    let cell = match *column.type_() {
        Type::INT8 => row
            .try_get::<_, Option<i64>>(index)
            .map(|value| value.map(SqlCell::Integer)),
        Type::INT4 => row
            .try_get::<_, Option<i32>>(index)
            .map(|value| value.map(|value| SqlCell::Integer(i64::from(value)))),
        Type::TEXT | Type::VARCHAR => row
            .try_get::<_, Option<String>>(index)
            .map(|value| value.map(SqlCell::Text)),
        Type::BYTEA => row
            .try_get::<_, Option<Vec<u8>>>(index)
            .map(|value| value.map(SqlCell::Bytes)),
        ref other => {
            return Err(Error::invariant(format!(
                "postgres column type {other} is not supported"
            )));
        }
    };
    Ok(cell.map_err(backend)?.unwrap_or(SqlCell::Null))
}

/// One transaction on a pooled connection.
///
/// Dropped without commit or rollback, the connection is closed instead of returned to the
/// pool; the server then rolls the transaction back, and no half-open transaction is reused.
pub struct PostgresTransaction {
    connection: Option<Object>,
}

impl PostgresTransaction {
    fn connection(&self) -> Result<&Object> {
        self.connection
            .as_ref()
            .ok_or_else(|| Error::invariant("postgres transaction already ended"))
    }

    fn end(&mut self, keep: bool) {
        if let Some(connection) = self.connection.take()
            && !keep
        {
            drop(Object::take(connection));
        }
    }
}

impl Drop for PostgresTransaction {
    fn drop(&mut self) {
        self.end(false);
    }
}

fn parameters<'a>(values: &'a [SqlValue<'a>]) -> Vec<&'a (dyn ToSql + Sync)> {
    values
        .iter()
        .map(|value| value as &(dyn ToSql + Sync))
        .collect()
}

impl SqlTransaction for PostgresTransaction {
    async fn execute(&mut self, statement: &str, values: &[SqlValue<'_>]) -> Result<u64> {
        let connection = self.connection()?;
        let prepared = connection
            .prepare_cached(statement)
            .await
            .map_err(backend)?;
        connection
            .execute(&prepared, &parameters(values))
            .await
            .map_err(backend)
    }

    async fn query(&mut self, statement: &str, values: &[SqlValue<'_>]) -> Result<Vec<SqlRow>> {
        let connection = self.connection()?;
        let prepared = connection
            .prepare_cached(statement)
            .await
            .map_err(backend)?;
        let rows = connection
            .query(&prepared, &parameters(values))
            .await
            .map_err(backend)?;
        rows.iter()
            .map(|row| {
                (0..row.len())
                    .map(|index| cell(row, index))
                    .collect::<Result<Vec<_>>>()
                    .map(SqlRow)
            })
            .collect()
    }

    async fn batch(&mut self, statements: &str) -> Result<()> {
        self.connection()?
            .batch_execute(statements)
            .await
            .map_err(backend)
    }

    async fn commit(mut self) -> Result<()> {
        let result = self
            .connection()?
            .batch_execute("COMMIT")
            .await
            .map_err(backend);
        self.end(result.is_ok());
        result
    }

    async fn rollback(mut self) -> Result<()> {
        let result = self
            .connection()?
            .batch_execute("ROLLBACK")
            .await
            .map_err(backend);
        self.end(result.is_ok());
        result
    }
}

impl SqlDatabase for Postgres {
    type Transaction<'a> = PostgresTransaction;

    fn dialect(&self) -> Dialect {
        Dialect {
            placeholder: |index| format!("${index}"),
            integer: "BIGINT",
            bytes: "BYTEA",
            key: "TEXT",
            row_lock: " FOR UPDATE",
            // Replicas starting together create the tables one at a time.
            migration_lock: Some("SELECT pg_advisory_xact_lock(7596553476052414323)"),
            version_source: "CREATE SEQUENCE IF NOT EXISTS infer_record_versions",
            next_version: "SELECT nextval('infer_record_versions')",
        }
    }

    fn shared(&self) -> bool {
        true
    }

    async fn begin(&self) -> Result<PostgresTransaction> {
        let connection =
            self.pool.get().await.map_err(|error| {
                Error::new(ErrorCode::Backend, format!("postgres pool: {error}"))
            })?;
        let mut transaction = PostgresTransaction {
            connection: Some(connection),
        };
        transaction.batch("BEGIN").await?;
        Ok(transaction)
    }
}
