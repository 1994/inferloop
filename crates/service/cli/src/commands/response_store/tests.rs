use super::ResponseStoreLocation;
use infer_core::{ErrorCode, Result};
use std::path::PathBuf;
use std::{env, fs, process};

#[test]
fn locations_parse_by_scheme() -> Result<()> {
    assert_eq!(
        "memory".parse::<ResponseStoreLocation>()?,
        ResponseStoreLocation::Memory
    );
    let path = PathBuf::from("/var/lib/infer/responses.db");
    for spelling in [
        "sqlite:/var/lib/infer/responses.db",
        "sqlite:///var/lib/infer/responses.db",
    ] {
        assert_eq!(
            spelling.parse::<ResponseStoreLocation>()?,
            ResponseStoreLocation::Sqlite(path.clone())
        );
    }
    for url in ["postgres://u:p@db/infer", "postgresql://db/infer"] {
        assert_eq!(
            url.parse::<ResponseStoreLocation>()?,
            ResponseStoreLocation::Postgres(url.to_owned())
        );
    }
    for bad in ["", "sqlite:", "redis://cache", "Memory"] {
        let code = bad
            .parse::<ResponseStoreLocation>()
            .err()
            .map(|error| error.code);
        assert_eq!(code, Some(ErrorCode::InvalidInput), "{bad:?}");
    }
    Ok(())
}

#[test]
fn debug_output_hides_postgres_credentials() -> Result<()> {
    let location: ResponseStoreLocation = "postgres://user:hunter2@db/infer".parse()?;
    assert_eq!(format!("{location:?}"), "Postgres(..)");
    Ok(())
}

#[tokio::test]
async fn sqlite_locations_open_a_database_file() -> Result<()> {
    let path = env::temp_dir().join(format!("infer-cli-store-{}.sqlite", process::id()));
    let store = ResponseStoreLocation::Sqlite(path.clone()).open().await?;
    assert!(store.capabilities().persistent);
    drop(store);
    for suffix in ["", "-wal", "-shm"] {
        let mut file = path.clone().into_os_string();
        file.push(suffix);
        let _missing = fs::remove_file(file);
    }
    Ok(())
}
