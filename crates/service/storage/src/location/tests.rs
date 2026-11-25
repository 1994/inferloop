use super::StorageLocation;
use infer_core::{ErrorCode, Result};

#[test]
fn locations_split_into_scheme_and_rest() -> Result<()> {
    let memory: StorageLocation = "memory".parse()?;
    assert_eq!((memory.scheme(), memory.rest()), ("memory", ""));
    let sqlite: StorageLocation = "sqlite:/var/lib/infer/state.db".parse()?;
    assert_eq!(
        (sqlite.scheme(), sqlite.rest()),
        ("sqlite", "/var/lib/infer/state.db")
    );
    let postgres: StorageLocation = "postgres://db/infer".parse()?;
    assert_eq!(
        (postgres.scheme(), postgres.rest()),
        ("postgres", "//db/infer")
    );
    for bad in ["", ":x", "Memory", "1db:x", "my db:x"] {
        let code = bad.parse::<StorageLocation>().err().map(|error| error.code);
        assert_eq!(code, Some(ErrorCode::InvalidInput), "{bad:?}");
    }
    Ok(())
}

#[test]
fn display_and_debug_hide_url_passwords() -> Result<()> {
    let location: StorageLocation =
        "postgres://infer:hunter2@db.internal:5432/infer?sslmode=require".parse()?;
    let shown = format!("{location} {location:?}");
    assert!(!shown.contains("hunter2"), "{shown}");
    assert!(
        shown.contains("postgres://infer:***@db.internal:5432/infer?sslmode=require"),
        "{shown}"
    );
    assert_eq!(
        location.expose(),
        "postgres://infer:hunter2@db.internal:5432/infer?sslmode=require"
    );
    let without: StorageLocation = "postgres://db/infer".parse()?;
    assert_eq!(without.to_string(), "postgres://db/infer");
    let user_only: StorageLocation = "postgres://infer@db/infer".parse()?;
    assert_eq!(user_only.to_string(), "postgres://infer@db/infer");
    Ok(())
}
