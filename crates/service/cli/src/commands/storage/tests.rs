use super::{StorageBinding, StorageDomain, locations};
use infer_core::{ErrorCode, Result};

#[test]
fn bindings_name_a_domain_and_a_location() -> Result<()> {
    let binding: StorageBinding = "responses=postgres://user:secret@db/infer".parse()?;
    assert_eq!(binding.domain, StorageDomain::Responses);
    assert_eq!(binding.location.scheme(), "postgres");
    assert!(!format!("{binding:?}").contains("secret"));
    for bad in ["responses", "sessions=memory", "responses=Memory"] {
        let code = bad.parse::<StorageBinding>().err().map(|error| error.code);
        assert_eq!(code, Some(ErrorCode::InvalidInput), "{bad:?}");
    }
    Ok(())
}

#[test]
fn the_shorthand_is_memory_and_a_domain_binds_once() -> Result<()> {
    assert_eq!(locations(Vec::new(), false)?, None);
    let shorthand = locations(Vec::new(), true)?;
    assert_eq!(
        shorthand
            .map(|location| location.scheme().to_owned())
            .as_deref(),
        Some("memory")
    );
    let sqlite = vec!["responses=sqlite:/tmp/state.db".parse()?];
    let chosen = locations(sqlite.clone(), false)?;
    assert_eq!(
        chosen.map(|location| location.rest().to_owned()).as_deref(),
        Some("/tmp/state.db")
    );
    let twice = locations(sqlite, true).err().map(|error| error.code);
    assert_eq!(twice, Some(ErrorCode::InvalidInput));
    Ok(())
}
