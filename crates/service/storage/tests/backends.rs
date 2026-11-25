//! Choosing record backends by the scheme of a storage location.
use infer_core::{ErrorCode, Result};
use infer_spi::{RecordFuture, RecordStore};
use infer_storage::{MemoryRecordStore, RecordBackends, RecordOpener, StorageLocation};
use std::sync::Arc;

struct Fixed;

impl RecordOpener for Fixed {
    fn open<'a>(&'a self, _: &'a StorageLocation) -> RecordFuture<'a, Arc<dyn RecordStore>> {
        Box::pin(async { Ok(Arc::new(MemoryRecordStore::default()) as Arc<dyn RecordStore>) })
    }
}

fn location(text: &str) -> Result<StorageLocation> {
    text.parse()
}

#[tokio::test]
async fn memory_is_built_in_and_takes_no_location() -> Result<()> {
    let backends = RecordBackends::with_memory();
    let store = backends.open(&location("memory")?).await?;
    assert!(!store.capabilities().durable);
    let code = backends
        .open(&location("memory:/tmp/x")?)
        .await
        .err()
        .map(|error| error.code);
    assert_eq!(code, Some(ErrorCode::InvalidInput));
    Ok(())
}

#[tokio::test]
async fn registered_schemes_open_and_unknown_ones_are_named() -> Result<()> {
    let mut backends = RecordBackends::with_memory();
    backends.register("custom", Arc::new(Fixed))?;
    backends.open(&location("custom:anything")?).await?;
    let mysql = location("mysql://db/infer")?;
    assert_eq!(
        backends.check(&mysql).err().map(|error| error.code),
        Some(ErrorCode::Unsupported)
    );
    let unknown = backends.open(&mysql).await.err();
    assert_eq!(
        unknown.as_ref().map(|error| error.code),
        Some(ErrorCode::Unsupported)
    );
    let message = unknown.map(|error| error.message).unwrap_or_default();
    assert!(
        message.contains("custom") && message.contains("memory"),
        "{message}"
    );
    Ok(())
}

#[test]
fn a_scheme_registers_once() {
    let mut backends = RecordBackends::with_memory();
    let code = backends
        .register("memory", Arc::new(Fixed))
        .err()
        .map(|error| error.code);
    assert_eq!(code, Some(ErrorCode::InvalidInput));
}
