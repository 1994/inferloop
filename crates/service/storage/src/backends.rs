//! Record backends selected by the scheme of a [`StorageLocation`].
use crate::location::StorageLocation;
use crate::memory::MemoryRecordStore;
use infer_core::{Error, Result};
use infer_spi::{RecordFuture, RecordStore};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Opens record stores of one kind from their locations.
pub trait RecordOpener: Send + Sync {
    /// Open the store at `location`, whose scheme selected this opener.
    ///
    /// # Errors
    /// Returns an invalid-input error for a location this backend cannot read, and the
    /// backend's errors when the store cannot be opened.
    fn open<'a>(&'a self, location: &'a StorageLocation) -> RecordFuture<'a, Arc<dyn RecordStore>>;
}

/// The record backends a deployment can name, by scheme.
///
/// The deployment registers the backends it links; a location then picks one by its scheme.
/// An SDK adds a backend by registering another scheme, and every domain store built on
/// records can use it.
#[derive(Default)]
pub struct RecordBackends {
    openers: BTreeMap<Box<str>, Arc<dyn RecordOpener>>,
}

impl RecordBackends {
    /// Backends with the in-process `memory` scheme registered.
    #[must_use]
    pub fn with_memory() -> Self {
        let mut backends = Self::default();
        backends
            .openers
            .insert("memory".into(), Arc::new(MemoryOpener));
        backends
    }

    /// Make `scheme` open stores with `opener`.
    ///
    /// # Errors
    /// Returns an invalid-input error when the scheme is already registered.
    pub fn register(&mut self, scheme: &str, opener: Arc<dyn RecordOpener>) -> Result<()> {
        if self.openers.contains_key(scheme) {
            return Err(Error::invalid(format!(
                "storage scheme {scheme} is registered twice"
            )));
        }
        self.openers.insert(scheme.into(), opener);
        Ok(())
    }

    /// The opener for `location`'s scheme, so a deployment can reject an unknown scheme before
    /// any slow startup work.
    ///
    /// # Errors
    /// Returns an unsupported error for a scheme nobody registered, naming those that are.
    pub fn check(&self, location: &StorageLocation) -> Result<()> {
        self.opener(location).map(drop)
    }

    fn opener(&self, location: &StorageLocation) -> Result<&dyn RecordOpener> {
        self.openers
            .get(location.scheme())
            .map(AsRef::as_ref)
            .ok_or_else(|| {
                let known: Vec<&str> = self.openers.keys().map(AsRef::as_ref).collect();
                Error::unsupported(format!(
                    "no storage backend for scheme {}; available: {}",
                    location.scheme(),
                    known.join(", ")
                ))
            })
    }

    /// Open the store `location` names.
    ///
    /// # Errors
    /// Returns the [`Self::check`] error for an unknown scheme, and the opener's errors.
    pub async fn open(&self, location: &StorageLocation) -> Result<Arc<dyn RecordStore>> {
        self.opener(location)?.open(location).await
    }
}

struct MemoryOpener;

impl RecordOpener for MemoryOpener {
    fn open<'a>(&'a self, location: &'a StorageLocation) -> RecordFuture<'a, Arc<dyn RecordStore>> {
        Box::pin(async move {
            if !location.rest().is_empty() {
                return Err(Error::invalid(
                    "the memory store takes no location: write `memory`",
                ));
            }
            Ok(Arc::new(MemoryRecordStore::default()) as Arc<dyn RecordStore>)
        })
    }
}
