//! Where each persistent domain keeps its state, chosen by `--storage DOMAIN=LOCATION`.
use infer_core::{Error, Result};
use infer_storage::StorageLocation;
use std::str::FromStr;

/// A persistent domain the server can back with storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDomain {
    /// Stored responses, chaining, retrieval, background jobs and cancellation.
    Responses,
}

impl StorageDomain {
    const ALL: [(&'static str, Self); 1] = [("responses", Self::Responses)];

    #[cfg(any(test, native_backend))]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "responses",
        }
    }
}

/// One `--storage DOMAIN=LOCATION` argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageBinding {
    pub domain: StorageDomain,
    pub location: StorageLocation,
}

impl FromStr for StorageBinding {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let (domain, location) = text.split_once('=').ok_or_else(|| {
            Error::invalid("storage must be DOMAIN=LOCATION, such as responses=memory")
        })?;
        let domain = StorageDomain::ALL
            .iter()
            .find(|(name, _)| *name == domain)
            .map(|(_, domain)| *domain)
            .ok_or_else(|| {
                let known: Vec<&str> = StorageDomain::ALL.iter().map(|(name, _)| *name).collect();
                Error::invalid(format!(
                    "unknown storage domain {domain:?}; known: {}",
                    known.join(", ")
                ))
            })?;
        Ok(Self {
            domain,
            location: location.parse()?,
        })
    }
}

/// The location of each domain, from `--storage` and the `--enable-response-store` shorthand.
///
/// # Errors
/// Returns an invalid-input error when a domain is bound twice.
#[cfg(any(test, native_backend))]
pub fn locations(
    bindings: Vec<StorageBinding>,
    enable_response_store: bool,
) -> Result<Option<StorageLocation>> {
    let mut responses = enable_response_store
        .then(|| "memory".parse())
        .transpose()?;
    for binding in bindings {
        match binding.domain {
            StorageDomain::Responses if responses.is_some() => {
                return Err(Error::invalid(format!(
                    "storage for {} is given twice",
                    binding.domain.as_str()
                )));
            }
            StorageDomain::Responses => responses = Some(binding.location),
        }
    }
    Ok(responses)
}

#[cfg(test)]
mod tests;
