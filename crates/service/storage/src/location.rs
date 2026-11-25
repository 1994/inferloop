//! Where a backend keeps its data, as the deployment wrote it.
use infer_core::{Error, Result};
use std::fmt;
use std::str::FromStr;

/// What separates a URL's scheme from its authority (`user:password@host`).
const AUTHORITY: &str = "://";

/// A backend location such as `memory`, `sqlite:/var/lib/infer/state.db` or
/// `postgres://user:secret@db/infer`: a scheme, then whatever that backend reads.
///
/// The text may carry credentials, so `Debug` and `Display` hide a URL's password; only the
/// backend that opens the location reads it whole, through [`Self::expose`].
#[derive(Clone, PartialEq, Eq)]
pub struct StorageLocation {
    text: Box<str>,
    scheme: usize,
}

impl StorageLocation {
    /// The scheme, which selects the backend.
    #[must_use]
    pub fn scheme(&self) -> &str {
        &self.text[..self.scheme]
    }

    /// What follows `scheme:`; empty for a bare scheme.
    #[must_use]
    pub fn rest(&self) -> &str {
        self.text.get(self.scheme + 1..).unwrap_or_default()
    }

    /// The whole location, credentials included, for the backend that opens it.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.text
    }
}

impl FromStr for StorageLocation {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let scheme = text.find(':').unwrap_or(text.len());
        let name = &text[..scheme];
        let valid = name.starts_with(|first: char| first.is_ascii_lowercase())
            && name.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"+.-".contains(&byte)
            });
        if !valid {
            return Err(Error::invalid(
                "storage location must start with a lowercase scheme, such as memory, sqlite: or postgres://",
            ));
        }
        Ok(Self {
            text: text.into(),
            scheme,
        })
    }
}

impl fmt::Display for StorageLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = &*self.text;
        // `scheme://user:password@host...`: show everything but the password.
        let redacted = text.find(AUTHORITY).and_then(|start| {
            let authority = start + AUTHORITY.len();
            let end = text[authority..]
                .find('/')
                .map_or(text.len(), |slash| authority + slash);
            let at = text[authority..end].rfind('@')? + authority;
            let colon = text[authority..at].find(':')? + authority;
            Some(format!("{}:***{}", &text[..colon], &text[at..]))
        });
        formatter.write_str(redacted.as_deref().unwrap_or(text))
    }
}

impl fmt::Debug for StorageLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "StorageLocation({self})")
    }
}

#[cfg(test)]
mod tests;
