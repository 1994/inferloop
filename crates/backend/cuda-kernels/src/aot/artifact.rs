//! AOT artifact manifest: one file describes the cubins built for one architecture.
//!
//! The manifest is a build product, not a source file: CI compiles `profile × arch`, writes this
//! manifest next to the cubins, and the package ships them as sidecars. The loader validates the
//! schema, selects the target and verifies every digest before a module is loaded.

use crate::contract::target::{Profile, Target};
use infer_core::{Error, ErrorCode, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// Schema version accepted by this loader.
const SCHEMA: u32 = 1;
/// Characters for a lowercase hex digest.
const HEX: &[u8; 16] = b"0123456789abcdef";

/// One architecture's AOT artifact set.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    /// Manifest schema; refuses anything else.
    pub schema: u32,
    /// Build profile, always `aot` for this loader.
    pub profile: String,
    /// Architecture the cubins target, e.g. `sm_89`.
    pub arch: String,
    /// Toolchain that produced the cubins.
    pub toolchain: String,
    /// Kernel entries.
    pub entries: Vec<Entry>,
}

/// One kernel cubin in the set.
#[derive(Debug, Deserialize)]
pub struct Entry {
    /// Kernel name used by the provider, e.g. `projection.dense_bf16`.
    pub name: String,
    /// Numerical contract of the kernel.
    pub contract: String,
    /// Cubin path relative to the manifest.
    pub cubin: String,
    /// Lowercase SHA-256 of the cubin.
    pub sha256: String,
}

impl Manifest {
    /// Parse and validate a manifest.
    /// # Errors
    /// Rejects malformed JSON, an unknown schema, or a non-AOT profile.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| Error::invalid(format!("AOT manifest: {error}")))?;
        if manifest.schema != SCHEMA {
            return Err(Error::invalid("unsupported AOT manifest schema"));
        }
        if manifest.profile != "aot" {
            return Err(Error::invalid("AOT manifest profile mismatch"));
        }
        Ok(manifest)
    }

    /// Entries for one target.
    /// # Errors
    /// Rejects a non-AOT target or an architecture mismatch.
    pub fn select(&self, target: Target) -> Result<&[Entry]> {
        if target.profile != Profile::Aot || self.arch != target.arch.name() {
            return Err(Error::unsupported("no AOT manifest for this target"));
        }
        Ok(&self.entries)
    }

    /// Find one entry by kernel name.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Verify every cubin digest relative to `root`.
    /// # Errors
    /// Fails when a cubin is missing, unreadable, or its digest differs from the manifest.
    pub fn verify(&self, root: &Path) -> Result<()> {
        for entry in &self.entries {
            let path = root.join(&entry.cubin);
            let bytes = fs::read(&path).map_err(|error| {
                Error::new(
                    ErrorCode::Backend,
                    format!("AOT artifact {}: {error}", path.display()),
                )
            })?;
            let actual = hex(&Sha256::digest(&bytes));
            if actual != entry.sha256 {
                return Err(Error::invariant(format!(
                    "AOT artifact digest mismatch: {}",
                    entry.cubin
                )));
            }
        }
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
#[path = "../../tests/unit/aot_artifact.rs"]
mod tests;
