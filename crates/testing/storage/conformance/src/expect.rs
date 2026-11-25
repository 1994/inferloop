//! Expectations shared by the suites; failures are errors, not panics.
use infer_core::{Error, ErrorCode, Result};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static FRESH: AtomicU64 = AtomicU64::new(0);

/// An identifier no other case, process or earlier run uses.
pub fn fresh_id(label: &str) -> Result<String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::invariant("clock before the Unix epoch"))?
        .as_nanos();
    let count = FRESH.fetch_add(1, Ordering::Relaxed);
    Ok(format!("{label}_{nanos}_{}_{count}", process::id()))
}

pub fn check(condition: bool, expectation: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::invariant(format!(
            "conformance: expected {expectation}"
        )))
    }
}

pub fn assert_code<T>(result: Result<T>, code: ErrorCode, case: &str) -> Result<()> {
    match result {
        Err(error) if error.code == code => Ok(()),
        Err(error) => Err(Error::invariant(format!(
            "conformance: {case} failed with {:?} ({}), expected {code:?}",
            error.code, error.message
        ))),
        Ok(_) => Err(Error::invariant(format!(
            "conformance: {case} succeeded, expected {code:?}"
        ))),
    }
}
