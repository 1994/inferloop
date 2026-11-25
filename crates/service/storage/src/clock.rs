//! Wall-clock time for retention deadlines.
use infer_core::{Error, Result};
use std::time::{SystemTime, UNIX_EPOCH};

/// Milliseconds since the Unix epoch. Deadlines are hours long, so clock skew between replicas
/// of a few seconds does not change what is kept.
pub fn now() -> Result<u64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::invariant("system clock is before the Unix epoch"))?;
    u64::try_from(elapsed.as_millis()).map_err(|_| Error::invariant("system clock overflow"))
}
