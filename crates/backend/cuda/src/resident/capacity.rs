//! Storage geometry follows model metadata and the tensor ABI, not a context preset.
use infer_core::{Error, Result};

/// # Errors
/// Rejects empty, overflowing or unrepresentable tensor capacities.
pub fn padded(capacity: usize) -> Result<usize> {
    if capacity == 0 {
        return Err(Error::invalid("CUDA sequence capacity must be positive"));
    }
    let padded = capacity
        .checked_next_power_of_two()
        .ok_or_else(|| Error::invalid("CUDA sequence capacity padding overflow"))?;
    i32::try_from(padded)
        .map_err(|_| Error::invalid("CUDA sequence capacity exceeds tensor index range"))?;
    Ok(padded)
}

/// # Errors
/// Rejects a request beyond the model context or an invalid storage capacity.
pub fn for_model(capacity: usize, model_limit: usize) -> Result<usize> {
    if capacity > model_limit {
        return Err(Error::invalid(format!(
            "CUDA sequence capacity {capacity} exceeds model context {model_limit}"
        )));
    }
    padded(capacity)
}

#[cfg(test)]
mod tests;
