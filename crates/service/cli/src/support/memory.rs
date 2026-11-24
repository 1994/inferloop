//! Memory budget helpers shared by backend loaders.
use infer_core::{Error, Result};

/// Bytes covered by a fraction of total device memory, following vLLM's
/// `--gpu-memory-utilization` contract: the fraction bounds the whole engine, weights included.
/// # Errors
/// Rejects fractions outside `0.0..=1.0`.
pub fn utilization_bytes(total_bytes: u64, utilization: f64) -> Result<u64> {
    if !(0.0..=1.0).contains(&utilization) {
        return Err(Error::invalid(
            "gpu memory utilization must be within 0.0..=1.0",
        ));
    }
    if utilization == 0.0 {
        // Zero means "no explicit cap": the device itself stays the bound.
        return Ok(total_bytes);
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "Device memory sizes are far below f64 mantissa precision"
    )]
    let scaled = (total_bytes as f64) * utilization;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "The product is non-negative and bounded by total device memory"
    )]
    Ok(scaled as u64)
}

#[cfg(test)]
mod tests;
