//! Math workload responsibilities.
use infer_core::{Error, Result};
use infer_ir::Pooling;

/// # Errors
/// Returns an invalid-input error for empty, inconsistent, or non-finite hidden rows.
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
pub fn pool(hidden: &[Vec<f32>], pooling: Pooling) -> Result<Vec<f32>> {
    let width = hidden.first().map_or(0, Vec::len);
    if width == 0
        || hidden
            .iter()
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
    {
        return Err(Error::invalid("invalid hidden states"));
    }
    if pooling == Pooling::Last {
        return Ok(hidden
            .last()
            .ok_or_else(|| Error::invariant("nonempty hidden"))?
            .clone());
    }
    Ok((0..width)
        .map(|i| (hidden.iter().map(|v| f64::from(v[i])).sum::<f64>() / hidden.len() as f64) as f32)
        .collect())
}
/// # Errors
/// Returns an invalid-input error for empty or non-finite logits, invalid temperature, or a non-finite normalization sum.
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
pub fn softmax(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if logits.is_empty()
        || logits.iter().any(|v| !v.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.0
    {
        return Err(Error::invalid("invalid softmax input"));
    }
    let max = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or_else(|| Error::invariant("nonempty logits"))?;
    let mut exp: Vec<f64> = logits
        .iter()
        .map(|x| ((f64::from(*x) - f64::from(max)) / f64::from(temperature)).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    for x in &mut exp {
        *x /= sum;
    }
    Ok(exp.into_iter().map(|x| x as f32).collect())
}

/// The log of the tempered softmax normalizer over every score, accumulated in F64.
///
/// # Errors
/// Returns an invalid-input error for empty or non-finite scores or an invalid temperature.
pub fn log_normalizer(logits: &[f32], temperature: f32) -> Result<f64> {
    if logits.is_empty()
        || logits.iter().any(|v| !v.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.0
    {
        return Err(Error::invalid("invalid softmax input"));
    }
    let scale = f64::from(temperature);
    let max = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .map(|max| f64::from(max) / scale)
        .ok_or_else(|| Error::invariant("nonempty logits"))?;
    let sum: f64 = logits
        .iter()
        .map(|x| (f64::from(*x) / scale - max).exp())
        .sum();
    Ok(max + sum.ln())
}
