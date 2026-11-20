//! Exact native aggregate updates. No assumption is made about arbitrary provider costs.
use super::{Shape, aggregate, bucket};
use infer_core::{Error, Result};
use infer_ir::{CostEstimate, CostQuery, ExecutionRole};
use infer_spi::BatchCostSummary;

/// `BatchCostSummary::roles` membership bit for prefill work.
const ROLE_PREFILL_BIT: u8 = 1;
/// `BatchCostSummary::roles` membership bit for decode work.
const ROLE_DECODE_BIT: u8 = 2;
/// `BatchCostSummary::roles` membership bit for forward work.
const ROLE_FORWARD_BIT: u8 = 4;

/// # Errors
/// Rejects invalid queries, nonmonotone feature replacement or overflowing native resources.
pub fn update_summary(
    mut summary: BatchCostSummary,
    previous: Option<CostQuery>,
    next: CostQuery,
) -> Result<BatchCostSummary> {
    if previous.is_some_and(|old| {
        old.role != next.role
            || old.workspace_bytes > next.workspace_bytes
            || old.context_tokens > next.context_tokens
    }) {
        return Err(Error::invariant("nonmonotone native summary replacement"));
    }
    let old = previous.map_or_else(|| Ok(CostEstimate::default()), |query| aggregate(&[query]))?;
    let new = aggregate(&[next])?;
    let error = || Error::invalid("batch summary cost overflow");
    let total = &mut summary.fallback;
    for (value, old, new) in [
        (&mut total.gpu_us, old.gpu_us, new.gpu_us),
        (&mut total.state_bytes, old.state_bytes, new.state_bytes),
        (&mut total.transfer_us, old.transfer_us, new.transfer_us),
        (&mut total.encoder_us, old.encoder_us, new.encoder_us),
    ] {
        *value = value
            .checked_sub(old)
            .and_then(|value| value.checked_add(new))
            .ok_or_else(error)?;
    }
    for (value, old, new) in [
        (
            &mut total.num_gpu_blocks,
            old.num_gpu_blocks,
            new.num_gpu_blocks,
        ),
        (
            &mut total.logical_pages,
            old.logical_pages,
            new.logical_pages,
        ),
    ] {
        *value = value
            .checked_sub(old)
            .and_then(|value| value.checked_add(new))
            .ok_or_else(error)?;
    }
    total.workspace_bytes = total.workspace_bytes.max(new.workspace_bytes);
    summary.tokens = summary
        .tokens
        .checked_sub(previous.map_or(0, |query| query.tokens))
        .and_then(|tokens| tokens.checked_add(next.tokens))
        .ok_or_else(error)?;
    summary.batch = summary
        .batch
        .checked_add(usize::from(previous.is_none()))
        .ok_or_else(error)?;
    summary.context_tokens = summary.context_tokens.max(next.context_tokens);
    summary.roles |= match next.role {
        ExecutionRole::Prefill => ROLE_PREFILL_BIT,
        ExecutionRole::Decode => ROLE_DECODE_BIT,
        ExecutionRole::Forward => ROLE_FORWARD_BIT,
        ExecutionRole::Mixed => return Err(Error::invalid("mixed summary query")),
    };
    Ok(summary)
}
pub(super) fn shape(summary: &BatchCostSummary) -> Shape {
    Shape {
        role: match summary.roles {
            ROLE_PREFILL_BIT => ExecutionRole::Prefill,
            ROLE_DECODE_BIT => ExecutionRole::Decode,
            ROLE_FORWARD_BIT => ExecutionRole::Forward,
            _ => ExecutionRole::Mixed,
        },
        tokens: bucket(summary.tokens),
        context: bucket(summary.context_tokens),
        batch: bucket(summary.batch),
    }
}

#[cfg(test)]
mod tests;
