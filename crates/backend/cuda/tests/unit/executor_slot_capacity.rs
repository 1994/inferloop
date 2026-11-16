use super::widest_capacity;
use infer_core::{Error, Result};

/// Funded width of a pool whose fixed cost and one complete context share `tokens` of KV.
fn funded(tokens: usize, maximum: usize) -> impl FnMut(usize) -> Result<usize> {
    move |capacity| Ok((tokens.saturating_sub(capacity) / 1024).min(maximum))
}

#[test]
fn capacity_grows_until_it_would_cost_width() -> Result<()> {
    // Width saturates at 8 until one full context eats into the eight 1K-token slots.
    assert_eq!(widest_capacity(262_144, funded(16_384, 8))?, (8_192, 8));
    // A larger budget keeps the same width at a longer per-slot context.
    assert_eq!(widest_capacity(262_144, funded(73_728, 8))?, (65_536, 8));
    Ok(())
}

#[test]
fn capacity_reaches_the_model_context_when_width_is_free() -> Result<()> {
    assert_eq!(widest_capacity(40_000, funded(1 << 20, 8))?, (40_000, 8));
    Ok(())
}

#[test]
fn unfunded_pool_leaves_the_quote_as_the_only_gate() -> Result<()> {
    assert_eq!(widest_capacity(32_768, |_| Ok(0))?, (32_768, 0));
    Ok(())
}

#[test]
fn every_probe_stays_inside_the_model_context() -> Result<()> {
    let mut probes = Vec::new();
    let found = widest_capacity(24_000, |capacity| {
        probes.push(capacity);
        Ok(usize::from(capacity <= 2_048) + 1)
    })?;
    assert_eq!(found, (2_048, 2));
    assert!(probes.iter().all(|&capacity| capacity <= 24_000));
    Ok(())
}

#[test]
fn short_contexts_and_failures_propagate() -> Result<()> {
    assert_eq!(widest_capacity(16, funded(1 << 20, 8))?, (16, 8));
    assert!(widest_capacity(0, funded(1 << 20, 8)).is_err());
    assert!(widest_capacity(4_096, |_| Err(Error::invalid("quote"))).is_err());
    Ok(())
}
