//! Exact speculative rejection sampling over the effective, filtered distributions.
use infer_core::{Error, Result};

/// Absolute tolerance accepted when checking that a distribution sums to one.
const NORMALIZATION_TOLERANCE: f64 = 1e-8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verification {
    Accepted(u32),
    Replaced(u32),
}

/// Draw from nonnegative unnormalized weights.
/// # Errors
/// Rejects invalid weights, empty mass or a uniform value outside [0, 1).
pub fn draw_distribution(weights: &[f64], uniform: f64) -> Result<u32> {
    if !uniform.is_finite()
        || !(0.0..1.0).contains(&uniform)
        || weights.is_empty()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return Err(Error::invalid("invalid speculative distribution"));
    }
    let sum: f64 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::invalid("empty speculative probability mass"));
    }
    let threshold = uniform * sum;
    let mut cumulative = 0.0;
    let mut last = 0;
    for (index, weight) in weights.iter().enumerate() {
        if *weight > 0.0 {
            last = index;
        }
        cumulative += weight;
        if threshold < cumulative {
            return u32::try_from(index).map_err(|_| Error::invalid("vocabulary overflow"));
        }
    }
    u32::try_from(last).map_err(|_| Error::invalid("vocabulary overflow"))
}

/// How many tokens a draft may propose: bounded by its depth, the verification lanes left after the
/// submitted token, and the room left in the sequence.
#[must_use]
pub const fn draft_budget(depth: usize, lanes: usize, position: usize, capacity: usize) -> usize {
    let room = capacity.saturating_sub(position.saturating_add(1));
    let lanes = lanes.saturating_sub(1);
    let bound = if lanes < room { lanes } else { room };
    if depth < bound { depth } else { bound }
}

/// Whether a draft may propose under this temperature. A rejection under a non-zero temperature
/// needs residual sampling that the output stage cannot reproduce from plain target logits.
#[must_use]
pub fn draft_may_propose(temperature: f32) -> bool {
    temperature == 0.0
}

/// The longest candidate prefix the target selects itself, stopping after an end token.
///
/// `select` answers the target's choice for one candidate offset, given the tokens generated so
/// far; `generated` grows by each accepted candidate.
/// # Errors
/// Returns what `select` returns.
pub fn accepted_prefix(
    candidates: &[u32],
    generated: &mut Vec<u32>,
    mut select: impl FnMut(usize, &[u32]) -> Result<u32>,
    is_end: impl Fn(u32) -> bool,
) -> Result<Vec<u32>> {
    let mut accepted = Vec::new();
    for (offset, &candidate) in candidates.iter().enumerate() {
        if select(offset, generated)? != candidate {
            break;
        }
        accepted.push(candidate);
        generated.push(candidate);
        if is_end(candidate) {
            break;
        }
    }
    Ok(accepted)
}

/// Accept q's proposal with min(1,p/q), otherwise sample normalized max(p-q,0).
/// # Errors
/// Rejects non-normalized distributions, invalid proposals or invalid random inputs.
pub fn verify_draft(
    p: &[f64],
    q: &[f64],
    proposal: u32,
    acceptance: f64,
    correction: f64,
) -> Result<Verification> {
    let index = proposal as usize;
    if p.len() != q.len()
        || index >= p.len()
        || !acceptance.is_finite()
        || !(0.0..1.0).contains(&acceptance)
        || !correction.is_finite()
        || !(0.0..1.0).contains(&correction)
    {
        return Err(Error::invalid("invalid speculative verification input"));
    }
    for distribution in [p, q] {
        if distribution.iter().any(|w| !w.is_finite() || *w < 0.0)
            || (distribution.iter().sum::<f64>() - 1.0).abs() > NORMALIZATION_TOLERANCE
        {
            return Err(Error::invalid(
                "speculative distributions must be normalized",
            ));
        }
    }
    if q[index] == 0.0 {
        return Err(Error::invalid("draft proposed a zero-probability token"));
    }
    if acceptance < (p[index] / q[index]).min(1.0) {
        return Ok(Verification::Accepted(proposal));
    }
    let residual: Vec<_> = p.iter().zip(q).map(|(p, q)| (p - q).max(0.0)).collect();
    draw_distribution(&residual, correction).map(Verification::Replaced)
}

/// How a loaded draft runs beside the target's slot pool, which decides the pool policy that
/// differs by draft. A new draft picks one of these instead of adding executor branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftPooling {
    /// No draft, or one whose proposals ride inside each sequence's own target timeline.
    Separate,
    /// One pooled proposal graph drafts for every slot at once.
    Pooled,
}

/// Fewest sequences a slot pool must hold to beat the private single-sequence path.
const SHARED_POOL_WIDTH: usize = 2;

impl DraftPooling {
    /// The narrowest slot pool worth capturing. A pooled proposal only runs on a pool, so one
    /// slot is enough; otherwise a pool must hold two sequences to beat the private path.
    #[must_use]
    pub const fn minimum_pool_width(self) -> usize {
        match self {
            Self::Separate => SHARED_POOL_WIDTH,
            Self::Pooled => 1,
        }
    }

    /// Whether a slot pool is worth warming when at most `states` sequences may be resident.
    #[must_use]
    pub const fn warms_pool(self, states: usize) -> bool {
        states >= self.minimum_pool_width()
    }

    /// Whether the private path's single-sequence roofline may stand in for a pool. A pooled
    /// proposal needs the pool whatever the roofline says.
    #[must_use]
    pub const fn serial_roofline(self) -> bool {
        matches!(self, Self::Separate)
    }

    /// Whether this draft only runs on a slot pool, so a request it cannot fund fails instead of
    /// running privately.
    #[must_use]
    pub const fn requires_pool(self) -> bool {
        matches!(self, Self::Pooled)
    }

    /// Whether sequences run on the private path instead of a slot pool, given how many may be
    /// resident and the measured single-sequence roofline bound.
    #[must_use]
    pub const fn private(self, states: usize, roofline: Option<usize>) -> bool {
        !self.requires_pool() && (!self.warms_pool(states) || matches!(roofline, Some(1)))
    }

    /// Whether concurrent prompts may share one multi-sequence prompt graph. A pooled proposal
    /// follows each prompt through that prompt's own graph and its feature rows.
    #[must_use]
    pub const fn prompt_burst(self) -> bool {
        matches!(self, Self::Separate)
    }
}

#[cfg(test)]
#[path = "../tests/unit/speculative.rs"]
mod tests;
