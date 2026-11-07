//! Paired device timing: two workloads interleaved rep by rep, each rep timed on the stream.
use super::{Attribute, Device};
use infer_core::{Error, Result};
use std::time::{Duration, Instant};

/// How long [`Device::time_paired`] measures.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// Untimed warmup budget; at least one warmup of each side always runs.
    pub warmup: Duration,
    /// Budget the timed reps should roughly fill.
    pub rep: Duration,
    /// Fewest timed reps.
    pub min_reps: usize,
    /// Most timed reps.
    pub max_reps: usize,
    /// Evict the L2 cache before every timed rep, so reps measure cold-cache behavior.
    pub clear_l2: bool,
}

/// The timed reps of one side of a pair.
#[derive(Debug)]
pub struct Samples {
    times: Vec<f32>,
    median: f32,
}

impl Samples {
    fn new(times: Vec<f32>) -> Self {
        let mut sorted = times.clone();
        sorted.sort_by(f32::total_cmp);
        let middle = sorted.len() / 2;
        let median = match (sorted.get(middle.wrapping_sub(1)), sorted.get(middle)) {
            (Some(low), Some(high)) if sorted.len().is_multiple_of(2) => f32::midpoint(*low, *high),
            (_, Some(middle)) => *middle,
            _ => f32::NAN,
        };
        Self { times, median }
    }

    /// Every timed rep, in milliseconds.
    #[must_use]
    pub fn times_ms(&self) -> &[f32] {
        &self.times
    }

    /// The median rep, in milliseconds.
    #[must_use]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "Not const in the cuTile build, so host callers compile under either source"
    )]
    pub fn median_ms(&self) -> f32 {
        self.median
    }
}

/// Bytes zeroed to evict L2, as a multiple of its size so no line survives.
const L2_FLUSH_FACTOR: usize = 2;

impl Device {
    /// Time `first` against `second`, alternating them so drift affects both alike.
    /// # Errors
    /// Returns the first failure either side reports, or a timing failure.
    pub fn time_paired(
        &self,
        timing: &Timing,
        mut first: impl FnMut() -> Result<()>,
        mut second: impl FnMut() -> Result<()>,
    ) -> Result<(Samples, Samples)> {
        if timing.min_reps == 0 || timing.min_reps > timing.max_reps {
            return Err(Error::invalid("paired timing rep bounds"));
        }
        let flush = if timing.clear_l2 {
            let l2 = usize::try_from(self.attribute(Attribute::L2CacheBytes)?)
                .map_err(|_| Error::invalid("L2 size exceeds usize"))?;
            Some(self.zeros::<u8>(l2.saturating_mul(L2_FLUSH_FACTOR))?)
        } else {
            None
        };
        let (start, end) = (self.event()?, self.event()?);
        let timed = |side: &mut dyn FnMut() -> Result<()>| -> Result<f32> {
            if let Some(flush) = &flush {
                self.zero(flush)?;
            }
            start.record(self)?;
            side()?;
            end.record(self)?;
            self.synchronize()?;
            start.elapsed_ms(&end)
        };
        let began = Instant::now();
        let mut pair_ms = timed(&mut first)? + timed(&mut second)?;
        while began.elapsed() < timing.warmup {
            pair_ms = timed(&mut first)? + timed(&mut second)?;
        }
        let budget_ms = timing.rep.as_secs_f32() * MS_PER_SECOND;
        let reps = estimated_reps(budget_ms, pair_ms).clamp(timing.min_reps, timing.max_reps);
        let (mut a, mut b) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
        for _ in 0..reps {
            a.push(timed(&mut first)?);
            b.push(timed(&mut second)?);
        }
        Ok((Samples::new(a), Samples::new(b)))
    }
}

/// Milliseconds in one second.
const MS_PER_SECOND: f32 = 1000.0;

/// How many pairs of `pair_ms` fill `budget_ms`; at least one, and a whole count.
fn estimated_reps(budget_ms: f32, pair_ms: f32) -> usize {
    if pair_ms > 0.0 && budget_ms.is_finite() {
        // Saturating float-to-int: a huge ratio becomes `usize::MAX`, then the bounds clamp it.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "A positive, finite rep count, saturated by the cast and then clamped"
        )]
        let reps = (budget_ms / pair_ms).ceil() as usize;
        reps.max(1)
    } else {
        1
    }
}
