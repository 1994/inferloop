//! Paired device timing: two workloads interleaved rep by rep, each rep timed on the stream.
use super::Device;
use crate::tile::backend_error;
use cutile::bench::{BenchOptions, Measurement, do_bench_paired};
use cutile::error::tensor_error;
use infer_core::Result;
use std::time::Duration;

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
pub struct Samples(Measurement);

impl Samples {
    /// Every timed rep, in milliseconds.
    #[must_use]
    pub fn times_ms(&self) -> &[f32] {
        self.0.times_ms()
    }

    /// The median rep, in milliseconds.
    #[must_use]
    pub fn median_ms(&self) -> f32 {
        self.0.median_ms()
    }
}

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
        let options = BenchOptions {
            warmup: timing.warmup,
            rep: timing.rep,
            min_reps: timing.min_reps,
            max_reps: timing.max_reps,
            clear_l2: timing.clear_l2,
        };
        let (a, b) = do_bench_paired(
            self.stream(),
            &options,
            |_| first().map_err(|error| tensor_error(&error.to_string())),
            |_| second().map_err(|error| tensor_error(&error.to_string())),
        )
        .map_err(backend_error)?;
        Ok((Samples(a), Samples(b)))
    }
}
