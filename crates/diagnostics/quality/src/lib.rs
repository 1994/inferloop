//! Numerical verification, progress contracts and reproducible experiment evidence.
//! Cold-path verification, progress diagnostics, measurements and experiment gates.
#[cfg(test)]
mod tests;
mod verification;
pub use verification::{VerificationReport, compare};
mod calibration;
pub use calibration::{CalibrationReport, calibration};
mod progress;
pub use progress::ProgressGuard;
mod benchmark;
pub use benchmark::{BenchmarkReport, RequestMeasurement, benchmark};
mod experiment;
pub use experiment::{ExperimentVerdict, experiment};
