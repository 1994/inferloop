use super::*;
/// Absolute tolerance smaller than the divergence of the mismatched candidate sample.
const ATOL_BELOW_DIVERGENCE: f64 = 0.01;
/// Candidate sample diverging from the unit reference beyond the absolute tolerance.
const DIVERGED_SAMPLE: f32 = 1.1;
/// Probability of each class in a uniform two-class distribution.
const UNIFORM_PROBABILITY: f32 = 0.5;
/// Calibration bin count of the known-values dataset.
const CALIBRATION_BINS: usize = 10;
/// Expected Brier score of the known-values dataset.
const EXPECTED_BRIER: f64 = 0.25;
/// Expected expected calibration error of the known-values dataset.
const EXPECTED_ECE: f64 = 0.25;
#[test]
fn rejects_nonfinite_and_detects_divergence() {
    assert!(compare(&[1.0], &[f32::NAN], 0.0, 0.0).is_err());
    assert!(
        !compare(&[1.0], &[DIVERGED_SAMPLE], ATOL_BELOW_DIVERGENCE, 0.0)
            .unwrap()
            .passed
    );
}
#[test]
fn calibration_known_values() {
    let r = calibration(
        &[
            vec![UNIFORM_PROBABILITY, UNIFORM_PROBABILITY],
            vec![0.0, 1.0],
        ],
        &[0, 1],
        CALIBRATION_BINS,
    )
    .unwrap();
    assert_eq!(r.brier, EXPECTED_BRIER);
    assert_eq!(r.accuracy, 1.0);
    assert_eq!(r.ece, EXPECTED_ECE);
}
#[test]
fn progress_guard_only_trips_for_feasible_runnable_work() {
    let mut g = ProgressGuard::new(2).unwrap();
    g.observe(1, false, false).unwrap();
    g.observe(1, true, false).unwrap();
    assert!(g.observe(1, true, false).is_err());
    g.observe(1, true, true).unwrap();
}
