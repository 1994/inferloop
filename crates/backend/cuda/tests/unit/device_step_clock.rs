use super::millis_to_micros;

#[test]
fn only_usable_samples_reach_the_cost_model() {
    assert_eq!(millis_to_micros(0.0), None);
    assert_eq!(millis_to_micros(-1.0), None);
    assert_eq!(millis_to_micros(f32::NAN), None);
    assert_eq!(millis_to_micros(f32::INFINITY), None);
    assert_eq!(millis_to_micros(2.0), Some(2000));
}

#[test]
fn a_sub_microsecond_step_still_counts() {
    assert_eq!(millis_to_micros(0.0001), Some(1));
}
