use infer_core::StepId;

use super::*;
#[test]
fn metadata_cannot_be_recycled_before_completion() {
    let mut slots = SubmissionSlots::new(1).unwrap();
    let a = StepId::new(1).unwrap();
    let b = StepId::new(2).unwrap();
    assert_eq!(slots.acquire(a).unwrap(), 0);
    assert!(slots.acquire(b).is_err());
    assert!(slots.release(0, b).is_err());
    slots.release(0, a).unwrap();
    assert_eq!(slots.acquire(b).unwrap(), 0);
}
