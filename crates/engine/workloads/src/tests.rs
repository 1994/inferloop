use infer_ir::{Pooling, Sampling};

use super::*;
#[test]
fn stable_softmax_and_sampling_tie_break() {
    let p = softmax(&[10000.0, 10000.0], 1.0).unwrap();
    assert_eq!(p, vec![0.5, 0.5]);
    assert_eq!(
        sample(&[2.0, 2.0, 0.0], &Sampling::default(), 1, 0).unwrap(),
        0
    );
    assert!(softmax(&[f32::NAN], 1.0).is_err());
}
#[test]
fn mean_pool_has_known_answer() {
    assert_eq!(
        pool(&[vec![1.0, 2.0], vec![3.0, 4.0]], Pooling::Mean).unwrap(),
        vec![2.0, 3.0]
    );
}
