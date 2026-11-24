use super::*;

#[test]
fn category_labels_match_op_census_names() {
    assert_eq!(category(&TensorOp::Linear), "linear");
    assert_eq!(
        category(&TensorOp::Norm {
            epsilon: 0.0,
            offset: 0.0,
            head_dim: 1
        }),
        "norm"
    );
    assert_eq!(
        category(&TensorOp::GatedNorm {
            head_dim: 1,
            epsilon: 0.0
        }),
        "gated_norm"
    );
    assert_eq!(category(&TensorOp::Silu), "silu");
    assert_eq!(category(&TensorOp::Multiply), "multiply");
    assert_eq!(category(&TensorOp::Add), "add");
    assert_eq!(
        category(&TensorOp::Split {
            widths: vec![],
            heads: 0
        }),
        "split"
    );
    assert_eq!(category(&TensorOp::Embedding), "embedding");
}
