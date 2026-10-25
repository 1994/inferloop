use super::add_norm_prefix;
use super::{apply_control, prompt_record};
use infer_core::{OpId, TensorId};
use infer_ir::{TensorNode, TensorOp};

fn node(op: TensorOp, inputs: &[u64], outputs: &[u64]) -> TensorNode {
    TensorNode {
        id: OpId::new(outputs[0]).expect("op id"),
        layer: Some(1),
        op,
        inputs: inputs
            .iter()
            .map(|id| TensorId::new(*id).expect("tensor id"))
            .collect(),
        outputs: outputs
            .iter()
            .map(|id| TensorId::new(*id).expect("tensor id"))
            .collect(),
        states: Vec::new(),
    }
}

#[test]
fn ordinary_norm_fusion_continues_and_add_norm_skips_the_consumed_norm() {
    let mut skip = Some(42);
    assert!(!apply_control(prompt_record(false, false, 7), &mut skip));
    assert_eq!(
        skip,
        Some(42),
        "unhandled preserves an unrelated pending skip"
    );

    assert!(apply_control(prompt_record(false, true, 7), &mut skip));
    assert_eq!(
        skip,
        Some(42),
        "ordinary handled norm must not invent a skip"
    );

    assert!(apply_control(prompt_record(true, true, 7), &mut skip));
    assert_eq!(skip, Some(8), "Add+Norm consumes the following Norm");

    assert!(!apply_control(prompt_record(true, false, 7), &mut skip));
    assert_eq!(
        skip,
        Some(8),
        "unsupported preserves the prior pending skip"
    );
}

#[test]
fn add_norm_prefix_requires_one_consumer_and_the_immediate_norm() {
    let add = node(TensorOp::Add, &[1, 2], &[3]);
    let norm = node(
        TensorOp::Norm {
            head_dim: 128,
            epsilon: 1.0e-6,
            offset: 0.0,
        },
        &[3, 4],
        &[5],
    );
    let linear = node(TensorOp::Linear, &[5, 6], &[7]);
    let nodes = vec![add.clone(), norm.clone(), linear.clone()];
    assert!(add_norm_prefix(&nodes, 0, [None, None]));
    assert!(!add_norm_prefix(
        &nodes,
        0,
        [Some(TensorId::new(3).unwrap()), None]
    ));

    let mut extra = nodes;
    extra.push(node(TensorOp::Silu, &[3], &[8]));
    assert!(!add_norm_prefix(&extra, 0, [None, None]));

    let not_immediate = vec![add, linear, norm];
    assert!(!add_norm_prefix(&not_immediate, 0, [None, None]));
}

#[test]
fn add_norm_max_kernel_compiles_with_the_norm_tile() {
    use crate::resident::norm_maximum_kernel::kernels;
    use cutile::cutile_compiler::compile_api::KernelCompiler;
    KernelCompiler::new(kernels::__module_ast_self, "kernels", "add_norm_max")
        .target("sm_120")
        .generics(vec!["128".to_string(), "128".to_string()])
        .strides(&[
            ("out", &[128, 1]),
            ("scale", &[1, 1]),
            ("x", &[128, 1]),
            ("residual", &[128, 1]),
            ("weight", &[1]),
        ])
        .compile()
        .expect("add_norm_max compiles");
}
