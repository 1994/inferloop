use super::{AttentionProfile, PRECISE_PREFIX_POSITIONS, precise_prefix_from};

#[test]
fn precise_prefix_defaults_to_full_bf16_and_restores_explicit_positions() {
    assert_eq!(PRECISE_PREFIX_POSITIONS, 0);
    assert_eq!(precise_prefix_from(None), 0);
    assert_eq!(precise_prefix_from(Some("2048")), 2048);
    assert_eq!(precise_prefix_from(Some(" 1024 ")), 1024);
    assert_eq!(precise_prefix_from(Some("0")), 0);
    for invalid in ["", "on", "-1", "2k"] {
        assert_eq!(precise_prefix_from(Some(invalid)), PRECISE_PREFIX_POSITIONS);
    }
}

use infer_core::{OpId, TensorId};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};

fn graph() -> DataflowGraph {
    DataflowGraph {
        nodes: vec![TensorNode {
            id: OpId::ONE,
            layer: None,
            op: TensorOp::Attention {
                query_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                window: None,
            },
            inputs: vec![],
            outputs: vec![],
            states: vec![TensorId::ONE],
        }],
        ..DataflowGraph::default()
    }
}
#[test]
fn fast_prefill_requires_an_available_artifact_and_every_attention_geometry() {
    let mut g = graph();
    assert_eq!(
        AttentionProfile::supports_bf16_prefill("sm_120", &g, 0),
        cfg!(target_arch = "x86_64")
    );
    assert_eq!(
        AttentionProfile::supports_bf16_prefill("sm_120", &g, 1024),
        cfg!(target_arch = "x86_64")
    );
    assert_eq!(
        AttentionProfile::supports_bf16_prefill("sm_120", &g, 2048),
        cfg!(target_arch = "x86_64")
    );
    for width in [1, 2, 3, 32, 65] {
        assert!(!AttentionProfile::supports_bf16_prefill(
            "sm_120", &g, width
        ));
    }
    assert!(!AttentionProfile::supports_bf16_prefill("sm_80", &g, 2048));
    let mut unsupported = g.nodes[0].clone();
    unsupported.op = TensorOp::Attention {
        query_heads: 8,
        kv_heads: 4,
        head_dim: 256,
        window: None,
    };
    g.nodes.push(unsupported);
    assert!(!AttentionProfile::supports_bf16_prefill("sm_120", &g, 2048));
    g.nodes.clear();
    assert!(!AttentionProfile::supports_bf16_prefill("sm_120", &g, 2048));
}
#[test]
fn sliding_attention_and_missing_state_do_not_opt_targets_into_the_fast_pair() {
    let mut g = graph();
    g.nodes[0].states.clear();
    assert!(!AttentionProfile::supports_bf16_prefill("sm_120", &g, 2048));
    g = graph();
    g.nodes[0].op = TensorOp::Attention {
        query_heads: 24,
        kv_heads: 4,
        head_dim: 256,
        window: Some(4096),
    };
    assert!(!AttentionProfile::supports_bf16_prefill("sm_120", &g, 2048));
}
