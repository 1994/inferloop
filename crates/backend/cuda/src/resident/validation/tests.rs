use super::check_kernel_geometry;
use crate::constants::CONV_KERNEL_SIZE;
use infer_core::ErrorCode;
use infer_ir::TensorOp;

const fn rope(head_dim: usize, rotary_dim: usize) -> TensorOp {
    TensorOp::Rope {
        heads: 4,
        head_dim,
        rotary_dim,
        theta: 10_000.0,
    }
}

const fn attention(query_heads: usize, kv_heads: usize, head_dim: usize) -> TensorOp {
    TensorOp::Attention {
        query_heads,
        kv_heads,
        head_dim,
        window: None,
    }
}

const fn delta(key_heads: usize, value_heads: usize, key_dim: usize, value_dim: usize) -> TensorOp {
    TensorOp::Delta {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    }
}

const fn conv(kernel: usize) -> TensorOp {
    TensorOp::Conv {
        channels: 1024,
        kernel,
    }
}

#[test]
fn recordable_geometry_loads() {
    for op in [
        delta(16, 48, 128, 128),
        conv(CONV_KERNEL_SIZE),
        rope(128, 128),
        rope(256, 64),
        attention(4, 2, 64),
        attention(4, 4, 128),
    ] {
        assert!(check_kernel_geometry(&op).is_ok(), "{op:?}");
    }
}

#[test]
fn geometry_without_a_kernel_is_refused_as_unsupported() {
    for op in [
        rope(48, 48),
        rope(100, 100),
        attention(4, 2, 48),
        attention(5, 2, 64),
        attention(4, 0, 64),
        delta(16, 48, 128, 64),
        delta(16, 48, 96, 96),
        delta(16, 40, 128, 128),
        delta(0, 48, 128, 128),
        conv(CONV_KERNEL_SIZE + 1),
    ] {
        let error = check_kernel_geometry(&op).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported, "{op:?}");
    }
}
