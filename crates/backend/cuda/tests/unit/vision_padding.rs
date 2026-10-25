//! Host-only audit of the vision head-width padding contract.
//!
//! The projection kernel partitions columns in fixed 64-wide tiles. A head whose real width is
//! not a multiple of 64 must be padded to the next 64 boundary before any tile kernel can read
//! it. This test records the exact retained weight bytes for the 27B vision geometry so a future
//! compact-layout change has a concrete baseline; it does not claim a GPU result.
use crate::attention::kernels::sdpa;
use infer_spi::{HiddenActivation, ModalityEncoder};
use std::error::Error;

fn encoder() -> ModalityEncoder {
    ModalityEncoder {
        prefix: "model.visual.".into(),
        depth: 24,
        hidden_size: 1152,
        intermediate_size: 4096,
        heads: 16,
        in_channels: 3,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 2048,
        position_embeddings: 2304,
        rope_theta: 10_000.0,
        norm_epsilon: 1e-6,
        hidden_activation: HiddenActivation::GeluTanh,
    }
}

#[test]
fn head_width_72_uses_a_compact_16_multiple_half_tile() {
    let encoder = encoder();
    let head = encoder.hidden_size / encoder.heads;
    assert_eq!(head, 72);
    let half = super::weights::half_tile(&encoder).expect("half tile");
    let width = half * 2;
    assert_eq!(half, 48);
    assert_eq!(width, 96);
    assert_eq!(width % 16, 0);
    assert_eq!((encoder.heads * width) % super::program::column_tile(), 0);

    let real_rows = u64::try_from(encoder.heads * head).expect("rows");
    let padded_rows = u64::try_from(encoder.heads * width).expect("padded rows");
    let hidden = u64::try_from(encoder.hidden_size).expect("hidden");
    let depth = u64::try_from(encoder.depth).expect("depth");
    let bytes = 2u64;
    let real = (3 * real_rows * hidden + hidden * real_rows) * bytes * depth;
    let padded = (3 * padded_rows * hidden + hidden * padded_rows) * bytes * depth;
    let overhead = padded - real;
    assert_eq!(real, 254_803_968);
    assert_eq!(padded, 339_738_624);
    assert_eq!(overhead, 84_934_656);
    eprintln!(
        "vision head padding: real={real} padded={padded} overhead={overhead} bytes (depth={depth}, width={width})"
    );
}

#[test]
fn a_64_multiple_head_has_no_padding() {
    let mut encoder = encoder();
    encoder.hidden_size = 1024;
    let half = super::weights::half_tile(&encoder).expect("half tile");
    assert_eq!(half * 2, 64);
    assert_eq!(encoder.hidden_size / encoder.heads, 64);
}

#[test]
fn compact_half_layout_preserves_split_half_order_and_padding() {
    let layout = super::weights::half_layout(72, 48);
    assert_eq!(layout.len(), 96);
    for (index, lane) in layout.iter().enumerate() {
        match index {
            0..36 => assert_eq!(*lane, Some(index)),
            36..48 | 84..96 => assert_eq!(*lane, None),
            48..84 => assert_eq!(*lane, Some(index - 12)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn compact_projection_rope_attention_and_merge_compile_without_a_device()
-> Result<(), Box<dyn Error>> {
    use cutile::cutile_compiler::compile_api::KernelCompiler;

    // Real compact geometry: 16 heads, real head 72, half 48, width 96, projection 1536.
    let (heads, half, width) = (16usize, 48i32, 96i32);
    assert_eq!(half * 2, width);
    let projection = heads * usize::try_from(width)?;
    assert_eq!(projection, 1536);
    eprintln!(
        "cuTile compact compile: head=72 heads={heads} half={half} width={width} projection={projection}"
    );
    for k in [1152i32, 1536, 4608] {
        KernelCompiler::new(
            super::kernels::encoder::__module_ast_self,
            "encoder",
            "dense_bias",
        )
        .target("sm_120")
        .generics(vec![k.to_string()])
        .strides(&[
            ("out", &[64, 1]),
            ("input", &[k, 1]),
            ("weight", &[k, 1]),
            ("bias", &[1]),
        ])
        .compile()?;
    }
    for hd in [half, 32] {
        KernelCompiler::new(
            super::kernels::encoder::__module_ast_self,
            "encoder",
            "rope",
        )
        .target("sm_120")
        .generics(vec![hd.to_string()])
        .strides(&[
            ("out", &[hd, 1]),
            ("input", &[2 * hd, 1]),
            ("cos", &[hd, 1]),
            ("sin", &[hd, 1]),
        ])
        .compile()?;
    }
    // Exact vision launch shape, plus smaller tail tiles and the masked video path.
    for (width, q, k, pipe, mask) in [
        (96, 32, 32, 0, 3),
        (96, 16, 16, 0, 3),
        (96, 32, 16, 1, 3),
        (96, 16, 32, 1, 1),
        (64, 32, 32, 0, 3),
        (64, 16, 16, 0, 0),
    ] {
        KernelCompiler::new(sdpa::__module_ast_self, "sdpa", "attention")
            .target("sm_120")
            .generics(
                [width, 1, q, k, pipe, 1, width, mask]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .strides(&[
                ("out", &[width, 1]),
                ("query", &[width, 1]),
                ("key", &[width, 1]),
                ("value", &[width, 1]),
            ])
            .compile()?;
    }
    // Merge/block elementwise pieces used by the compact pipeline.
    for n in [1152i32, 4608] {
        for entry in ["gelu_tanh", "gelu_erf"] {
            KernelCompiler::new(super::kernels::encoder::__module_ast_self, "encoder", entry)
                .target("sm_120")
                .generics(vec![n.to_string()])
                .strides(&[("out", &[64, 1]), ("x", &[n, 1])])
                .compile()?;
        }
        KernelCompiler::new(super::kernels::encoder::__module_ast_self, "encoder", "add")
            .target("sm_120")
            .generics(vec![n.to_string()])
            .strides(&[("out", &[64, 1]), ("left", &[n, 1]), ("right", &[n, 1])])
            .compile()?;
    }
    KernelCompiler::new(
        super::kernels::encoder::__module_ast_self,
        "encoder",
        "layernorm",
    )
    .target("sm_120")
    .generics(vec![1152.to_string(), 2048.to_string()])
    .strides(&[
        ("out", &[1]),
        ("x", &[2048, 1]),
        ("weight", &[1]),
        ("bias", &[1]),
    ])
    .compile()?;
    Ok(())
}
