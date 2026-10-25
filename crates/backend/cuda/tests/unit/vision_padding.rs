//! Host audit and pure cuTile compile regression for the vision head-width contract.
//!
//! The projection kernel still partitions output columns in 64-wide tiles; the per-head
//! split-half tile is aligned to 16 lanes. This test records the exact retained weight bytes for
//! the 27B compact geometry and compiles the real projection/RoPE/attention/merge kernels through
//! the cuTile compiler without a device. It does not claim a GPU numeric or performance result.
use crate::attention::kernels::sdpa;
use cutile::cutile_compiler::compile_api::KernelCompiler;
use infer_models::vision::{merged_width, row_tile};
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
fn head_width_72_uses_the_device_valid_power_of_two_half_tile() {
    let encoder = encoder();
    let head = encoder.hidden_size / encoder.heads;
    assert_eq!(head, 72);
    let half = super::weights::half_tile(&encoder).expect("half tile");
    let width = half * 2;
    assert_eq!(half, 64);
    assert_eq!(width, 128);
    assert_eq!(width.next_power_of_two(), width);
    assert_eq!((encoder.heads * width) % super::program::column_tile(), 0);
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

fn compile_dense(
    hidden: i32,
    patch: i32,
    projections: i32,
    mlp: i32,
    merged: i32,
) -> Result<(), Box<dyn Error>> {
    // Production dense_bias views: (output columns, input columns, K).
    for (out, input_columns, k) in [
        (hidden, patch, patch),             // patch_embed
        (projections, hidden, hidden),      // q, k, v
        (hidden, projections, projections), // attention projection
        (mlp, hidden, hidden),              // block fc1
        (hidden, mlp, mlp),                 // block fc2
        (merged, hidden, hidden),           // merger fc1
        (hidden, merged, merged),           // merger fc2
    ] {
        KernelCompiler::new(
            super::kernels::encoder::__module_ast_self,
            "encoder",
            "dense_bias",
        )
        .target("sm_120")
        .generics(vec![k.to_string()])
        .strides(&[
            ("out", &[out, 1]),
            ("input", &[input_columns, 1]),
            ("weight", &[k, 1]),
            ("bias", &[1]),
        ])
        .compile()?;
    }
    // The existing 64-wide head (2B geometry): hidden 1024, projections 1024, mlp 4096.
    for (out, input_columns, k) in [
        (1024, patch, patch),
        (1024, 1024, 1024),
        (4096, 1024, 1024),
        (1024, 4096, 4096),
    ] {
        KernelCompiler::new(
            super::kernels::encoder::__module_ast_self,
            "encoder",
            "dense_bias",
        )
        .target("sm_120")
        .generics(vec![k.to_string()])
        .strides(&[
            ("out", &[out, 1]),
            ("input", &[input_columns, 1]),
            ("weight", &[k, 1]),
            ("bias", &[1]),
        ])
        .compile()?;
    }
    Ok(())
}

fn compile_rope(width: i32, projections: i32, heads: i32) -> Result<(), Box<dyn Error>> {
    // Production rope views: [padded, projections] input/output, [padded, width] tables.
    for (width, projections) in [(width, projections), (64, heads * 64)] {
        KernelCompiler::new(
            super::kernels::encoder::__module_ast_self,
            "encoder",
            "rope",
        )
        .target("sm_120")
        .generics(vec![(width / 2).to_string()])
        .strides(&[
            ("out", &[projections, 1]),
            ("input", &[projections, 1]),
            ("cos", &[width, 1]),
            ("sin", &[width, 1]),
        ])
        .compile()?;
    }
    Ok(())
}

fn compile_attention(width: i32, projections: i32, heads: i32) -> Result<(), Box<dyn Error>> {
    // Production attention views: [padded, projections], with tail tiles and masked paths.
    for (width, projections) in [(width, projections), (64, heads * 64)] {
        for (q, k, pipe, mask) in [
            (32, 32, 0, 3),
            (16, 16, 0, 3),
            (32, 16, 1, 3),
            (16, 32, 1, 1),
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
                    ("out", &[projections, 1]),
                    ("query", &[projections, 1]),
                    ("key", &[projections, 1]),
                    ("value", &[projections, 1]),
                ])
                .compile()?;
        }
    }
    Ok(())
}

fn compile_elementwise(hidden: i32, mlp: i32, merged: i32) -> Result<(), Box<dyn Error>> {
    // Production elementwise/merge views: [padded, columns].
    for n in [hidden, mlp, merged] {
        for entry in ["gelu_tanh", "gelu_erf"] {
            KernelCompiler::new(super::kernels::encoder::__module_ast_self, "encoder", entry)
                .target("sm_120")
                .generics(vec![n.to_string()])
                .strides(&[("out", &[n, 1]), ("x", &[n, 1])])
                .compile()?;
        }
        KernelCompiler::new(super::kernels::encoder::__module_ast_self, "encoder", "add")
            .target("sm_120")
            .generics(vec![n.to_string()])
            .strides(&[("out", &[n, 1]), ("left", &[n, 1]), ("right", &[n, 1])])
            .compile()?;
    }
    Ok(())
}

fn compile_layernorm(hidden: i32, row_tile: i32) -> Result<(), Box<dyn Error>> {
    // Production layernorm views: [rows, row_tile] input, [rows * row_tile] output.
    KernelCompiler::new(
        super::kernels::encoder::__module_ast_self,
        "encoder",
        "layernorm",
    )
    .target("sm_120")
    .generics(vec![hidden.to_string(), row_tile.to_string()])
    .strides(&[
        ("out", &[1]),
        ("x", &[row_tile, 1]),
        ("weight", &[1]),
        ("bias", &[1]),
    ])
    .compile()?;
    Ok(())
}

#[test]
fn compact_projection_rope_attention_and_merge_compile_without_a_device()
-> Result<(), Box<dyn Error>> {
    let encoder = encoder();
    let hidden = i32::try_from(encoder.hidden_size)?;
    let heads = i32::try_from(encoder.heads)?;
    let patch = i32::try_from(super::program::patch_width(&encoder)?)?;
    let mlp = i32::try_from(super::program::mlp_width(&encoder)?)?;
    let merged = i32::try_from(merged_width(&encoder)?)?;
    let row_tile = i32::try_from(row_tile(&encoder)?)?;
    let half = super::weights::half_tile(&encoder)?;
    let width = i32::try_from(half * 2)?;
    let projections = heads * width;
    eprintln!(
        "cuTile compact compile: hidden={hidden} heads={heads} half={half} width={width} projections={projections} patch={patch} mlp={mlp} merged={merged} row_tile={row_tile}"
    );
    compile_dense(hidden, patch, projections, mlp, merged)?;
    compile_rope(width, projections, heads)?;
    compile_attention(width, projections, heads)?;
    compile_elementwise(hidden, mlp, merged)?;
    compile_layernorm(hidden, row_tile)?;
    Ok(())
}
