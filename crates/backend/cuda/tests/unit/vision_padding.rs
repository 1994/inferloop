//! Host-only audit of the vision head-width padding contract.
//!
//! The projection kernel partitions columns in fixed 64-wide tiles. A head whose real width is
//! not a multiple of 64 must be padded to the next 64 boundary before any tile kernel can read
//! it. This test records the exact retained weight bytes for the 27B vision geometry so a future
//! compact-layout change has a concrete baseline; it does not claim a GPU result.
use infer_spi::{HiddenActivation, ModalityEncoder};

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
fn head_width_72_is_padded_to_the_64_column_tile() {
    let encoder = encoder();
    let head = encoder.hidden_size / encoder.heads;
    assert_eq!(head, 72);
    let half = super::weights::half_tile(&encoder).expect("half tile");
    let width = half * 2;
    assert_eq!(half, 64);
    assert_eq!(width, 128);
    assert_eq!(head.next_multiple_of(super::program::column_tile()), 128);

    let real_rows = u64::try_from(encoder.heads * head).expect("rows");
    let padded_rows = u64::try_from(encoder.heads * width).expect("padded rows");
    let hidden = u64::try_from(encoder.hidden_size).expect("hidden");
    let depth = u64::try_from(encoder.depth).expect("depth");
    let bytes = 2u64;
    let real = (3 * real_rows * hidden + hidden * real_rows) * bytes * depth;
    let padded = (3 * padded_rows * hidden + hidden * padded_rows) * bytes * depth;
    let overhead = padded - real;
    assert!(overhead > 100 * 1024 * 1024, "overhead {overhead}");
    eprintln!(
        "vision head padding: real={real} padded={padded} overhead={overhead} bytes (depth={depth})"
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
