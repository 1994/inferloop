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
