use super::*;

const WY_BLOCK: usize = 32;
const DELTA: Delta = Delta {
    key_heads: 16,
    value_heads: 48,
    key_dim: ORDERED_DIM,
    value_dim: ORDERED_DIM,
};
const CHUNK: Chunk = Chunk {
    lanes: 64,
    dynamic: true,
    scratch: true,
    prefer_wy: false,
    wy_block: None,
    pinned: Pinned::No,
};

fn delta_op(delta: Delta) -> TensorOp {
    TensorOp::Delta {
        key_heads: delta.key_heads,
        value_heads: delta.value_heads,
        key_dim: delta.key_dim,
        value_dim: delta.value_dim,
    }
}

#[test]
fn only_geometries_a_resident_kernel_addresses_are_recurrences() {
    assert_eq!(
        Recurrence::of(&delta_op(DELTA)).unwrap(),
        Some(Recurrence::Delta(DELTA))
    );
    let conv = TensorOp::Conv {
        channels: 8,
        kernel: CONV_KERNEL_SIZE,
    };
    assert_eq!(
        Recurrence::of(&conv).unwrap(),
        Some(Recurrence::Conv { channels: 8 })
    );
    assert!(
        Recurrence::of(&TensorOp::Conv {
            channels: 8,
            kernel: CONV_KERNEL_SIZE + 1,
        })
        .is_err()
    );
    for bad in [
        Delta {
            value_dim: ORDERED_DIM * 2,
            ..DELTA
        },
        Delta {
            key_dim: 96,
            value_dim: 96,
            ..DELTA
        },
        Delta {
            key_heads: 0,
            ..DELTA
        },
        Delta {
            value_heads: 47,
            ..DELTA
        },
    ] {
        assert!(Recurrence::of(&delta_op(bad)).is_err(), "{bad:?}");
    }
    assert_eq!(Recurrence::of(&TensorOp::Add).unwrap(), None);
}

#[test]
fn decode_steps_one_lane_and_the_pooled_graph_advances_every_lane() {
    for recurrence in [Recurrence::Conv { channels: 8 }, Recurrence::Delta(DELTA)] {
        assert_eq!(lower(recurrence, Mode::Lane), Lowering::Step);
        assert_eq!(lower(recurrence, Mode::Pooled), Lowering::Lanes);
    }
}

#[test]
fn a_prompt_chunk_keeps_state_across_rows_only_for_dynamic_inputs_it_prepared() {
    let conv = Recurrence::Conv { channels: 8 };
    assert_eq!(lower(conv, Mode::Prompt(CHUNK)), Lowering::ConvPrefill);
    let delta = Recurrence::Delta(DELTA);
    assert_eq!(
        lower(delta, Mode::Prompt(CHUNK)),
        Lowering::Chunked {
            block: Block::Fixed(ORDERED_BLOCK),
            wy: false,
        }
    );
    for chunk in [
        Chunk {
            dynamic: false,
            ..CHUNK
        },
        Chunk {
            pinned: Pinned::Rows,
            ..CHUNK
        },
    ] {
        assert_eq!(lower(conv, Mode::Prompt(chunk)), Lowering::Rows);
        assert_eq!(lower(delta, Mode::Prompt(chunk)), Lowering::Rows);
    }
    for chunk in [
        Chunk {
            scratch: false,
            ..CHUNK
        },
        Chunk {
            pinned: Pinned::RowsDelta,
            ..CHUNK
        },
        Chunk {
            prefer_wy: true,
            ..CHUNK
        },
    ] {
        assert_eq!(lower(delta, Mode::Prompt(chunk)), Lowering::Rows);
    }
    let rows_delta = Chunk {
        pinned: Pinned::RowsDelta,
        ..CHUNK
    };
    assert_eq!(lower(conv, Mode::Prompt(rows_delta)), Lowering::ConvPrefill);
}

#[test]
fn the_ordered_geometry_pins_its_block_and_others_take_the_tuned_or_split_block() {
    let unordered = Recurrence::Delta(Delta {
        key_dim: ORDERED_DIM * 2,
        value_dim: ORDERED_DIM * 2,
        ..DELTA
    });
    let tuned = Chunk {
        prefer_wy: true,
        wy_block: Some(WY_BLOCK),
        ..CHUNK
    };
    assert_eq!(
        lower(unordered, Mode::Prompt(tuned)),
        Lowering::Chunked {
            block: Block::Fixed(WY_BLOCK),
            wy: true,
        }
    );
    assert_eq!(
        lower(unordered, Mode::Prompt(CHUNK)),
        Lowering::Chunked {
            block: Block::Split,
            wy: false,
        }
    );
    let narrow = Chunk { lanes: 32, ..CHUNK };
    assert_eq!(
        lower(Recurrence::Delta(DELTA), Mode::Prompt(narrow)),
        Lowering::Chunked {
            block: Block::Split,
            wy: false,
        }
    );
}
