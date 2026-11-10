use super::*;
use crate::resident::attention_mma::catalog::Variant;

const SHORT_WINDOW: usize = ATTENTION_SPLIT_THRESHOLD - 1;
const PREPARED: Prepared = Prepared {
    split: true,
    grouped_verify: true,
};

fn prefill(selected: Choice, fp8: bool, fits: bool) -> Lowering {
    let mode = Mode::Prefill {
        selected: Some(selected),
        fp8,
        fits,
    };
    lower(mode, None, Prepared::default())
}

#[test]
fn decode_splits_only_prepared_geometries_over_a_long_enough_span() {
    assert_eq!(lower(Mode::Decode, None, PREPARED), Lowering::Split);
    assert_eq!(
        lower(Mode::Decode, Some(ATTENTION_SPLIT_THRESHOLD), PREPARED),
        Lowering::Split
    );
    assert_eq!(
        lower(Mode::Decode, Some(SHORT_WINDOW), PREPARED),
        Lowering::Decode
    );
    assert_eq!(
        lower(Mode::Decode, None, Prepared::default()),
        Lowering::Decode
    );
}

#[test]
fn verification_groups_exactly_three_lanes_with_shared_operands() {
    let verify = |lanes, operands| Mode::Verify { lanes, operands };
    assert_eq!(
        lower(verify(VERIFY_LANES, true), None, PREPARED),
        Lowering::Grouped
    );
    for (mode, window, prepared) in [
        (verify(VERIFY_LANES - 1, true), None, PREPARED),
        (verify(VERIFY_LANES, false), None, PREPARED),
        (verify(VERIFY_LANES, true), Some(SHORT_WINDOW), PREPARED),
        (
            verify(VERIFY_LANES, true),
            None,
            Prepared {
                grouped_verify: false,
                ..PREPARED
            },
        ),
    ] {
        assert_eq!(lower(mode, window, prepared), Lowering::PerLane, "{mode:?}");
    }
}

#[test]
fn a_selected_prefill_kernel_records_only_when_it_fits_the_pool() {
    let mma = Choice::Mma {
        variant: Variant::Sm120,
    };
    for kernel in [mma, Choice::MmaF16] {
        assert_eq!(prefill(kernel, true, true), Lowering::Mma);
        assert_eq!(prefill(kernel, true, false), Lowering::Prefill);
    }
    for kernel in [
        Choice::RustSimt,
        Choice::SharedKv,
        Choice::VectorSharedKv,
        Choice::VectorSharedKvCombined,
    ] {
        assert_eq!(prefill(kernel, true, true), Lowering::Simt);
        assert_eq!(prefill(kernel, true, false), Lowering::Prefill);
    }
}

#[test]
fn the_query_tile_applies_only_to_quantized_pages() {
    let tiled = Choice::Tiled { query_tile: 32 };
    assert_eq!(prefill(tiled, true, true), Lowering::Tiled(32));
    assert_eq!(prefill(tiled, false, true), Lowering::Prefill);
    assert_eq!(prefill(Choice::Simt, true, true), Lowering::Prefill);
    let untuned = Mode::Prefill {
        selected: None,
        fp8: true,
        fits: true,
    };
    assert_eq!(lower(untuned, None, PREPARED), Lowering::Prefill);
}
