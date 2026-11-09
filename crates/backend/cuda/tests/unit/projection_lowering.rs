use super::*;

const SHAPE: [usize; 3] = [4, 64, 4096];
const SCALE: f32 = 0.5;

fn rows(mode: Mode, weight: Format) -> Lowering {
    lower(weight, None, mode, SHAPE).unwrap()
}

#[test]
fn quantized_activations_lower_the_same_way_in_every_mode_that_accepts_them() {
    let fp4 = Some(ActivationQuantization::Fp4(SCALE));
    let fp8 = Some(ActivationQuantization::Fp8Token);
    let modes = [
        Mode::Single,
        Mode::Rows { arithmetic_rows: 4 },
        Mode::Slots {
            arithmetic_rows: 4,
            priced: false,
            pool: PoolDense::Native,
        },
        Mode::Prompt,
        Mode::Logits,
    ];
    for mode in modes {
        assert_eq!(
            lower(Format::Fp4, fp4, mode, SHAPE).unwrap(),
            Lowering::Quantized(Quantized::Fp4(SCALE)),
            "{mode:?}"
        );
        for weight in [Format::Fp8, Format::Fp8Block] {
            assert_eq!(
                lower(weight, fp8, mode, SHAPE).unwrap(),
                Lowering::Quantized(Quantized::Fp8Token),
                "{mode:?}"
            );
        }
    }
}

#[test]
fn a_mismatched_activation_format_is_an_error_only_where_nothing_else_can_record() {
    let fp8 = Some(ActivationQuantization::Fp8Token);
    assert!(lower(Format::Fp4, fp8, Mode::Single, SHAPE).is_err());
    assert!(lower(Format::Dense, fp8, Mode::Logits, SHAPE).is_err());
    assert!(lower(Format::Dense, None, Mode::Logits, SHAPE).is_err());
    let gathered = lower(Format::Fp4, fp8, Mode::Rows { arithmetic_rows: 4 }, SHAPE).unwrap();
    assert!(matches!(gathered, Lowering::Rows(_)));
    assert_eq!(
        lower(Format::Fp4, fp8, Mode::Prompt, SHAPE).unwrap(),
        Lowering::PromptFp4
    );
}

#[test]
fn prompt_tiles_lower_by_weight_format() {
    assert_eq!(rows(Mode::Prompt, Format::Dense), Lowering::PromptDense);
    assert_eq!(rows(Mode::Prompt, Format::Fp8), Lowering::PromptFp8);
    assert_eq!(rows(Mode::Prompt, Format::Fp4), Lowering::PromptFp4);
    assert!(lower(Format::Fp8Block, None, Mode::Prompt, SHAPE).is_err());
    assert_eq!(rows(Mode::Single, Format::Fp4), Lowering::Matvec);
}

#[test]
fn gathered_rows_keep_the_narrow_kernel_only_for_dense_and_channel_fp8() {
    let narrow = Mode::Rows {
        arithmetic_rows: SMALL_GEMV_MAX_ROWS + 1,
    };
    assert_eq!(
        rows(narrow, Format::Dense),
        Lowering::Rows(RowsKernel::Small)
    );
    assert_eq!(rows(narrow, Format::Fp8), Lowering::Rows(RowsKernel::Small));
    assert!(matches!(
        rows(narrow, Format::Fp4),
        Lowering::Rows(RowsKernel::Tiled { .. })
    ));
    let wide_head = [4, 4096 * (GEMV_WIDE_OUTPUT_RATIO + 1), 4096];
    let tiled = lower(
        Format::Dense,
        None,
        Mode::Rows { arithmetic_rows: 4 },
        wide_head,
    )
    .unwrap();
    assert!(matches!(tiled, Lowering::Rows(RowsKernel::Tiled { .. })));
}

#[test]
fn only_slot_rows_without_quantized_activations_take_the_pooled_dense_kernel() {
    let slots = |pool| Mode::Slots {
        arithmetic_rows: 4,
        priced: false,
        pool,
    };
    assert!(!matches!(
        rows(slots(PoolDense::Native), Format::Dense),
        Lowering::PoolDense(_)
    ));
    assert!(!matches!(
        rows(slots(PoolDense::Gemv), Format::Fp8),
        Lowering::PoolDense(_)
    ));
    let fp4 = Some(ActivationQuantization::Fp4(SCALE));
    assert_eq!(
        lower(Format::Fp4, fp4, slots(PoolDense::Gemv), SHAPE).unwrap(),
        Lowering::Quantized(Quantized::Fp4(SCALE))
    );
}

#[test]
fn draft_projections_split_only_channel_fp8_with_a_partial_buffer() {
    let draft = |split, weight| rows(Mode::Draft { split }, weight);
    assert_eq!(draft(true, Format::Fp8), Lowering::DraftSplit);
    assert_eq!(draft(false, Format::Fp8), Lowering::Draft);
    for weight in [Format::Dense, Format::Fp8Block, Format::Fp4] {
        assert_eq!(draft(true, weight), Lowering::Draft);
    }
}
