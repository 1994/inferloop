use super::*;

const SCALE: f32 = 0.5;
const WIDE: usize = 5120;
const NORM: Norm = Norm {
    dimension: WIDE,
    epsilon: 1e-6,
    offset: 1.0,
};

fn consumer(format: Format, activation: Option<ActivationQuantization>) -> Consumer {
    Consumer {
        fuses: true,
        format,
        activation,
        fp8_input: true,
        planned_fp4: true,
    }
}

#[test]
fn row_tiles_need_a_narrow_power_of_two_row_and_whole_tiles_of_serving_rows() {
    let rows = NormLayout::Rows { tile: NORM_ROWS };
    let heads = NORM_ROWS * NORM_ROWS;
    assert_eq!(layout(NORM_ROWS_MAX_DIMENSION, heads), rows);
    assert_eq!(
        layout(NORM_ROWS_MAX_DIMENSION * 2, heads),
        NormLayout::Block
    );
    assert_eq!(
        layout(NORM_ROWS_MAX_DIMENSION - 1, heads),
        NormLayout::Block
    );
    assert_eq!(
        layout(NORM_ROWS_MAX_DIMENSION, heads - 1),
        NormLayout::Block
    );
    assert_eq!(layout(NORM_ROWS_MAX_DIMENSION, heads + NORM_ROWS), rows);
}

#[test]
fn a_prompt_norm_fuses_into_the_quantizer_its_consumer_declares() {
    let fp4 = Some(ActivationQuantization::Fp4(SCALE));
    assert_eq!(
        fusion(NORM, consumer(Format::Fp4, fp4)),
        Some(Fusion::Fp4(SCALE))
    );
    assert_eq!(
        fusion(NORM, consumer(Format::Fp8, None)),
        Some(Fusion::Maximum)
    );
    for (format, activation) in [
        (Format::Dense, None),
        (Format::Fp8Block, None),
        (Format::Fp4, None),
    ] {
        assert_eq!(
            fusion(NORM, consumer(format, activation)),
            None,
            "{format:?}"
        );
    }
}

#[test]
fn fusion_keeps_the_canonical_dispatch_outside_its_proven_envelope() {
    let fp4 = consumer(Format::Fp4, Some(ActivationQuantization::Fp4(SCALE)));
    let fp8 = consumer(Format::Fp8, None);
    for consumer in [fp4, fp8] {
        let canonical = Consumer {
            fuses: false,
            ..consumer
        };
        assert_eq!(fusion(NORM, canonical), None);
    }
    let unplanned = Consumer {
        planned_fp4: false,
        ..fp4
    };
    assert_eq!(fusion(NORM, unplanned), None);
    let unquantized = Consumer {
        fp8_input: false,
        ..fp8
    };
    assert_eq!(fusion(NORM, unquantized), None);
    let narrow = Norm {
        dimension: NORM_ROWS_MAX_DIMENSION,
        ..NORM
    };
    assert_eq!(fusion(narrow, fp4), None);
    for epsilon in [0.0, f32::NAN] {
        assert_eq!(fusion(Norm { epsilon, ..NORM }, fp4), None);
    }
    let unscaled = consumer(Format::Fp4, Some(ActivationQuantization::Fp4(0.0)));
    assert_eq!(fusion(NORM, unscaled), None);
}
