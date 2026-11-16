use super::{GroupedIntTiled, code_shape};
use infer_cuda_kernels::contract::{CodePacking, GROUPED_INT_TILE_ROWS};
use infer_ir::{GROUPED_INT_SPAN, GroupedIntFormat};
use infer_models::span_bytes;

const ROWS: usize = 32;
const COLUMNS: usize = 2 * GROUPED_INT_SPAN;

/// Deterministic bytes whose F16 fields stay finite: the top exponent bits are never all set.
fn payload(format: GroupedIntFormat) -> Vec<u8> {
    let len = ROWS * (COLUMNS / GROUPED_INT_SPAN) * span_bytes(format).unwrap();
    let mut state = 0x9e37_79b9_u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8 & 0xbf
        })
        .collect()
}

/// Read value `(row, column)` back out of the tile-major layout.
fn packed_value(tiled: &GroupedIntTiled, row: usize, column: usize) -> f32 {
    let format = tiled.format();
    let spans = COLUMNS / GROUPED_INT_SPAN;
    let tile_span = (row / GROUPED_INT_TILE_ROWS) * spans + column / GROUPED_INT_SPAN;
    let lane_row = row % GROUPED_INT_TILE_ROWS;
    let within = column % GROUPED_INT_SPAN;
    let code = match CodePacking::of(format) {
        CodePacking::Byte => tiled.codes()
            [(tile_span * GROUPED_INT_SPAN + within) * GROUPED_INT_TILE_ROWS + lane_row],
        CodePacking::Nibble => {
            let (group, offset) = (within / 64, within % 64);
            let byte = tiled.codes()
                [((tile_span * 4 + group) * 32 + offset % 32) * GROUPED_INT_TILE_ROWS + lane_row];
            if offset < 32 { byte & 0xf } else { byte >> 4 }
        }
    };
    let term = (tile_span * format.span_groups() + within / format.group_values())
        * GROUPED_INT_TILE_ROWS
        + lane_row;
    let bias = tiled.biases().map_or(0.0, |biases| biases[term]);
    format.value(code, tiled.factors()[term], bias)
}

#[test]
fn every_format_round_trips_through_the_tiled_layout() {
    for format in GroupedIntFormat::ALL {
        let blocks = payload(format);
        let mut expected = vec![0.0f32; ROWS * COLUMNS];
        let ty = infer_format::stored_type(format);
        infer_format::dequantize(ty, &blocks, &mut expected).unwrap();
        let tiled = GroupedIntTiled::from_blocks(format, ROWS, COLUMNS, &blocks).unwrap();
        assert_eq!(tiled.code_shape(), code_shape(format, ROWS, COLUMNS));
        assert_eq!(tiled.biases().is_some(), format.has_biases());
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let actual = packed_value(&tiled, row, column);
                assert_eq!(
                    actual.to_bits(),
                    expected[row * COLUMNS + column].to_bits(),
                    "{} at ({row}, {column})",
                    format.label()
                );
            }
        }
    }
}

#[test]
fn rejects_partial_tiles_and_short_payloads() {
    let format = GroupedIntFormat::Q8_0;
    let blocks = payload(format);
    assert!(GroupedIntTiled::from_blocks(format, ROWS - 1, COLUMNS, &blocks).is_err());
    assert!(GroupedIntTiled::from_blocks(format, ROWS, COLUMNS, &blocks[1..]).is_err());
}
