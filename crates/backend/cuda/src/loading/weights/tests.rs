use super::e4m3_code;
use std::cmp::Ordering;

/// Independent reference: the nearest finite E4M3 magnitude by search over the code table, ties
/// to the even code, saturating at 448.
fn reference(value: f32) -> u8 {
    let magnitudes: Vec<f32> = (0u8..127)
        .map(|code| {
            let exponent = i32::from(code >> 3);
            let mantissa = f32::from(code & 7);
            if exponent == 0 {
                mantissa / 8.0 * 2f32.powi(-6)
            } else {
                (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
            }
        })
        .collect();
    let magnitude = value.abs().min(448.0);
    let upper = magnitudes.partition_point(|&m| m < magnitude);
    let code = if upper == 0 {
        0
    } else if upper >= magnitudes.len() {
        magnitudes.len() - 1
    } else {
        let below = magnitude - magnitudes[upper - 1];
        let above = magnitudes[upper] - magnitude;
        match below.total_cmp(&above) {
            Ordering::Less => upper - 1,
            Ordering::Equal if (upper - 1) % 2 == 0 => upper - 1,
            Ordering::Greater | Ordering::Equal => upper,
        }
    };
    let sign = if value.is_sign_negative() && code != 0 {
        0x80
    } else {
        0
    };
    u8::try_from(code).expect("finite code") | sign
}

fn check(value: f32) {
    assert_eq!(
        e4m3_code(value),
        reference(value),
        "value {value:e} bits {:#010x}",
        value.to_bits()
    );
}

#[test]
fn every_code_its_midpoints_and_their_neighbours_match_the_reference() {
    let mut magnitudes = Vec::new();
    for code in 0u8..127 {
        let exponent = i32::from(code >> 3);
        let mantissa = f32::from(code & 7);
        magnitudes.push(if exponent == 0 {
            mantissa / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
        });
    }
    let mut points = magnitudes.clone();
    points.extend(
        magnitudes
            .windows(2)
            .map(|pair| f32::midpoint(pair[0], pair[1])),
    );
    points.push(448.0 + 16.0);
    for point in points {
        for value in [
            point,
            f32::from_bits(point.to_bits().saturating_sub(1)),
            f32::from_bits(point.to_bits() + 1),
        ] {
            check(value);
            check(-value);
        }
    }
}

#[test]
fn special_values_match_the_reference() {
    for value in [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        448.0,
        449.0,
        1.0e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
    ] {
        check(value);
    }
}

#[test]
fn a_strided_sweep_of_every_float_matches_the_reference() {
    const STRIDE: usize = 4_099;
    for bits in (0..=u32::MAX).step_by(STRIDE) {
        check(f32::from_bits(bits));
    }
}
