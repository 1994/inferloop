use super::*;

#[test]
fn nibble_order_signs_and_block_scales() -> Result<()> {
    let packed: Vec<_> = (0u8..8).map(|x| (2 * x) | ((2 * x + 1) << 4)).collect();
    let packed: Vec<_> = packed.repeat(4);
    let got = to_bf16(&packed, &[0x38, 0x40, 0x30, 0x28], 2.0, 2, 32, 128)?;
    let expected: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for (block, factor) in [0.5, 1.0, 0.25, 0.125].into_iter().enumerate() {
        for (i, value) in expected.iter().enumerate() {
            let decoded = f32::from_bits(u32::from(got[block * 16 + i]) << 16);
            assert_eq!(decoded.to_bits(), (value * factor).to_bits());
        }
    }
    Ok(())
}

#[test]
fn rounds_expanded_weights_to_bf16() -> Result<()> {
    let got = to_bf16(&[0xa2; 8], &[0x38], 3.0, 1, 16, 32)?;
    assert_eq!(got, [0x3eab, 0xbeab].repeat(8));
    assert_eq!(to_bf16(&[0x22; 8], &[0], 1.0, 1, 16, 32)?, vec![0; 16]);
    Ok(())
}

#[test]
fn rejects_invalid_or_unbounded_conversion() {
    for scale in [127, 128, 255] {
        assert!(to_bf16(&[0x22; 8], &[scale], 1.0, 1, 16, 32).is_err());
    }
    for global in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(to_bf16(&[0x22; 8], &[0x38], global, 1, 16, 32).is_err());
    }
    assert!(to_bf16(&[0x22; 8], &[0x38], 1.0, 1, 16, 31).is_err());
    assert!(to_bf16(&[], &[], 1.0, usize::MAX, 16, usize::MAX).is_err());
    assert!(to_bf16(&[0x22; 8], &[0x38], 1.0, 1, 15, 32).is_err());
    assert!(to_bf16(&[0x22; 7], &[0x38], 1.0, 1, 16, 32).is_err());
    assert!(to_bf16(&[0x77; 8], &[126], f32::MIN_POSITIVE, 1, 16, 32).is_err());
}
