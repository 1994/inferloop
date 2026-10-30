use super::AxisTable;
use infer_core::Result;

#[test]
fn bicubic_coefficients_match_downsample_reference() -> Result<()> {
    let table = AxisTable::new(4, 2)?;
    assert_eq!(table.taps, 4);
    assert_eq!(table.denominator, 65_536);
    assert_eq!(table.indices, [0, 1, 2, 3, 0, 1, 2, 3]);
    assert_eq!(
        table.weights,
        [30_060, 30_060, 7_853, -2_437, -2_437, 7_853, 30_060, 30_060]
    );
    Ok(())
}

#[test]
fn bicubic_coefficients_match_upsample_reference() -> Result<()> {
    let table = AxisTable::new(3, 5)?;
    assert_eq!(table.taps, 3);
    assert_eq!(table.denominator, 16_384);
    assert_eq!(table.indices, [0, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0]);
    assert_eq!(
        table.weights,
        [
            17_621, -1_237, 0, 10_637, 6_480, -734, 0, 16_384, 0, -734, 6_480, 10_637, -1_237,
            17_621, 0,
        ]
    );
    Ok(())
}

#[test]
fn filter_preserves_a_constant_and_keeps_indices_in_range() -> Result<()> {
    for (source, target) in [(1, 32), (32, 1), (32, 7), (333, 352), (819, 128)] {
        let table = AxisTable::new(source, target)?;
        assert_eq!(table.indices.len(), target * table.taps);
        assert_eq!(table.weights.len(), table.indices.len());
        for (indices, weights) in table
            .indices
            .chunks_exact(table.taps)
            .zip(table.weights.chunks_exact(table.taps))
        {
            assert!(
                indices
                    .iter()
                    .all(|&index| usize::try_from(index).is_ok_and(|value| value < source))
            );
            let sum: i64 = weights.iter().map(|&weight| i64::from(weight)).sum();
            let scaled = sum * 127 + i64::from(table.denominator / 2);
            assert_eq!(scaled / i64::from(table.denominator), 127);
        }
    }
    Ok(())
}

#[test]
fn invalid_and_unrepresentable_dimensions_are_rejected() {
    assert!(AxisTable::new(0, 1).is_err());
    assert!(AxisTable::new(1, 0).is_err());
    assert!(AxisTable::new(usize::MAX, 1).is_err());
    assert!(AxisTable::new(1, usize::MAX).is_err());
}
