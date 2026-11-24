use super::*;

#[test]
fn context_above_old_presets_follows_the_model() -> Result<()> {
    for capacity in [32_769, 65_537, 131_072, 262_144, 1_048_576] {
        assert_eq!(
            for_model(capacity, 1_048_576)?,
            capacity.next_power_of_two()
        );
    }
    // Internal storage padding can exceed a non-power-of-two model ceiling.
    assert_eq!(for_model(90_000, 100_000)?, 131_072);
    assert!(for_model(100_001, 100_000).is_err());
    Ok(())
}

#[test]
fn invalid_and_unrepresentable_capacities_fail_before_allocation() {
    for capacity in [
        0,
        usize::MAX,
        usize::try_from(i32::MAX).unwrap_or(usize::MAX),
    ] {
        assert!(padded(capacity).is_err());
    }
}
