use super::*;

#[test]
fn utilization_bounds_the_engine_share() -> Result<()> {
    let total = 32 * 1024 * 1024 * 1024_u64;
    assert_eq!(utilization_bytes(total, 0.9)?, 30_923_764_531);
    assert_eq!(utilization_bytes(total, 1.0)?, total);
    // Zero disables the cap instead of forbidding every byte.
    assert_eq!(utilization_bytes(total, 0.0)?, total);
    assert!(utilization_bytes(total, 1.5).is_err());
    Ok(())
}
