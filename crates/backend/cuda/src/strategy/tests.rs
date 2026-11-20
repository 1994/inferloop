use super::*;

#[test]
fn default_tiling_is_a_supported_launch_geometry() -> Result<()> {
    let tiling = default_tiling()?;
    assert!(SUPPORTED_TILE_ROWS.contains(&tiling.rows()));
    assert!((MIN_TILE_COLUMNS..=MAX_TILE_COLUMNS).contains(&tiling.columns()));
    assert!(tiling.columns().is_power_of_two());
    Ok(())
}
