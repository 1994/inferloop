use super::super::geometry::Geometry;
use super::TableCache;
use infer_core::Result;
use std::sync::Arc;

#[test]
fn filter_tables_reuse_axes_and_preserve_in_flight_entries() -> Result<()> {
    let cache = TableCache::default();
    let single = Geometry::new(1, [256, 256], [256, 256])?;
    let batch = Geometry::new(16, [256, 256], [256, 256])?;
    let first = cache.get(&single)?;
    let reused = cache.get(&batch)?;
    assert!(Arc::ptr_eq(&first, &reused));

    let different = Geometry::new(1, [333, 571], [320, 576])?;
    let other = cache.get(&different)?;
    assert!(!Arc::ptr_eq(&first, &other));
    let rebuilt = cache.get(&single)?;
    assert!(!Arc::ptr_eq(&first, &rebuilt));
    assert_eq!(first.values, rebuilt.values);
    assert_eq!(first.offsets, rebuilt.offsets);
    Ok(())
}
