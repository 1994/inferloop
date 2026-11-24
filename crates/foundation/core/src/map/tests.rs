use super::*;
use std::collections::HashMap;
#[test]
fn churn_preserves_storage_and_matches_reference() -> Result<()> {
    let mut map = BoundedMap::new(128)?;
    let pointer = map.buckets.as_ptr();
    let mut reference = HashMap::new();
    let mut seed = 17_u64;
    for step in 0..100_000 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let key = (seed >> 32) % 128;
        if seed & 1 == 0 {
            assert_eq!(map.insert(key, step)?, reference.insert(key, step));
        } else {
            assert_eq!(map.remove(&key), reference.remove(&key));
        }
        assert_eq!(map.len(), reference.len());
        assert_eq!(map.get(&key), reference.get(&key));
        assert_eq!(map.buckets.as_ptr(), pointer);
    }
    for (key, value) in reference {
        assert_eq!(map.get(&key), Some(&value));
    }
    Ok(())
}
#[test]
fn full_map_and_snapshot_do_not_alias_or_accept_duplicate_keys() -> Result<()> {
    let mut map = BoundedMap::new(1)?;
    assert_eq!(map.insert(1, 2)?, None);
    assert!(map.insert(2, 3).is_err());
    assert_eq!(map.insert(1, 4)?, Some(2));
    assert_eq!(map.remove(&1), Some(4));
    assert_eq!(map.insert(2, 3)?, None);
    Ok(())
}
