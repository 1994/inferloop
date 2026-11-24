use super::*;
#[test]
fn full_membership_rejects_new_keys_and_reuses_removed_slots() -> Result<()> {
    let mut set = BoundedSet::new(2)?;
    assert!(set.insert(2)?);
    assert!(set.insert(1)?);
    assert!(!set.insert(1)?);
    assert!(set.insert(3).is_err());
    assert_eq!(set.pop_first(), Some(1));
    assert!(set.insert(3)?);
    assert!(set.remove(&2));
    assert!(!set.contains(&2));
    assert_eq!(set.pop_first(), Some(3));
    assert!(set.is_empty());
    Ok(())
}
