use super::*;
use crate::RequestId;
#[test]
fn stale_handle_cannot_alias_reused_slot() {
    let mut arena = Arena::<RequestId, _>::default();
    let old = arena.insert(12).unwrap();
    assert_eq!(arena.remove(old), Some(12));
    let new = arena.insert(24).unwrap();
    assert_ne!(old, new);
    assert_eq!(arena.get(old), None);
    assert_eq!(arena.get(new), Some(&24));
    assert_eq!(arena.get(RequestId::new(1).unwrap()), None);
}
#[test]
fn fixed_slots_never_move_and_exhausted_generations_retire() -> Result<()> {
    let mut arena = Arena::<RequestId, _>::with_capacity(2)?;
    let storage = arena.slots.as_ptr();
    let free = arena.free.as_ptr();
    let first = arena.insert(1)?;
    let second = arena.insert(2)?;
    assert!(arena.insert(3).is_err());
    assert_eq!(arena.remove(first), Some(1));
    arena.slots[0].generation = u32::MAX;
    let last = arena.insert(3)?;
    assert_eq!(arena.remove(last), Some(3));
    assert!(arena.insert(4).is_err());
    assert_eq!(arena.get(first), None);
    assert_eq!(arena.get(second), Some(&2));
    assert_eq!(arena.slots.as_ptr(), storage);
    assert_eq!(arena.free.as_ptr(), free);
    Ok(())
}
