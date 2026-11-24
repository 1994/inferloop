use super::*;
#[test]
fn readers_and_unit_credits_are_independent_and_return_exactly_once() -> Result<()> {
    let pool = CreditPool::new(2, 10)?;
    let owner = pool.reserve(7)?;
    let reader = owner.clone();
    assert!(pool.reserve(4).is_err());
    let other = pool.reserve(3)?;
    drop(owner);
    assert_eq!(pool.used(), 10);
    drop(reader);
    assert_eq!(pool.used(), 3);
    let next = pool.reserve(7)?;
    drop(other);
    drop(next);
    assert_eq!(pool.used(), 0);
    Ok(())
}
#[test]
fn live_readers_block_slot_reuse_even_when_units_remain() -> Result<()> {
    let pool = CreditPool::new(1, 100)?;
    let first = pool.reserve(1)?;
    let generation = first.generation;
    let reader = first.clone();
    drop(first);
    assert!(pool.reserve(1).is_err());
    assert_eq!(pool.used(), 1);
    drop(reader);
    let next = pool.reserve(1)?;
    assert!(next.generation > generation);
    Ok(())
}
