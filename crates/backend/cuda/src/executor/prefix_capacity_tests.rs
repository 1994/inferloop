use super::prefix_cache_bytes;

const MIB: u64 = 1024 * 1024;

#[test]
fn capacity_is_a_quarter_of_the_budget_above_the_floor() {
    assert_eq!(prefix_cache_bytes(4 * 1024 * MIB), 1024 * MIB);
    assert_eq!(prefix_cache_bytes(8 * 1024 * MIB), 2 * 1024 * MIB);
}

#[test]
fn capacity_is_disabled_when_a_quarter_would_be_useless() {
    // Below the floor a prefix entry could not be stored, so the cache stays off.
    assert_eq!(prefix_cache_bytes(0), 0);
    assert_eq!(prefix_cache_bytes(511 * MIB), 0);
    // Exactly at the floor it turns on.
    assert_eq!(prefix_cache_bytes(512 * MIB), 128 * MIB);
}
