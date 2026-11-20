use super::OrderedSet;
use std::{
    collections::BTreeSet,
    ops::Bound::{Excluded, Included, Unbounded},
};

#[test]
fn pooled_library_tree_matches_standard_set_through_random_churn_and_clone() {
    let mut random = 42u64;
    for capacity in [1, 7, 8, 31, 256, 1024] {
        let mut tree = OrderedSet::new(capacity, |slot| slot as u64);
        let mut reference = BTreeSet::new();
        for _ in 0..10_000 {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let key = random % 1024;
            if random & 1024 == 0 {
                let expected = reference.len() < capacity && reference.insert(key);
                assert_eq!(tree.insert(key), expected);
            } else {
                assert_eq!(tree.remove(&key), reference.remove(&key));
            }
            assert_eq!(tree.first(), reference.first());
            assert!(tree.iter().eq(reference.iter()));
            assert_eq!(
                tree.range(key..).copied().collect::<Vec<_>>(),
                reference.range(key..).copied().collect::<Vec<_>>(),
                "capacity={capacity} key={key}"
            );
        }
        for lower in 0..1025 {
            for bounds in [
                (Included(lower), Unbounded),
                (Excluded(lower), Unbounded),
                (Unbounded, Included(lower)),
                (Unbounded, Excluded(lower)),
                (Included(lower), Included(lower + 10)),
                (Excluded(lower), Excluded(lower + 10)),
            ] {
                assert!(
                    tree.range(bounds).eq(reference.range(bounds)),
                    "capacity={capacity} bounds={bounds:?}"
                );
            }
        }
        assert_eq!(tree.clone(), tree);
    }
}
