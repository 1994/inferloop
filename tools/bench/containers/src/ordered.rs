use crate::{OPERATIONS, ROUNDS, Sample, measure};
use bbtree::unsync::BBTreeSet;
use infer_core::RequestId;
use std::collections::BTreeSet;
use std::hint::black_box;

// Exercise the production slab wrapper, including its preallocation and range workaround.
#[path = "../../../../crates/engine/scheduler/src/queue/ordered.rs"]
mod production;
use production::OrderedSet;
type Key = (u64, u64, RequestId, usize);
const fn key(slot: usize) -> Key {
    (slot as u64 / 16, slot as u64, RequestId::ONE, slot)
}
trait Index {
    fn insert(&mut self, key: Key);
    fn remove(&mut self, key: &Key);
    fn first(&self) -> Option<Key>;
    fn lower(&self, key: Key) -> Option<Key>;
}
macro_rules! index {
    ($ty:ty) => {
        impl Index for $ty {
            fn insert(&mut self, key: Key) {
                self.insert(key);
            }
            fn remove(&mut self, key: &Key) {
                self.remove(key);
            }
            fn first(&self) -> Option<Key> {
                self.first().copied()
            }
            fn lower(&self, key: Key) -> Option<Key> {
                self.range(key..).next().copied()
            }
        }
    };
}
index!(OrderedSet<Key>);
index!(BTreeSet<Key>);
index!(BBTreeSet<Key>);
fn sample(mut tree: impl Index, capacity: usize, round: usize, name: &str) -> Sample {
    let mut live: Vec<_> = (0..capacity).map(key).collect();
    // Production prewarms and clears nodes. Try the same on the candidate via the caller.
    for k in &live {
        tree.insert(*k);
    }
    measure("ordered_churn_first_range", name, capacity, round, || {
        let mut checksum = 0_u64;
        for i in 0..OPERATIONS {
            let slot = (i * 17) % capacity;
            tree.remove(black_box(&live[slot]));
            live[slot] = key(capacity + i);
            tree.insert(black_box(live[slot]));
            checksum = checksum.wrapping_add(tree.first().map_or(0, |k| k.1));
            checksum = checksum.wrapping_add(tree.lower(black_box(key(i))).map_or(0, |k| k.1));
        }
        checksum
    })
}
pub fn run(capacity: usize, samples: &mut Vec<Sample>) {
    // Oracle: the actual multi-field key ordering and missing-key lower bounds.
    let mut slab = OrderedSet::new(capacity, key);
    let mut bb = BBTreeSet::new();
    let mut std = BTreeSet::new();
    for slot in (0..capacity).rev() {
        slab.insert(key(slot * 2));
        bb.insert(key(slot * 2));
        std.insert(key(slot * 2));
    }
    assert_eq!(slab.len(), std.len());
    assert!(slab.iter().eq(std.iter()));
    assert!(bb.iter().eq(std.iter()));
    for slot in 0..capacity * 2 {
        assert!(bb.range(key(slot)..).eq(std.range(key(slot)..)));
        assert!(slab.range(key(slot)..).eq(std.range(key(slot)..)));
    }
    for round in 0..ROUNDS {
        let start = samples.len();
        for offset in 0..3 {
            samples.push(match (round + offset) % 3 {
                0 => sample(
                    OrderedSet::new(capacity, key),
                    capacity,
                    round,
                    "btree-slab",
                ),
                1 => sample(BTreeSet::new(), capacity, round, "std"),
                _ => {
                    let mut tree = BBTreeSet::new();
                    for slot in 0..capacity * 2 {
                        tree.insert(key(slot));
                    }
                    tree.clear();
                    sample(tree, capacity, round, "bbtree")
                }
            });
        }
        assert!(
            samples[start..]
                .iter()
                .all(|s| s.checksum == samples[start].checksum)
        );
    }
}
