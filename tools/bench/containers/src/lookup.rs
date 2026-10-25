use crate::{OPERATIONS, ROUNDS, Sample, measure};
use ahash::{AHashMap, RandomState};
use bbtree::unsync::BBTreeMap;
use hashbrown::HashMap;
use infer_core::map::BoundedMap;
use std::collections::BTreeMap;
use std::error::Error;
use std::hint::black_box;
type Key = [u8; 32];
fn key(slot: usize) -> Key {
    let mut out = [0; 32];
    let mut random = slot as u64;
    for word in out.as_chunks_mut::<8>().0 {
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        word.copy_from_slice(&random.to_le_bytes());
    }
    out
}
trait Lookup {
    fn put(&mut self, key: Key, value: u64);
    fn remove(&mut self, key: &Key) -> Option<u64>;
    fn get(&self, key: &Key) -> Option<&u64>;
}
macro_rules! lookup {
    ($ty:ty) => {
        impl Lookup for $ty {
            fn put(&mut self, key: Key, value: u64) {
                self.insert(key, value);
            }
            fn remove(&mut self, key: &Key) -> Option<u64> {
                self.remove(key)
            }
            fn get(&self, key: &Key) -> Option<&u64> {
                self.get(key)
            }
        }
    };
}
lookup!(BTreeMap<Key, u64>);
lookup!(BBTreeMap<Key, u64>);
lookup!(AHashMap<Key, u64>);
lookup!(HashMap<Key, u64, RandomState>);
impl Lookup for BoundedMap<Key, u64> {
    fn put(&mut self, key: Key, value: u64) {
        assert!(self.insert(key, value).is_ok());
    }
    fn remove(&mut self, key: &Key) -> Option<u64> {
        self.remove(key)
    }
    fn get(&self, key: &Key) -> Option<&u64> {
        self.get(key)
    }
}
fn sample(mut map: impl Lookup, capacity: usize, round: usize, name: &str) -> Sample {
    let mut live: Vec<_> = (0..capacity).map(key).collect();
    let incoming: Vec<_> = (capacity..capacity + OPERATIONS).map(key).collect();
    for (i, k) in live.iter().enumerate() {
        map.put(*k, i as u64);
    }
    measure("digest_lookup_churn", name, capacity, round, || {
        let mut checksum = 0_u64;
        for (i, k) in incoming.iter().enumerate() {
            let slot = (i * 17) % capacity;
            let hit = black_box(&live[slot]);
            checksum = checksum.wrapping_add(*map.get(hit).unwrap_or(&0));
            assert!(map.get(black_box(k)).is_none());
            checksum = checksum.wrapping_add(map.remove(hit).unwrap_or(0));
            live[slot] = *k;
            map.put(black_box(*k), i as u64);
        }
        checksum
    })
}
pub fn run(capacity: usize, samples: &mut Vec<Sample>) -> Result<(), Box<dyn Error>> {
    for round in 0..ROUNDS {
        let start = samples.len();
        for offset in 0..5 {
            samples.push(match (round + offset) % 5 {
                0 => sample(BTreeMap::new(), capacity, round, "std-btree"),
                1 => sample(BBTreeMap::new(), capacity, round, "bbtree"),
                2 => sample(
                    AHashMap::with_capacity(capacity),
                    capacity,
                    round,
                    "std-ahash",
                ),
                3 => sample(BoundedMap::new(capacity)?, capacity, round, "bounded"),
                _ => sample(
                    HashMap::with_capacity_and_hasher(capacity, RandomState::new()),
                    capacity,
                    round,
                    "hashbrown-ahash",
                ),
            });
        }
        assert!(
            samples[start..]
                .iter()
                .all(|s| s.checksum == samples[start].checksum)
        );
    }
    Ok(())
}
