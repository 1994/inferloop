use crate::{OPERATIONS, ROUNDS, Sample, measure};
use infer_state::cache::PrefixCache;
use std::error::Error;
use std::hint::black_box;
pub fn run(capacity: usize, samples: &mut Vec<Sample>) -> Result<(), Box<dyn Error>> {
    for (block, length) in [(2, 2), (16, 128)] {
        let tokens: Vec<_> = (0..u32::try_from(capacity)?)
            .map(|i| vec![i; length])
            .collect();
        let lookup = format!("prefix_lookup_touch_b{block}_t{length}");
        let contains = format!("prefix_contains_b{block}_t{length}");
        for round in 0..ROUNDS {
            let mut cache = PrefixCache::new(b"container-benchmark", block, u64::MAX, capacity)?;
            for (i, t) in tokens.iter().enumerate() {
                cache.insert(t.clone(), i as u64, 8)?;
            }
            // Rotate read and touching lookup order as well as the container comparisons.
            for offset in 0..2 {
                let touch = (round + offset) % 2 == 0;
                samples.push(measure(
                    if touch { &lookup } else { &contains },
                    "production",
                    capacity,
                    round,
                    || {
                        let mut checksum = 0_u64;
                        for i in 0..OPERATIONS {
                            let tokens = black_box(&tokens[(i * 17) % capacity]);
                            checksum = checksum.wrapping_add(if touch {
                                cache.lookup(tokens, length).unwrap_or(0)
                            } else {
                                u64::from(cache.contains(tokens))
                            });
                        }
                        checksum
                    },
                ));
            }
        }
    }
    Ok(())
}
