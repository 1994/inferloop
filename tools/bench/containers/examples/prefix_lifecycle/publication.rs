//! Cold publication cost with token buffers prepared before timing; no device snapshots/copies.
use super::{
    BLOCK, Legacy, PrefixCache, PrefixPolicy, PrefixSnapshot, Sample, WARMUP, allocator, shared,
};
use std::collections::VecDeque;
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;
pub const ITERATIONS: u32 = 256;
trait Publish {
    fn put(&mut self, snapshot: PrefixSnapshot<u64>) -> Result<(), Box<dyn Error>>;
}
impl Publish for Legacy {
    fn put(&mut self, snapshot: PrefixSnapshot<u64>) -> Result<(), Box<dyn Error>> {
        self.insert(snapshot);
        Ok(())
    }
}
impl<M: PrefixPolicy> Publish for PrefixCache<u64, M> {
    fn put(&mut self, snapshot: PrefixSnapshot<u64>) -> Result<(), Box<dyn Error>> {
        // Keep the retirement list observable, as it is for non-trivial device payload cleanup.
        self.publish(snapshot)
            .map(|retired| drop(black_box(retired)))
            .map_err(Into::into)
    }
}
fn measure(
    mut cache: impl Publish,
    tokens: &[Vec<u32>],
    case: &'static str,
    name: &'static str,
    round: usize,
) -> Result<Sample, Box<dyn Error>> {
    let length = tokens[0].len();
    for i in 0..WARMUP {
        cache.put(PrefixSnapshot {
            tokens: tokens[i % tokens.len()].clone(),
            value: 0,
            bytes: 1,
        })?;
    }
    let mut inputs = VecDeque::new();
    for i in 0..usize::try_from(ITERATIONS)? {
        let mut input = tokens[i % tokens.len()].clone();
        if case == "publish-evict" {
            input[0] = u32::try_from(tokens.len() + i)?;
        }
        inputs.push_back(PrefixSnapshot {
            tokens: input,
            value: 0,
            bytes: 1,
        });
    }
    allocator::start();
    let start = Instant::now();
    let mut checksum = 0;
    while let Some(input) = inputs.pop_front() {
        cache.put(black_box(input))?;
        checksum += 1;
    }
    let elapsed = start.elapsed();
    let counts = allocator::stop();
    black_box(&cache);
    Ok(Sample {
        length,
        entries: tokens.len(),
        case,
        implementation: name,
        round,
        iterations: ITERATIONS,
        ns_per_operation: elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS),
        checksum,
        allocator: counts,
    })
}
fn sample(
    tokens: &[Vec<u32>],
    case: &'static str,
    kind: usize,
    round: usize,
) -> Result<Sample, Box<dyn Error>> {
    use infer_state::cache::{BlockHash, SnapshotScan};
    match kind {
        0 => measure(
            Legacy::new(tokens)?,
            tokens,
            case,
            "legacy-cuda-policy",
            round,
        ),
        1 => measure(
            shared(tokens, SnapshotScan)?,
            tokens,
            case,
            "shared-scan",
            round,
        ),
        _ => measure(
            shared(tokens, BlockHash)?,
            tokens,
            case,
            "shared-block-hash",
            round,
        ),
    }
}
pub fn run(samples: &mut Vec<Sample>) -> Result<(), Box<dyn Error>> {
    use super::ROUNDS;
    for length in [BLOCK, BLOCK * 8] {
        for entries in [1, 8, 32] {
            let tokens: Vec<_> = (0..entries)
                .map(|i| {
                    let mut t = vec![0_u32; length];
                    t[0] = u32::try_from(i).unwrap_or(0);
                    t
                })
                .collect();
            for case in ["publish-replace", "publish-evict"] {
                for round in 0..ROUNDS {
                    let first = samples.len();
                    for offset in 0..3 {
                        samples.push(sample(&tokens, case, (round + offset) % 3, round)?);
                    }
                    assert!(
                        samples[first..]
                            .iter()
                            .all(|s| s.checksum == samples[first].checksum)
                    );
                }
            }
        }
    }
    Ok(())
}
