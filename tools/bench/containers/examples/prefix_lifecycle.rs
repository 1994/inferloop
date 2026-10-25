//! CPU snapshot-policy comparison; backend device copies are intentionally outside this scope.
#[allow(
    unsafe_code,
    reason = "Reuse the audited standalone measurement allocator only"
)]
#[allow(
    dead_code,
    reason = "The shared CPU allocator also offers traps unused by this example"
)]
#[path = "../../cpu/src/allocator.rs"]
mod allocator;
#[path = "prefix_lifecycle/publication.rs"]
mod publication;
use infer_state::cache::{
    BlockHash, PrefixCache, PrefixPolicy, PrefixReuse, PrefixSnapshot, SnapshotScan,
};
use serde::Serialize;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::error::Error;
use std::hint::black_box;
use std::result::Result as StdResult;
use std::time::Instant;
#[global_allocator]
static ALLOCATOR: allocator::CountingAllocator = allocator::CountingAllocator;
const ITERATIONS: u32 = 4096;
const ROUNDS: usize = 7;
const WARMUP: usize = 64;
const BLOCK: usize = 4096;

trait Reuse {
    fn run(&mut self, tokens: &[u32], maximum: usize, history: &mut Vec<u32>) -> usize;
}
// Historical CUDA matching/take/reinsert policy from the base revision. Payloads here are u64
// placeholders, not CUDA tensors: only CPU search, metadata/LRU and history copying are measured.
struct Legacy {
    entries: VecDeque<PrefixSnapshot<u64>>,
    bytes: u64,
    capacity: u64,
    lookups: u64,
    hits: u64,
    reused_tokens: u64,
}
impl Legacy {
    fn new(tokens: &[Vec<u32>]) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            entries: tokens
                .iter()
                .map(|t| PrefixSnapshot {
                    tokens: t.clone(),
                    value: 0,
                    bytes: 1,
                })
                .collect(),
            bytes: u64::try_from(tokens.len())?,
            capacity: u64::try_from(tokens.len())?,
            lookups: 0,
            hits: 0,
            reused_tokens: 0,
        })
    }
    fn insert(&mut self, entry: PrefixSnapshot<u64>) {
        if self.capacity == 0 || entry.bytes > self.capacity {
            return;
        }
        if let Some(index) = self.entries.iter().position(|e| e.tokens == entry.tokens)
            && let Some(previous) = self.entries.remove(index)
        {
            self.bytes -= previous.bytes;
        }
        self.bytes += entry.bytes;
        self.entries.push_back(entry);
        while self.bytes > self.capacity {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= evicted.bytes;
        }
    }
}
impl Reuse for Legacy {
    fn run(&mut self, tokens: &[u32], maximum: usize, history: &mut Vec<u32>) -> usize {
        self.lookups = self.lookups.saturating_add(1);
        let wanted = self
            .entries
            .iter()
            .filter(|e| e.tokens.len() <= maximum && tokens.starts_with(&e.tokens))
            .map(|e| e.tokens.len())
            .max()
            .unwrap_or(0);
        if wanted == 0 {
            return 0;
        }
        let Some(index) = self
            .entries
            .iter()
            .position(|e| e.tokens.len() == wanted && tokens.starts_with(&e.tokens))
        else {
            return 0;
        };
        let Some(entry) = self.entries.remove(index) else {
            return 0;
        };
        self.bytes -= entry.bytes;
        self.hits = self.hits.saturating_add(1);
        history.clear();
        history.extend_from_slice(&entry.tokens);
        self.insert(entry);
        self.reused_tokens = self.reused_tokens.saturating_add(wanted as u64);
        wanted
    }
}
struct CopyHistory<'a>(&'a mut Vec<u32>);
impl PrefixReuse<u64> for CopyHistory<'_> {
    type Output = usize;
    type Error = Infallible;
    fn compatible(&self, _: &[u32], _: &u64) -> bool {
        true
    }
    fn reuse(&mut self, tokens: &[u32], _: &u64) -> StdResult<usize, Infallible> {
        self.0.clear();
        self.0.extend_from_slice(tokens);
        Ok(tokens.len())
    }
}
impl<M: PrefixPolicy> Reuse for PrefixCache<u64, M> {
    fn run(&mut self, tokens: &[u32], maximum: usize, history: &mut Vec<u32>) -> usize {
        match self.reuse(tokens, maximum, &mut CopyHistory(history)) {
            Ok(value) => value.unwrap_or(0),
            Err(never) => match never {},
        }
    }
}
fn shared<M: PrefixPolicy>(
    tokens: &[Vec<u32>],
    matcher: M,
) -> Result<PrefixCache<u64, M>, Box<dyn Error>> {
    let mut cache = PrefixCache::with_policy(
        b"snapshot-policy",
        BLOCK,
        u64::try_from(tokens.len())?,
        tokens.len(),
        matcher,
    )?;
    for tokens in tokens {
        cache.insert(tokens.clone(), 0, 1)?;
    }
    Ok(cache)
}
#[derive(Serialize)]
struct Sample {
    length: usize,
    entries: usize,
    case: &'static str,
    implementation: &'static str,
    round: usize,
    iterations: u32,
    ns_per_operation: f64,
    checksum: usize,
    allocator: allocator::Counts,
}
struct Workload<'a> {
    queries: &'a [Vec<u32>],
    maximum: usize,
    length: usize,
    entries: usize,
    case: &'static str,
}
fn measure(
    mut cache: impl Reuse,
    w: &Workload<'_>,
    name: &'static str,
    round: usize,
) -> Result<Sample, Box<dyn Error>> {
    let mut history = Vec::with_capacity(w.length);
    for i in 0..WARMUP {
        black_box(cache.run(&w.queries[i % w.queries.len()], w.maximum, &mut history));
    }
    let iterations = usize::try_from(ITERATIONS)?;
    let mut checksum = 0;
    allocator::start();
    let start = Instant::now();
    for i in 0..iterations {
        checksum += cache.run(
            black_box(&w.queries[i % w.queries.len()]),
            w.maximum,
            &mut history,
        );
    }
    let elapsed = start.elapsed();
    let counts = allocator::stop();
    if name == "shared-scan" {
        assert_eq!(
            (
                counts.allocations,
                counts.reallocations,
                counts.deallocations
            ),
            (0, 0, 0)
        );
    }
    black_box(&history);
    black_box(&cache);
    Ok(Sample {
        length: w.length,
        entries: w.entries,
        case: w.case,
        implementation: name,
        round,
        iterations: ITERATIONS,
        ns_per_operation: elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS),
        checksum,
        allocator: counts,
    })
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut samples = Vec::new();
    for length in [4096, 32768] {
        for entries in [1, 8, 32] {
            let tokens: Vec<_> = (0..entries)
                .map(|i| {
                    let mut tokens: Vec<u32> = (0..length)
                        .map(|j| u32::try_from(j % 251).unwrap_or(0))
                        .collect();
                    tokens[0] = u32::try_from(i).unwrap_or(0);
                    tokens
                })
                .collect();
            let miss = vec![vec![u32::MAX; length]];
            for case in ["hit", "early-miss", "capped"] {
                let workload = Workload {
                    queries: if case == "early-miss" { &miss } else { &tokens },
                    maximum: if case == "capped" { length - 1 } else { length },
                    length,
                    entries,
                    case,
                };
                for round in 0..ROUNDS {
                    let begin = samples.len();
                    for offset in 0..3 {
                        samples.push(match (round + offset) % 3 {
                            0 => measure(
                                Legacy::new(&tokens)?,
                                &workload,
                                "legacy-cuda-policy",
                                round,
                            )?,
                            1 => measure(
                                shared(&tokens, SnapshotScan)?,
                                &workload,
                                "shared-scan",
                                round,
                            )?,
                            _ => measure(
                                shared(&tokens, BlockHash)?,
                                &workload,
                                "shared-block-hash",
                                round,
                            )?,
                        });
                    }
                    assert!(
                        samples[begin..]
                            .iter()
                            .all(|s| s.checksum == samples[begin].checksum)
                    );
                }
            }
        }
    }
    publication::run(&mut samples)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope":"CPU metadata policy and host history copy; synthetic u64 payload, no GPU copies or model/service latency",
            "iterations":ITERATIONS, "publication_iterations":publication::ITERATIONS, "rounds":ROUNDS, "warmup":WARMUP, "samples":samples
        }))?
    );
    Ok(())
}
