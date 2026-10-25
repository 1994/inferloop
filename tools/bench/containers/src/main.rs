//! Single-owner container comparisons. These are CPU microbenchmarks, not inference throughput.
#[allow(
    unsafe_code,
    reason = "Shared audited measurement allocator; never linked into production"
)]
#[allow(
    dead_code,
    reason = "Shared CPU counter also offers an optional trap unused here"
)]
#[path = "../../cpu/src/allocator.rs"]
mod allocator;
mod lookup;
mod ordered;
mod prefix;
use serde::Serialize;
use std::env::consts::{ARCH, OS};
use std::env::var_os;
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: allocator::CountingAllocator = allocator::CountingAllocator;
const ROUNDS: usize = 7;
const OPERATIONS: usize = 100_000;

#[derive(Serialize)]
struct Sample {
    case: String,
    implementation: String,
    capacity: usize,
    round: usize,
    ns_per_operation: f64,
    allocator: allocator::Counts,
    checksum: u64,
}
fn measure(
    case: &str,
    implementation: &str,
    capacity: usize,
    round: usize,
    mut run: impl FnMut() -> u64,
) -> Sample {
    allocator::start();
    let start = Instant::now();
    let checksum = black_box(run());
    let elapsed = start.elapsed();
    let counts = allocator::stop();
    Sample {
        case: case.into(),
        implementation: implementation.into(),
        capacity,
        round,
        ns_per_operation: elapsed.as_secs_f64() * 1e9 / 100_000.0,
        allocator: counts,
        checksum,
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut samples = Vec::new();
    for capacity in [16, 64, 1024, 8192] {
        if var_os("INFER_CONTAINER_PREFIX_ONLY").is_none() {
            ordered::run(capacity, &mut samples);
            lookup::run(capacity, &mut samples)?;
        }
        prefix::run(capacity, &mut samples)?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope": "Single-thread CPU container operations; construction outside timing; seven rotated rounds; no GPU/model throughput claim",
            "operations": OPERATIONS, "rounds": ROUNDS,
            "arch": ARCH, "os": OS,
            "samples": samples
        }))?
    );
    Ok(())
}
