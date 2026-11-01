//! Shared attention benchmark timing; amortize host graph submission over device repeats.
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::Result;
use infer_cuda_kernels::tile::runtime::Graph;

pub const REPEATS: usize = 8;

pub fn measure(
    device: &CudaDevice,
    graph: &Graph,
    tokens: i32,
    heads: usize,
    head: usize,
    tiles: [usize; 3],
) -> Result<()> {
    for _ in 0..100 {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
    }
    let mut samples = Vec::new();
    for _ in 0..60 {
        let start = device.stream.device().new_event().map_err(device_error)?;
        let end = device.stream.device().new_event().map_err(device_error)?;
        start.record(&device.stream).map_err(device_error)?;
        graph
            .launch()
            .map(|()| end.record(&device.stream))
            .sync_on(&device.stream)
            .map_err(device_error)?
            .map_err(device_error)?;
        samples.push(start.elapsed_time(&end).map_err(device_error)? / 8.0);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f32::total_cmp);
    let median = f32::midpoint(sorted[29], sorted[30]);
    println!(
        "ATTENTION_BENCH {}",
        serde_json::json!({
            "tokens": tokens, "heads": heads, "head_dim": head, "tiles": tiles,
            "median_ms": median, "samples_ms": samples,
            "replays_per_sample": REPEATS,
            "scope": "warm graph; per invocation averaged over repeated device operations",
        })
    );
    Ok(())
}
