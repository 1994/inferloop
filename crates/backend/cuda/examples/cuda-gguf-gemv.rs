//! Native GGUF tensor GEMV: read real blocks, pack, run the projection kernel, compare.
use infer_backend_cuda::device::CudaDevice;
use infer_cuda_kernels::contract::{GroupedIntPlan, WeightRef};
use infer_format::GgufType;
use infer_models::{GgufPackage, Q4KTiled};
use std::env::args;
use std::error::Error;
use std::time::Instant;

const REPLAYS: usize = 50;
const HBM_GBPS: f64 = 1792.0;

#[expect(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "benchmark fixture and bandwidth reporting"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: cuda-gguf-gemv <model.gguf> <hf-slot>")?;
    let slot_name = arguments
        .next()
        .ok_or("usage: cuda-gguf-gemv <model.gguf> <hf-slot>")?;
    let package = GgufPackage::open(path)?;
    let slot = package
        .slots()
        .get(&slot_name)
        .ok_or_else(|| format!("missing slot {slot_name}"))?
        .clone();
    let [rows, columns] = slot.shape.as_slice() else {
        return Err("GEMV slot must be rank 2".into());
    };
    if !rows.is_multiple_of(16) {
        return Err(format!("rows {rows} is not a multiple of 16").into());
    }
    let x: Vec<f32> = (0..*columns)
        .map(|index| ((index as f32) * 0.017).sin())
        .collect();
    let values = package.dense(&slot_name, u64::MAX)?;
    let expected: Vec<f32> = (0..*rows)
        .map(|row| {
            values[row * *columns..][..*columns]
                .iter()
                .zip(&x)
                .map(|(value, activation)| value * activation)
                .sum()
        })
        .collect();

    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let x_alloc = runtime.upload(x)?;
    let out_alloc = runtime.zeros::<f32>(*rows)?;
    let plan = GroupedIntPlan {
        tile_rows: 16,
        columns: *columns,
    };
    let kernels = device.kernels();
    let (graph, bytes) = match slot.dtype {
        GgufType::Q4K => {
            let device_layout = package.q4k(&slot_name, u64::MAX)?;
            let layout = Q4KTiled::from_device(&device_layout)?;
            if layout.rows() != *rows || layout.columns() != *columns {
                return Err("Q4_K layout shape".into());
            }
            let codes = runtime.upload(layout.codes().to_vec())?;
            let factors = runtime.upload(layout.factors().to_vec())?;
            let biases = runtime.upload(layout.biases().to_vec())?;
            let bytes = codes.len() + (factors.len() + biases.len()) * size_of::<f32>();
            let graph = device.tile().capture(|scope| {
                kernels.projection.grouped_int_row(
                    scope,
                    &plan,
                    out_alloc.buffer(),
                    x_alloc.buffer(),
                    WeightRef::GroupedInt {
                        codes: codes.buffer(),
                        factors: factors.buffer(),
                        biases: biases.buffer(),
                    },
                )
            })?;
            (graph, bytes)
        }
        GgufType::Q6K => {
            let layout = package.q6k(&slot_name, u64::MAX)?;
            if layout.rows() != *rows || layout.columns() != *columns {
                return Err("Q6_K layout shape".into());
            }
            let codes = runtime.upload(layout.codes().to_vec())?;
            let factors = runtime.upload(layout.factors().to_vec())?;
            let bytes = codes.len() + factors.len() * size_of::<f32>();
            let graph = device.tile().capture(|scope| {
                kernels.projection.grouped_int6_row(
                    scope,
                    &plan,
                    out_alloc.buffer(),
                    x_alloc.buffer(),
                    WeightRef::GroupedInt6 {
                        codes: codes.buffer(),
                        factors: factors.buffer(),
                    },
                )
            })?;
            (graph, bytes)
        }
        other => return Err(format!("unsupported type {}", other.name()).into()),
    };
    device.tile().sync(graph.launch())?;
    let actual = runtime.read(&out_alloc)?;
    let mut max_relative = 0.0f64;
    for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let relative = f64::from((actual - expected).abs()) / f64::from(expected.abs().max(1.0));
        max_relative = max_relative.max(relative);
        if relative > 2e-3 {
            return Err(format!("row {row}: actual {actual} expected {expected}").into());
        }
    }
    println!(
        "native {} {}x{} correctness max relative {max_relative:.3e}",
        slot.dtype.name(),
        rows,
        columns
    );
    let started = Instant::now();
    for _ in 0..REPLAYS {
        device.tile().sync(graph.launch())?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let bandwidth = bytes as f64 * REPLAYS as f64 / elapsed / 1e9;
    println!(
        "native GGUF GEMV {bandwidth:.1} GB/s ({:.1}% roofline)",
        bandwidth / HBM_GBPS * 100.0
    );
    Ok(())
}
