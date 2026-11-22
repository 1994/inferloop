//! Native GGUF tensor projection for any grouped-int format: read real blocks, pack, run the
//! GEMV (one row), the int8 dot-product kernel (two to eight rows) or the rows kernel (more rows)
//! and compare with the reference decode. The int8 kernel is held to its quantization bound; the
//! rows kernel multiplies BF16 operands, so its reference rounds both to BF16.
use infer_backend_cuda::device::CudaDevice;
use infer_backend_cuda::loading::GroupedIntTiled;
use infer_cuda_kernels::contract::{
    GROUPED_INT_TILE_ROWS, GroupedIntPlan, GroupedIntWeight, RowsKernel, RowsPlan, WeightRef,
};
use infer_cuda_kernels::runtime::{Allocation, Graph};
use infer_models::GgufPackage;
use std::env::args;
use std::error::Error;
use std::time::Instant;

const REPLAYS: usize = 50;
const HBM_GBPS: f64 = 1792.0;
/// Weight bytes the replays rotate over, past the L2 so every launch reads from DRAM.
const COLD_BYTES: usize = 256 << 20;
const USAGE: &str = "usage: cuda-gguf-gemv <model.gguf> <hf-slot> [rows]";
/// Largest relative error accepted against the reference.
const TOLERANCE: f64 = 2e-3;
/// Rows from which the int8 dot-product kernel replaces the F32 GEMV.
const DOT_ROWS: usize = 2;
/// The int8 kernel's error bound relative to an output's absolute products: each activation moves
/// by at most half an int8 step, `amax / 254`.
const DOT_TOLERANCE: f64 = 1.0 / 254.0;
/// Most rows the F32 GEMV takes; more take the BF16 rows kernel.
const GEMV_ROWS: usize = 8;
/// Low mantissa bits an F32 drops when rounded to BF16.
const BF16_SHIFT: u32 = 16;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = args().skip(1);
    let path = arguments.next().ok_or(USAGE)?;
    let slot_name = arguments.next().ok_or(USAGE)?;
    let batch = arguments
        .next()
        .map_or(Ok(1), |rows| rows.parse::<usize>())?;
    let package = GgufPackage::open(path)?;
    let slot = package
        .slots()
        .get(&slot_name)
        .ok_or_else(|| format!("missing slot {slot_name}"))?
        .clone();
    let [rows, columns] = slot.shape.as_slice() else {
        return Err("GEMV slot must be rank 2".into());
    };
    if !rows.is_multiple_of(GROUPED_INT_TILE_ROWS) {
        return Err(format!("rows {rows} is not a multiple of {GROUPED_INT_TILE_ROWS}").into());
    }
    let x = inputs(batch, *columns);
    let expected = expected(package.dense(&slot_name, u64::MAX)?, *columns, &x, batch);

    let device = CudaDevice::new(0)?;
    let runtime = device.runtime();
    let x_alloc = runtime.upload(x)?;
    let out_alloc = runtime.zeros::<f32>(batch * rows)?;
    let plan = GroupedIntPlan {
        tile_rows: 16,
        columns: *columns,
    };
    let kernels = device.kernels();
    let format = slot
        .dtype
        .grouped_int()
        .ok_or_else(|| format!("{} is not a grouped-int type", slot.dtype.name()))?;
    let layout = GroupedIntTiled::from_blocks(
        format,
        *rows,
        *columns,
        &package.blocks(&slot_name, u64::MAX)?,
    )?;
    let (codes, factors, biases) = layout.into_parts();
    let bytes =
        codes.len() + (factors.len() + biases.as_ref().map_or(0, Vec::len)) * size_of::<f32>();
    // Rotate over enough weight copies that each launch reads from DRAM, as decode does.
    let copies = COLD_BYTES.div_ceil(bytes).max(1);
    let weights = (0..copies)
        .map(|_| {
            Ok((
                runtime.upload(codes.clone())?,
                runtime.upload(factors.clone())?,
                biases
                    .clone()
                    .map(|biases| runtime.upload(biases))
                    .transpose()?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let rows_plan = RowsPlan {
        kernel: RowsKernel::GroupedInt,
        rows: batch,
        outputs: *rows,
        columns: *columns,
    };
    let graph = device.tile().capture(|scope| {
        for (codes, factors, biases) in &weights {
            let weight = GroupedIntWeight {
                format,
                codes: codes.buffer(),
                factors: factors.buffer(),
                biases: biases.as_ref().map(Allocation::buffer),
            };
            if batch == 1 {
                kernels.projection.grouped_int(
                    scope,
                    &plan,
                    out_alloc.buffer(),
                    x_alloc.buffer(),
                    weight,
                )?;
            } else {
                kernels.projection.rows(
                    scope,
                    &rows_plan,
                    out_alloc.buffer(),
                    x_alloc.buffer(),
                    WeightRef::GroupedInt(weight),
                )?;
            }
        }
        Ok(())
    })?;
    device.tile().sync(graph.launch())?;
    let actual = runtime.read(&out_alloc)?;
    let max_relative = check(&actual, &expected, tolerance(batch))?;
    println!(
        "native {} {batch}x{rows}x{columns} correctness max relative {max_relative:.3e}",
        slot.dtype.name(),
    );
    time(&device, &graph, bytes, copies)
}

/// `rows` activation rows; each row shifts the pattern so rows differ.
fn inputs(rows: usize, columns: usize) -> Vec<f32> {
    let mut values = vec![0.0f32; rows * columns];
    for (index, value) in values.iter_mut().enumerate() {
        *value = match (index % columns + index / columns) % 4 {
            0 => 0.5,
            1 => -0.25,
            2 => 1.0,
            _ => -1.0,
        };
    }
    values
}

/// The error bound of `batch` rows: the int8 kernel's quantization bound, or the F32/BF16 one.
fn tolerance(batch: usize) -> f64 {
    if (DOT_ROWS..=GEMV_ROWS).contains(&batch) {
        DOT_TOLERANCE
    } else {
        TOLERANCE
    }
}

/// The largest error of `actual` relative to each output's absolute products, at most `tolerance`.
fn check(actual: &[f32], expected: &[(f32, f32)], tolerance: f64) -> Result<f64, Box<dyn Error>> {
    let mut max_relative = 0.0f64;
    for (row, (actual, &(expected, magnitude))) in actual.iter().zip(expected).enumerate() {
        let relative = f64::from((actual - expected).abs()) / f64::from(magnitude.max(1.0));
        max_relative = max_relative.max(relative);
        if relative > tolerance {
            return Err(format!("row {row}: actual {actual} expected {expected}").into());
        }
    }
    Ok(max_relative)
}

/// Replay `graph`, whose launches each project one of `copies` weight copies of `bytes`.
fn time(
    device: &CudaDevice,
    graph: &Graph,
    bytes: usize,
    copies: usize,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    for _ in 0..REPLAYS {
        device.tile().sync(graph.launch())?;
    }
    report(
        bytes,
        started.elapsed().as_secs_f64() / f64::from(u32::try_from(copies)?),
    );
    Ok(())
}

fn report(bytes: usize, elapsed: f64) {
    let bytes = f64::from(u32::try_from(bytes / 1024).unwrap_or(u32::MAX)) * 1024.0;
    let replays = f64::from(u32::try_from(REPLAYS).unwrap_or(u32::MAX));
    let bandwidth = bytes * replays / elapsed / 1e9;
    println!(
        "native GGUF projection {:.1} us, {bandwidth:.1} GB/s weight ({:.1}% roofline)",
        elapsed / replays * 1e6,
        bandwidth / HBM_GBPS * 100.0
    );
}

/// Round to the nearest BF16, ties to even, as the device conversion does for finite values.
const fn to_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = (1 << (BF16_SHIFT - 1)) - 1 + ((bits >> BF16_SHIFT) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & !((1 << BF16_SHIFT) - 1))
}

/// The reference outputs of `batch` rows of `x`; more than [`GEMV_ROWS`] rows run the BF16 rows
/// kernel, so their reference rounds both operands to BF16.
fn expected(mut values: Vec<f32>, columns: usize, x: &[f32], batch: usize) -> Vec<(f32, f32)> {
    let rows = values.len() / columns;
    if batch <= GEMV_ROWS {
        return x
            .chunks_exact(columns)
            .flat_map(|x| reference(&values, rows, columns, x))
            .collect();
    }
    for value in &mut values {
        *value = to_bf16(*value);
    }
    let x = x.iter().copied().map(to_bf16).collect::<Vec<_>>();
    x.chunks_exact(columns)
        .flat_map(|x| reference(&values, rows, columns, x))
        .collect()
}

/// Each output of one activation row and the sum of its absolute products.
fn reference(values: &[f32], rows: usize, columns: usize, x: &[f32]) -> Vec<(f32, f32)> {
    (0..rows)
        .map(|row| {
            values[row * columns..][..columns].iter().zip(x).fold(
                (0.0, 0.0),
                |(sum, magnitude), (value, activation)| {
                    let product = value * activation;
                    (sum + product, magnitude + product.abs())
                },
            )
        })
        .collect()
}
