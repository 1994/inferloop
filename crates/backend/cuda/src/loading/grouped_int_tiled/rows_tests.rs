//! Grouped-int projections over one to nine activation rows, every format, against an independent
//! scalar dequantization of the stored GGML blocks; two to eight rows take the int8 dot-product
//! kernel, checked against the same int8 rounding of the activations.
use super::GroupedIntTiled;
use crate::device::{CudaDevice, asynchronous::block_on};
use crate::mlp::{Matrix, ProjectionWeight};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{RowsKernel, RowsPlan};
use infer_cuda_kernels::runtime::Allocation;
use infer_ir::{GROUPED_INT_SPAN, GroupedIntFormat};
use infer_models::{GroupedSpan, decode_span, span_bytes};

const OUTPUTS: usize = 64;
const COLUMNS: usize = 8 * GROUPED_INT_SPAN;
/// Row counts across the single-row GEMV, partial and full row blocks, a full block followed by
/// each partial one, two full blocks and the GEMM.
const ROWS: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
/// Largest error allowed relative to the sum of absolute products of one output.
const TOLERANCE: f64 = 2e-6;
/// Most rows the int8 dot-product kernel takes; from two rows it replaces the F32 GEMV.
const DOT_ROWS: usize = 8;
/// Activations one int8 scale covers.
const DOT_GROUP: usize = 32;
/// Largest int8 magnitude.
const INT8_MAX: f32 = 127.0;
/// The int8 kernel's agreement with the unquantized product, relative to the absolute products:
/// each activation moves by at most half an int8 step, `amax / 254`.
const DOT_QUANTIZATION: f64 = 1.0 / 254.0;
/// The GEMM rounds weights and activations to BF16.
const GEMM_TOLERANCE: f64 = 1e-2;

/// Deterministic blocks whose F16 fields stay finite: the top exponent bits are never all set.
fn blocks(format: GroupedIntFormat) -> Result<Vec<u8>> {
    let len = OUTPUTS * (COLUMNS / GROUPED_INT_SPAN) * span_bytes(format)?;
    let mut state = 0x2545_f491_u32;
    Ok((0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8 & 0xbf
        })
        .collect())
}

/// Row-major values of the stored blocks, decoded on the host, and the bias each value subtracts
/// (zero without biases).
fn reference_weight(format: GroupedIntFormat, stored: &[u8]) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut span = GroupedSpan::default();
    let mut values = Vec::with_capacity(OUTPUTS * COLUMNS);
    let mut biases = Vec::with_capacity(OUTPUTS * COLUMNS);
    for block in stored.chunks_exact(span_bytes(format)?) {
        decode_span(format, block, &mut span)?;
        for (index, code) in span.codes.iter().enumerate() {
            let group = index / format.group_values();
            let bias = span.biases[group];
            values.push(f64::from(format.value(*code, span.factors[group], bias)));
            biases.push(if format.has_biases() {
                f64::from(bias)
            } else {
                0.0
            });
        }
    }
    Ok((values, biases))
}

/// `x` rounded to int8 per [`DOT_GROUP`] values and scaled back, in the kernel's F32 steps:
/// `q = round(x * (127 / amax))`, ties to even, times `amax / 127`.
fn quantized(x: &[f32]) -> Vec<f64> {
    x.as_chunks::<DOT_GROUP>()
        .0
        .iter()
        .flat_map(|group| {
            let amax = group.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let inverse = if amax > 0.0 { INT8_MAX / amax } else { 0.0 };
            let scale = amax / INT8_MAX;
            group
                .iter()
                .map(move |value| f64::from((value * inverse).round_ties_even()) * f64::from(scale))
        })
        .collect()
}

fn activations(rows: usize) -> Vec<f32> {
    (0..rows * COLUMNS)
        .map(|i| {
            let step = u16::try_from((i * 37 + i / COLUMNS * 11) % 97).unwrap_or_default();
            (f32::from(step) - 48.0) / 29.0
        })
        .collect()
}

/// A packed weight on the device with its host values and per-value biases.
type Uploaded = (ProjectionWeight, (Vec<f64>, Vec<f64>));

fn upload_weight(device: &CudaDevice, format: GroupedIntFormat) -> Result<Uploaded> {
    let stored = blocks(format)?;
    let reference = reference_weight(format, &stored)?;
    let tiled = GroupedIntTiled::from_blocks(format, OUTPUTS, COLUMNS, &stored)?;
    let code_shape = tiled.code_shape();
    let (codes, factors, biases) = tiled.into_parts();
    let weight = ProjectionWeight::GroupedInt {
        codes: Matrix::upload(device, codes, &code_shape)?,
        factors: device.runtime().upload(factors)?,
        biases: biases
            .map(|biases| device.runtime().upload(biases))
            .transpose()?,
        format,
    };
    Ok((weight, reference))
}

fn project(device: &CudaDevice, weight: &ProjectionWeight, rows: usize) -> Result<Vec<f32>> {
    let input = Allocation::adopt(device.upload(activations(rows), &[rows, COLUMNS])?)?;
    let output =
        Allocation::adopt(device.upload(vec![f32::NAN; rows * OUTPUTS], &[rows, OUTPUTS])?)?;
    let kernels = device.kernels();
    let plan = RowsPlan {
        kernel: RowsKernel::GroupedInt,
        rows,
        outputs: OUTPUTS,
        columns: COLUMNS,
    };
    let graph = device.runtime().capture(|recording| {
        kernels.projection.rows(
            recording,
            &plan,
            output.buffer(),
            input.buffer(),
            weight.weight_ref()?,
        )
    })?;
    block_on(device.runtime().run(graph.launch()))?;
    device.read(output.tensor())
}

/// One output's expected value and the sum of its absolute products. The int8 kernel sees the
/// quantized activations `seen` in the code term and subtracts each bias times the exact `x`.
fn expected(values: &[f64], biases: &[f64], x: &[f32], seen: &[f64]) -> (f64, f64) {
    values.iter().zip(biases).zip(x.iter().zip(seen)).fold(
        (0.0, 0.0),
        |(sum, magnitude), ((value, bias), (x, seen))| {
            let product = (value + bias) * seen - bias * f64::from(*x);
            (sum + product, magnitude + product.abs())
        },
    )
}

/// Check every output of `rows` rows against `reference`, and the int8 rows against the
/// unquantized product within the quantization bound.
fn check_rows(
    format: GroupedIntFormat,
    (values, biases): &(Vec<f64>, Vec<f64>),
    rows: usize,
    output: &[f32],
) -> Result<()> {
    let x = activations(rows);
    let dot = (2..=DOT_ROWS).contains(&rows);
    let exact: Vec<f64> = x.iter().copied().map(f64::from).collect();
    let seen = if dot { quantized(&x) } else { exact.clone() };
    let tolerance = if rows > DOT_ROWS {
        GEMM_TOLERANCE
    } else {
        TOLERANCE
    };
    let bound = if dot { DOT_QUANTIZATION } else { tolerance };
    for row in 0..rows {
        let inputs = &x[row * COLUMNS..][..COLUMNS];
        for column in 0..OUTPUTS {
            let values = &values[column * COLUMNS..][..COLUMNS];
            let biases = &biases[column * COLUMNS..][..COLUMNS];
            let (want, scale) = expected(values, biases, inputs, &seen[row * COLUMNS..][..COLUMNS]);
            let (plain, plain_scale) =
                expected(values, biases, inputs, &exact[row * COLUMNS..][..COLUMNS]);
            let got = f64::from(output[row * OUTPUTS + column]);
            if !got.is_finite()
                || (got - want).abs() > tolerance * scale.max(1.0)
                || (got - plain).abs() > bound * plain_scale.max(1.0)
            {
                return Err(Error::invariant(format!(
                    "{} rows={rows} row={row} output={column}: got {got}, expected {want}, \
                     unquantized {plain}",
                    format.label()
                )));
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_format_projects_each_row_like_the_scalar_reference() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for format in GroupedIntFormat::ALL {
        let (weight, reference) = upload_weight(&device, format)?;
        for rows in ROWS {
            let output = project(&device, &weight, rows)?;
            check_rows(format, &reference, rows, &output)?;
        }
    }
    Ok(())
}
