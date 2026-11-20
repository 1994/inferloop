//! Grouped-int projections over one to nine activation rows, every format, against an independent
//! scalar dequantization of the stored GGML blocks.
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
/// Row counts across the single-row GEMV, partial and full row blocks, two blocks and the GEMM.
const ROWS: [usize; 7] = [1, 2, 3, 4, 5, 8, 9];
/// Largest error allowed relative to the sum of absolute products of one output.
const TOLERANCE: f64 = 2e-6;
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

/// Row-major values of the stored blocks, decoded on the host.
fn reference_weight(format: GroupedIntFormat, stored: &[u8]) -> Result<Vec<f64>> {
    let mut span = GroupedSpan::default();
    let mut values = Vec::with_capacity(OUTPUTS * COLUMNS);
    for block in stored.chunks_exact(span_bytes(format)?) {
        decode_span(format, block, &mut span)?;
        for (index, code) in span.codes.iter().enumerate() {
            let group = index / format.group_values();
            values.push(f64::from(format.value(
                *code,
                span.factors[group],
                span.biases[group],
            )));
        }
    }
    Ok(values)
}

fn activations(rows: usize) -> Vec<f32> {
    (0..rows * COLUMNS)
        .map(|i| {
            let step = u16::try_from((i * 37 + i / COLUMNS * 11) % 97).unwrap_or_default();
            (f32::from(step) - 48.0) / 29.0
        })
        .collect()
}

fn upload_weight(
    device: &CudaDevice,
    format: GroupedIntFormat,
) -> Result<(ProjectionWeight, Vec<f64>)> {
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

#[test]
#[ignore = "requires a CUDA device"]
fn every_format_projects_each_row_like_the_scalar_reference() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for format in GroupedIntFormat::ALL {
        let (weight, reference) = upload_weight(&device, format)?;
        for rows in ROWS {
            let output = project(&device, &weight, rows)?;
            let x = activations(rows);
            let tolerance = if rows > 8 { GEMM_TOLERANCE } else { TOLERANCE };
            for row in 0..rows {
                for column in 0..OUTPUTS {
                    let products = (0..COLUMNS)
                        .map(|k| reference[column * COLUMNS + k] * f64::from(x[row * COLUMNS + k]));
                    let (exact, scale) = products.fold((0.0, 0.0), |(sum, magnitude), p| {
                        (sum + p, magnitude + p.abs())
                    });
                    let got = f64::from(output[row * OUTPUTS + column]);
                    if !got.is_finite() || (got - exact).abs() > tolerance * scale.max(1.0) {
                        return Err(Error::invariant(format!(
                            "{} rows={rows} row={row} output={column}: got {got}, expected {exact}",
                            format.label()
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}
