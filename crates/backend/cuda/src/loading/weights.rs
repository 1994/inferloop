use super::grouped_int_tiled::GroupedIntTiled;
use super::head_padding::Pad;
use crate::constants::DEFAULT_TILE_COLUMNS;
use crate::constants::DEFAULT_TILE_ROWS;
use crate::constants::F32_BYTES;
use crate::constants::GIB;
use crate::mlp::{Matrix, ProjectionWeight};
use crate::nvfp4::to_bf16;
use crate::{device::CudaDevice, strategy::LinearTiling};
use infer_core::{Error, Result};
use infer_cuda_kernels::runtime::bf16;
use infer_cuda_kernels::runtime::{f4e2m1fnx2, f8e4m3fn};
use infer_ir::DType;
use infer_models::{
    DeviceWeight, QuantizedPackage, WeightEncoding, WeightSource, WeightView, convert_float_bytes,
};

const STAGING: u64 = 2 * 1024 * 1024 * 1024;
/// Decoded BF16 byte budget for one NVFP4 projection falling back to host decoding.
const NVFP4_DECODE_BUDGET: usize = GIB;

// E4M3 finite encoding: four biased exponent bits and three fraction bits.
const E4M3_NAN_CODE: u8 = 127;
const E4M3_MAX_CODE: u8 = E4M3_NAN_CODE - 1;
const E4M3_MANTISSA_BITS: u8 = 3;
const E4M3_MAX: f32 = 448.0;
const E4M3_SIGN: u8 = 0x80;
/// Smallest normal E4M3 magnitude, 2^-6.
const E4M3_MIN_NORMAL: f32 = 0.015_625;
/// Subnormal steps per unit, 2^9: E4M3 subnormals are multiples of 2^-9.
const E4M3_SUBNORMAL_STEPS: f32 = 512.0;
const F32_MANTISSA_BITS: u32 = 23;
/// The f32 exponent bias less the E4M3 one (127 - 7), in E4M3 code units.
const E4M3_REBIAS: u32 = (127 - 7) << 3;

/// Read floating weights under a 2 GiB encoded staging limit.
/// # Errors
/// Rejects invalid storage, unsupported floating formats or failed reads.
pub fn floats(package: &mut QuantizedPackage, source: &WeightSource) -> Result<Vec<f32>> {
    let gathered = source
        .view
        .as_ref()
        .filter(|view| view.gathered())
        .map(|_| package.read(source, STAGING))
        .transpose()?;
    let bytes = if let Some(bytes) = &gathered {
        bytes.as_slice()
    } else {
        package.read_view(source, STAGING)?
    };
    let converted = convert_float_bytes(bytes, source.dtype)?;
    Ok(converted
        .as_chunks::<{ F32_BYTES }>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// FP8 codes of one stored weight; a gathered view is copied out, a contiguous one is borrowed.
fn codes(package: &mut QuantizedPackage, source: &WeightSource) -> Result<Vec<f8e4m3fn>> {
    if source.view.as_ref().is_some_and(WeightView::gathered) {
        return Ok(package
            .read(source, STAGING)?
            .into_iter()
            .map(f8e4m3fn)
            .collect());
    }
    Ok(package
        .read_view(source, STAGING)?
        .iter()
        .copied()
        .map(f8e4m3fn)
        .collect())
}

/// Round to the nearest E4M3 code (ties to the even code), saturating at +-448; NaN saturates.
///
/// Normal magnitudes round the f32 fraction to three bits in place: the carry of a rounded-up
/// fraction moves into the exponent, which is the next code. Magnitudes below the smallest normal
/// are multiples of the subnormal step, scaled exactly by a power of two.
fn e4m3_code(value: f32) -> u8 {
    let magnitude = value.abs().min(E4M3_MAX);
    let code = if magnitude < E4M3_MIN_NORMAL {
        // At most eight steps, and exact: a power-of-two scale of a value below 2^-6.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a rounded non-negative multiple of the subnormal step, at most eight"
        )]
        let steps = (magnitude * E4M3_SUBNORMAL_STEPS).round_ties_even() as u8;
        steps
    } else {
        let bits = magnitude.to_bits();
        let dropped = F32_MANTISSA_BITS - u32::from(E4M3_MANTISSA_BITS);
        let half = (1 << (dropped - 1)) - 1;
        let rounded = (bits + half + ((bits >> dropped) & 1)) >> dropped;
        u8::try_from(rounded - E4M3_REBIAS).unwrap_or(E4M3_MAX_CODE)
    };
    let sign = if value.is_sign_negative() && code != 0 {
        E4M3_SIGN
    } else {
        0
    };
    code.min(E4M3_MAX_CODE) | sign
}

impl ProjectionWeight {
    /// Per-output-channel FP8 encoding of a dense `[rows, columns]` weight: each row is scaled by
    /// its largest magnitude over 448, matching the checkpoint's own FP8-channel projections.
    /// # Errors
    /// Rejects malformed shapes or failed CUDA allocations.
    pub fn quantize_fp8(device: &CudaDevice, values: &[f32], shape: &[usize]) -> Result<Self> {
        let [rows, columns] = shape else {
            return Err(Error::invalid("FP8 projection rank"));
        };
        if rows * columns != values.len() || *columns == 0 {
            return Err(Error::invalid("FP8 projection shape"));
        }
        let mut codes = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(*rows);
        for row in values.chunks(*columns) {
            let largest = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let scale = if largest > 0.0 {
                largest / E4M3_MAX
            } else {
                1.0
            };
            scales.push(scale);
            codes.extend(row.iter().map(|v| f8e4m3fn(e4m3_code(v / scale))));
        }
        Ok(Self::Fp8(
            Matrix::upload(device, codes, shape)?,
            device.runtime().upload(scales)?,
        ))
    }

    /// # Errors
    /// Rejects malformed quantization, staging limits or failed CUDA allocations.
    pub fn load(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        weight: &DeviceWeight,
    ) -> Result<Self> {
        let scale = || {
            weight
                .scale()
                .ok_or_else(|| Error::invalid("missing projection scale"))
        };
        match weight.encoding() {
            WeightEncoding::Float => {
                let values = floats(package, weight.data())?
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect();
                Ok(Self::Dense(Matrix::upload(
                    device,
                    values,
                    &weight.shape(),
                )?))
            }
            WeightEncoding::Fp8Channel => {
                let values = codes(package, weight.data())?;
                let scaling = floats(package, scale()?)?;
                if weight.shape().first() != Some(&scaling.len()) {
                    return Err(Error::invalid("FP8 channel scales"));
                }
                Ok(Self::Fp8(
                    Matrix::upload(device, values, &weight.shape())?,
                    device.runtime().upload(scaling)?,
                ))
            }
            WeightEncoding::Nvfp4 => Self::load_nvfp4(device, package, weight),
            WeightEncoding::GroupedInt(_) => Self::load_grouped_int(device, package, weight),
            WeightEncoding::Fp8Block => {
                let values = codes(package, weight.data())?;
                let shape = weight.shape();
                let [rows, columns] = shape.as_slice() else {
                    return Err(Error::invalid("block FP8 weights must be matrices"));
                };
                let [block_rows, block_columns] = weight
                    .block()
                    .ok_or_else(|| Error::invalid("missing FP8 block size"))?;
                let scaling = floats(package, scale()?)?;
                let blocks = columns / block_columns;
                // Expand one scale per output-channel block across its rows so the resident GEMM
                // indexes the scale exactly like the per-channel path. The block rows are output
                // channels, so every channel in a block shares one scale.
                let mut expanded = Vec::with_capacity(rows * blocks);
                for row in 0..*rows {
                    let start = (row / block_rows) * blocks;
                    expanded.extend_from_slice(&scaling[start..start + blocks]);
                }
                Ok(Self::Fp8Block(
                    Matrix::upload(device, values, &shape)?,
                    Matrix::upload(device, expanded, &[*rows, blocks])?,
                ))
            }
        }
    }

    /// [`Self::load`] into a head-padded layout. Only encodings whose padding is exact load:
    /// dense values, and FP8 with one scale per output channel.
    /// # Errors
    /// Returns `Unsupported` for block-scaled encodings, which padding would misalign.
    pub(crate) fn load_padded(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        weight: &DeviceWeight,
        pad: &Pad,
        padded_shape: &[usize],
    ) -> Result<Self> {
        // The graph carries the padded shape; recover the checkpoint's from the payload.
        let checkpoint = |len: usize| -> Result<(usize, usize)> {
            let &[rows, columns] = padded_shape else {
                return Err(Error::invalid("padded projection must be a matrix"));
            };
            let checkpoint = match pad {
                Pad::Columns { .. } if rows > 0 => (rows, len / rows),
                Pad::Rows { .. } if columns > 0 => (len / columns, columns),
                _ => return Err(Error::invalid("padded projection shape")),
            };
            let (rows, columns) = checkpoint;
            // Padding changes the graph shape, never the stored descriptor the payload keeps.
            if weight.data().shape != [rows, columns] {
                return Err(Error::invalid("padded weight source shape"));
            }
            Ok((rows, columns))
        };
        match weight.encoding() {
            WeightEncoding::Float => {
                let values = floats(package, weight.data())?;
                let (padded, shape) =
                    pad.apply(&values, checkpoint(values.len())?, 0.0, |value, scale| {
                        value * scale
                    })?;
                let values = padded.into_iter().map(bf16::from_f32).collect();
                Ok(Self::Dense(Matrix::upload(device, values, &shape)?))
            }
            WeightEncoding::Fp8Channel => {
                let values = codes(package, weight.data())?;
                let scaling = floats(
                    package,
                    weight
                        .scale()
                        .ok_or_else(|| Error::invalid("missing projection scale"))?,
                )?;
                // A query row's correction scales its channel scale, not its codes.
                let (codes, shape) = pad.apply(
                    &values,
                    checkpoint(values.len())?,
                    f8e4m3fn(0),
                    |code, _| code,
                )?;
                let scaling = match pad {
                    Pad::Rows { .. } => {
                        pad.apply(&scaling, (scaling.len(), 1), 0.0, |value, scale| {
                            value * scale
                        })?
                        .0
                    }
                    _ => scaling,
                };
                if shape.first() != Some(&scaling.len()) {
                    return Err(Error::invalid("FP8 channel scales"));
                }
                Ok(Self::Fp8(
                    Matrix::upload(device, codes, &shape)?,
                    device.runtime().upload(scaling)?,
                ))
            }
            WeightEncoding::Nvfp4 | WeightEncoding::Fp8Block | WeightEncoding::GroupedInt(_) => {
                Err(Error::unsupported(
                    "padded CUDA attention heads need dense or per-channel FP8 projections",
                ))
            }
        }
    }

    fn load_grouped_int(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        weight: &DeviceWeight,
    ) -> Result<Self> {
        let DeviceWeight::GroupedInt { data, format } = weight else {
            return Err(Error::invalid("grouped-int weight descriptor"));
        };
        let shape = weight.shape();
        let [rows, columns] = shape.as_slice() else {
            return Err(Error::invalid("grouped-int weight must be a matrix"));
        };
        let raw = package.read(data, STAGING)?;
        let layout = GroupedIntTiled::from_blocks(*format, *rows, *columns, &raw)
            .map_err(|error| Error::invalid(format!("{}: {error}", data.name)))?;
        // The raw blocks drop before upload and the packed parts move into it, so the host peak
        // holds at most the stored and the packed copy of one weight.
        drop(raw);
        let code_shape = layout.code_shape();
        let (codes, factors, biases) = layout.into_parts();
        Ok(Self::GroupedInt {
            codes: Matrix::upload(device, codes, &code_shape).map_err(|error| {
                Error::invalid(format!("{} grouped-int codes: {error}", data.name))
            })?,
            factors: device.runtime().upload(factors)?,
            biases: biases
                .map(|biases| device.runtime().upload(biases))
                .transpose()?,
            format: *format,
        })
    }

    fn load_nvfp4(
        device: &CudaDevice,
        package: &mut QuantizedPackage,
        weight: &DeviceWeight,
    ) -> Result<Self> {
        // The model provider declares its storage preference; the device answers what it can
        // compute. No architecture enum is consulted here.
        let native = package
            .imported
            .precision
            .resolve(|dtype| device.target().supports_compute(dtype))
            .storage
            == DType::Fp4E2M1;
        let scale = weight
            .scale()
            .ok_or_else(|| Error::invalid("missing NVFP4 block scale"))?;
        let global = weight
            .global_scale()
            .ok_or_else(|| Error::invalid("missing NVFP4 global scale"))?;
        let global = floats(package, global)?;
        let [global] = global.as_slice() else {
            return Err(Error::invalid("NVFP4 global scale must be scalar"));
        };
        if !global.is_finite() || *global <= 0.0 {
            return Err(Error::invalid("invalid NVFP4 global scale"));
        }
        let values = package.read_view(weight.data(), STAGING)?;
        let scaling = package.read_view(scale, STAGING)?;
        if native {
            Ok(Self::Fp4(
                Matrix::upload(
                    device,
                    values.iter().copied().map(f4e2m1fnx2::from_bits).collect(),
                    &weight.data().shape,
                )?,
                Matrix::upload(
                    device,
                    scaling.iter().copied().map(f8e4m3fn).collect(),
                    &scale.shape,
                )?,
                *global,
            ))
        } else {
            let shape = weight.shape();
            let [rows, columns] = shape.as_slice() else {
                return Err(Error::invalid("NVFP4 projection must be a matrix"));
            };
            let decoded = to_bf16(
                values,
                scaling,
                *global,
                *rows,
                *columns,
                NVFP4_DECODE_BUDGET,
            )?;
            Ok(Self::Dense(Matrix::upload(
                device,
                decoded.into_iter().map(bf16::from_bits).collect(),
                &shape,
            )?))
        }
    }

    /// # Errors
    /// Rejects incompatible projection dimensions or failed CUDA execution.
    pub fn apply(&self, device: &CudaDevice, input: &[f32]) -> Result<Vec<f32>> {
        let vector = device.runtime().upload(input.to_vec())?;
        let tile = LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS)?;
        let output = match self {
            Self::Dense(weights) => device.matvec_tiled(&vector, weights, tile)?,
            Self::Fp8(weights, scales) => device.fp8_matvec(&vector, weights, scales, tile)?,
            Self::Fp8Block(..) => {
                return Err(Error::unsupported("block FP8 has no host matvec fallback"));
            }
            Self::Fp4(weights, scales, global) => {
                device.nvfp4_matvec(&vector, weights, scales, *global, tile)?
            }
            Self::GroupedInt { .. } => {
                return Err(Error::unsupported("grouped-int host matvec"));
            }
        };
        device.runtime().read(&output)
    }
}

#[cfg(test)]
mod tests;
