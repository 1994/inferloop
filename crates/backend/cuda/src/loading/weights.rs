use crate::constants::DEFAULT_TILE_COLUMNS;
use crate::constants::DEFAULT_TILE_ROWS;
use crate::constants::F32_BYTES;
use crate::constants::GIB;
use crate::mlp::ProjectionWeight as MlpProjectionWeight;
use crate::nvfp4::to_bf16;
use crate::{device::CudaDevice, strategy::LinearTiling};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use infer_ir::DType;
use infer_models::{
    DeviceWeight, QuantizedPackage, WeightEncoding, WeightSource, WeightView, convert_float_bytes,
};
use std::cmp::Ordering;
use std::sync::Arc;

const STAGING: u64 = 2 * 1024 * 1024 * 1024;
/// Decoded BF16 byte budget for one NVFP4 projection falling back to host decoding.
const NVFP4_DECODE_BUDGET: usize = GIB;

// E4M3 finite encoding: four biased exponent bits and three fraction bits.
const E4M3_NAN_CODE: u8 = 127;
const E4M3_MAX_CODE: u8 = E4M3_NAN_CODE - 1;
const E4M3_MANTISSA_BITS: u8 = 3;
const E4M3_MANTISSA_MASK: u8 = 7;
const E4M3_MANTISSA_SCALE: f32 = 8.0;
const E4M3_EXPONENT_BIAS: i32 = 7;
const E4M3_MAX: f32 = 448.0;
const E4M3_SIGN: u8 = 0x80;

#[derive(Clone)]
pub enum Projection {
    Dense(Arc<Tensor<bf16>>),
    Fp8(Arc<Tensor<f8e4m3fn>>, Arc<Tensor<f32>>),
    Fp8Block(Arc<Tensor<f8e4m3fn>>, Arc<Tensor<f32>>),
    Fp4(Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>, f32),
}

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

/// Finite non-negative E4M3 values in code order (codes 0..=126; 127 is NaN).
fn e4m3_magnitudes() -> Vec<f32> {
    (0u8..E4M3_NAN_CODE)
        .map(|code| {
            let exponent = i32::from(code >> E4M3_MANTISSA_BITS);
            let mantissa = f32::from(code & E4M3_MANTISSA_MASK);
            if exponent == 0 {
                mantissa / E4M3_MANTISSA_SCALE * 2f32.powi(1 - E4M3_EXPONENT_BIAS)
            } else {
                (1.0 + mantissa / E4M3_MANTISSA_SCALE) * 2f32.powi(exponent - E4M3_EXPONENT_BIAS)
            }
        })
        .collect()
}

/// Round to the nearest E4M3 code (ties to the even code), saturating at +-448.
fn e4m3_code(value: f32, magnitudes: &[f32]) -> u8 {
    let magnitude = value.abs().min(E4M3_MAX);
    let upper = magnitudes.partition_point(|&m| m < magnitude);
    let code = if upper == 0 {
        0
    } else if upper >= magnitudes.len() {
        magnitudes.len() - 1
    } else {
        let below = magnitude - magnitudes[upper - 1];
        let above = magnitudes[upper] - magnitude;
        match below.total_cmp(&above) {
            Ordering::Less => upper - 1,
            Ordering::Equal if (upper - 1) % 2 == 0 => upper - 1,
            Ordering::Greater | Ordering::Equal => upper,
        }
    };
    let sign = if value.is_sign_negative() && code != 0 {
        E4M3_SIGN
    } else {
        0
    };
    u8::try_from(code).unwrap_or(E4M3_MAX_CODE) | sign
}

impl Projection {
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
        let magnitudes = e4m3_magnitudes();
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
            codes.extend(
                row.iter()
                    .map(|v| f8e4m3fn(e4m3_code(v / scale, &magnitudes))),
            );
        }
        Ok(Self::Fp8(
            device.upload(codes, shape)?,
            device.upload(scales, &[*rows])?,
        ))
    }

    #[must_use]
    pub fn resident(&self) -> MlpProjectionWeight {
        use crate::mlp::ProjectionWeight;
        match self {
            Self::Dense(w) => ProjectionWeight::Dense(w.clone()),
            Self::Fp8(w, s) => ProjectionWeight::Fp8(w.clone(), s.clone()),
            Self::Fp8Block(w, s) => ProjectionWeight::Fp8Block(w.clone(), s.clone()),
            Self::Fp4(w, s, g) => ProjectionWeight::Fp4(w.clone(), s.clone(), *g),
        }
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
                .scale
                .as_ref()
                .ok_or_else(|| Error::invalid("missing projection scale"))
        };
        match weight.encoding {
            WeightEncoding::Float => {
                let values = floats(package, &weight.data)?
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect();
                Ok(Self::Dense(device.upload(values, &weight.shape)?))
            }
            WeightEncoding::Fp8Channel => {
                let values = codes(package, &weight.data)?;
                let scaling = floats(package, scale()?)?;
                Ok(Self::Fp8(
                    device.upload(values, &weight.shape)?,
                    device.upload(scaling, &[weight.shape[0]])?,
                ))
            }
            WeightEncoding::Nvfp4 => Self::load_nvfp4(device, package, weight),
            WeightEncoding::Fp8Block => {
                let values = codes(package, &weight.data)?;
                let [rows, columns] = weight.shape.as_slice() else {
                    return Err(Error::invalid("block FP8 weights must be matrices"));
                };
                let [block_rows, block_columns] = weight
                    .weight_block
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
                    device.upload(values, &weight.shape)?,
                    device.upload(expanded, &[*rows, blocks])?,
                ))
            }
        }
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
            .scale
            .as_ref()
            .ok_or_else(|| Error::invalid("missing NVFP4 block scale"))?;
        let global = weight
            .global_scale
            .as_ref()
            .ok_or_else(|| Error::invalid("missing NVFP4 global scale"))?;
        let global = floats(package, global)?;
        let [global] = global.as_slice() else {
            return Err(Error::invalid("NVFP4 global scale must be scalar"));
        };
        if !global.is_finite() || *global <= 0.0 {
            return Err(Error::invalid("invalid NVFP4 global scale"));
        }
        let values = package.read_view(&weight.data, STAGING)?;
        let scaling = package.read_view(scale, STAGING)?;
        if native {
            Ok(Self::Fp4(
                device.upload(
                    values.iter().copied().map(f4e2m1fnx2::from_bits).collect(),
                    &weight.data.shape,
                )?,
                device.upload(
                    scaling.iter().copied().map(f8e4m3fn).collect(),
                    &scale.shape,
                )?,
                *global,
            ))
        } else {
            let [rows, columns] = weight.shape.as_slice() else {
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
            Ok(Self::Dense(device.upload(
                decoded.into_iter().map(bf16::from_bits).collect(),
                &weight.shape,
            )?))
        }
    }

    /// # Errors
    /// Rejects incompatible projection dimensions or failed CUDA execution.
    pub fn apply(&self, device: &CudaDevice, input: &[f32]) -> Result<Vec<f32>> {
        let vector = device.upload(input.to_vec(), &[input.len()])?;
        let tile = LinearTiling::new(DEFAULT_TILE_ROWS, DEFAULT_TILE_COLUMNS)?;
        let output = match self {
            Self::Dense(weights) => device.matvec_tiled(vector, weights.clone(), tile)?,
            Self::Fp8(weights, scales) => {
                device.fp8_matvec(vector, weights.clone(), scales.clone(), tile)?
            }
            Self::Fp8Block(..) => {
                return Err(Error::unsupported("block FP8 has no host matvec fallback"));
            }
            Self::Fp4(weights, scales, global) => {
                device.nvfp4_matvec(vector, weights.clone(), scales.clone(), *global, tile)?
            }
        };
        device.read(&output)
    }
}
