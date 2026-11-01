//! Storage strategy shared by graph capture and model-specific binding adapters.
use crate::constants::FP8_BLOCK_COLUMNS;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::resident::tensor_buffer as buffer;
use crate::strategy::LinearTiling;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::{half::bf16, prelude::*};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{DownPlan, Kernels, Owner, Recording, WeightRef};
use std::result::Result as StdResult;
use std::sync::Arc;

#[derive(Clone)]
pub enum ProjectionWeight {
    Dense(Arc<Tensor<bf16>>),
    Fp8(Arc<Tensor<f8e4m3fn>>, Arc<Tensor<f32>>),
    /// Block-scaled FP8: the second tensor holds one scale per output channel per 128-column
    /// input block, already expanded along rows so the GEMM can index it like a per-channel one.
    Fp8Block(Arc<Tensor<f8e4m3fn>>, Arc<Tensor<f32>>),
    Fp4(Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>, f32),
}

impl ProjectionWeight {
    /// What keeps this weight's storage alive; the FP4 and FP8 forms also hold their scales.
    pub(crate) fn owner(&self) -> Owner {
        Arc::new(self.clone())
    }

    pub(crate) fn validate(&self, rows: usize, columns: usize) -> Result<()> {
        let matches = |shape: &[i32], expected: &[usize]| {
            shape.len() == expected.len()
                && shape
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| usize::try_from(*a).ok() == Some(*b))
        };
        let valid = match self {
            Self::Dense(w) => matches(w.shape(), &[rows, columns]),
            Self::Fp8(w, s) => matches(w.shape(), &[rows, columns]) && matches(s.shape(), &[rows]),
            Self::Fp8Block(w, s) => {
                columns.is_multiple_of(FP8_BLOCK_COLUMNS)
                    && matches(w.shape(), &[rows, columns])
                    && matches!(s.shape(), [r, c]
                        if usize::try_from(*r).ok() == Some(rows)
                            && usize::try_from(*c).ok()
                                == Some(columns / FP8_BLOCK_COLUMNS))
            }
            Self::Fp4(w, s, g) => {
                columns.is_multiple_of(NVFP4_GROUP_SIZE)
                    && matches(w.shape(), &[rows, columns / 2])
                    && matches(s.shape(), &[rows, columns / NVFP4_GROUP_SIZE])
                    && g.is_finite()
                    && *g > 0.0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Error::invalid("MLP projection dimensions/scales"))
        }
    }

    /// The weight in the form the projection contract takes.
    pub(crate) fn weight_ref(&self) -> StdResult<WeightRef<'_>, DeviceError> {
        match self {
            Self::Dense(w) => Ok(WeightRef::Dense(buffer(w.as_ref())?)),
            Self::Fp8(w, s) => Ok(WeightRef::Fp8 {
                weight: buffer(w.as_ref())?,
                scales: buffer(s.as_ref())?,
            }),
            Self::Fp8Block(..) => Err(DeviceError::Launch(
                "block FP8 requires the resident FP8 GEMM".to_string(),
            )),
            Self::Fp4(w, s, global) => Ok(WeightRef::Fp4 {
                weight: buffer(w.as_ref())?,
                scales: buffer(s.as_ref())?,
                inverse_global: global.recip(),
            }),
        }
    }

    pub(crate) fn record(
        &self,
        kernels: &Kernels,
        scope: &Scope,
        out: &Tensor<f32>,
        input: &Tensor<f32>,
        columns: usize,
        tile: LinearTiling,
    ) -> StdResult<(), DeviceError> {
        kernels
            .projection
            .row(
                &Recording::scope(scope),
                &DownPlan {
                    tile_rows: tile.rows(),
                    tile_columns: tile.columns(),
                    columns,
                },
                buffer(out)?,
                buffer(input)?,
                self.weight_ref()?,
            )
            .map_err(|error| DeviceError::Internal(error.to_string()))?;
        Ok(())
    }
}
