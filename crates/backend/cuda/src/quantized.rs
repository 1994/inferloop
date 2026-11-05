//! Quantized storage launch adapters; precision policy stays explicit.
use crate::constants::NVFP4_GROUP_SIZE;
use crate::mlp::Matrix;
use crate::{device::CudaDevice, strategy::LinearTiling};
use cuda_core::{DType, f4e2m1fnx2, f8e4m3fn};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::projection::matvec::{self, MatvecGeometry};
use infer_cuda_kernels::tile::runtime::Allocation;

impl CudaDevice {
    /// FP8 channel-scaled weights with F32 activations and accumulation.
    /// # Errors
    /// Rejects incompatible shapes or failed device execution.
    pub fn fp8_matvec(
        &self,
        x: &Allocation<f32>,
        w: &Matrix<f8e4m3fn>,
        scales: &Allocation<f32>,
        tile: LinearTiling,
    ) -> Result<Allocation<f32>> {
        let (rows, columns) = shape(x, w, 1)?;
        if scales.len() != rows {
            return Err(Error::invalid("FP8 channel scales shape"));
        }
        matvec::fp8_matvec(
            &self.runtime(),
            x.buffer(),
            (w.allocation().buffer(), scales.buffer()),
            geometry(rows, columns, tile),
        )
    }

    /// Packed NVFP4 weights with F32 activations and accumulation.
    ///
    /// This reference path dequantizes weights in registers. It does not yet
    /// quantize activations or use block-scaled Tensor Core instructions.
    /// # Errors
    /// Rejects invalid scales, packed shapes or failed device execution.
    pub fn nvfp4_matvec(
        &self,
        x: &Allocation<f32>,
        w: &Matrix<f4e2m1fnx2>,
        scales: &Matrix<f8e4m3fn>,
        global_scale: f32,
        tile: LinearTiling,
    ) -> Result<Allocation<f32>> {
        self.target().require_native_nvfp4()?;
        let (rows, columns) = shape(x, w, 2)?;
        if columns % NVFP4_GROUP_SIZE != 0
            || scales.shape() != [rows, columns / NVFP4_GROUP_SIZE]
            || !global_scale.is_finite()
            || global_scale <= 0.0
        {
            return Err(Error::invalid("invalid NVFP4 scales or alignment"));
        }
        matvec::nvfp4_matvec(
            &self.runtime(),
            x.buffer(),
            (w.allocation().buffer(), scales.allocation().buffer()),
            (global_scale.recip(), NVFP4_GROUP_SIZE),
            geometry(rows, columns, tile),
        )
    }
}

const fn geometry(rows: usize, columns: usize, tile: LinearTiling) -> MatvecGeometry {
    MatvecGeometry {
        rows,
        columns,
        tile_rows: tile.rows(),
        tile_columns: tile.columns(),
    }
}

/// The logical `[rows, columns]` of `w` times `x`, whose stored columns pack `packing` values.
fn shape<T: DType>(x: &Allocation<f32>, w: &Matrix<T>, packing: usize) -> Result<(usize, usize)> {
    let rows = w.rows();
    let columns = w
        .columns()
        .checked_mul(packing)
        .ok_or_else(|| Error::invalid("matrix columns overflow"))?;
    if rows == 0 || columns == 0 || x.len() != columns {
        return Err(Error::invalid("quantized projection dimensions"));
    }
    Ok((rows, columns))
}
