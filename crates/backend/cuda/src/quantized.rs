//! Quantized storage launch adapters; precision policy stays explicit.
use crate::constants::NVFP4_GROUP_SIZE;
use crate::{device::CudaDevice, strategy::LinearTiling};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::projection::matvec::{self, MatvecGeometry};
use std::sync::Arc;

impl CudaDevice {
    /// FP8 channel-scaled weights with F32 activations and accumulation.
    /// # Errors
    /// Rejects incompatible tensor shapes or failed device execution.
    pub fn fp8_matvec(
        &self,
        x: Arc<Tensor<f32>>,
        w: Arc<Tensor<f8e4m3fn>>,
        scales: Arc<Tensor<f32>>,
        tile: LinearTiling,
    ) -> Result<Arc<Tensor<f32>>> {
        let (rows, columns) = shape(&x, &w, 1)?;
        if !matches_shape(scales.shape(), &[rows]) {
            return Err(Error::invalid("FP8 channel scales shape"));
        }
        matvec::fp8_matvec(&self.stream, x, w, scales, geometry(rows, columns, tile))
    }

    /// Packed NVFP4 weights with F32 activations and accumulation.
    ///
    /// This reference path dequantizes weights in registers. It does not yet
    /// quantize activations or use block-scaled Tensor Core instructions.
    /// # Errors
    /// Rejects invalid scales, packed shapes or failed device execution.
    pub fn nvfp4_matvec(
        &self,
        x: Arc<Tensor<f32>>,
        w: Arc<Tensor<f4e2m1fnx2>>,
        scales: Arc<Tensor<f8e4m3fn>>,
        global_scale: f32,
        tile: LinearTiling,
    ) -> Result<Arc<Tensor<f32>>> {
        self.target().require_native_nvfp4()?;
        let (rows, columns) = shape(&x, &w, 2)?;
        if columns % NVFP4_GROUP_SIZE != 0
            || !matches_shape(scales.shape(), &[rows, columns / NVFP4_GROUP_SIZE])
            || !global_scale.is_finite()
            || global_scale <= 0.0
        {
            return Err(Error::invalid("invalid NVFP4 scales or alignment"));
        }
        matvec::nvfp4_matvec(
            &self.stream,
            x,
            (w, scales),
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

fn shape<T: DType>(x: &Tensor<f32>, w: &Tensor<T>, packing: usize) -> Result<(usize, usize)> {
    let [rows, packed_columns] = w.shape() else {
        return Err(Error::invalid("quantized matrix rank"));
    };
    let rows = usize::try_from(*rows).map_err(|_| Error::invalid("matrix rows"))?;
    let columns = usize::try_from(*packed_columns)
        .ok()
        .and_then(|n| n.checked_mul(packing))
        .ok_or_else(|| Error::invalid("matrix columns overflow"))?;
    if rows == 0 || columns == 0 || !matches_shape(x.shape(), &[columns]) {
        return Err(Error::invalid("quantized projection dimensions"));
    }
    Ok((rows, columns))
}

fn matches_shape(actual: &[i32], expected: &[usize]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| usize::try_from(*a).ok() == Some(*e))
}
