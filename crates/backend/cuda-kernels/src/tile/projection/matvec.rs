//! Eager host-validated projections: weight self-checks and reference paths, not captured graphs.
use super::linear::{dense, fp8, nvfp4};
use crate::tile::backend_error;
use cuda_core::{DType, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::Result;
use std::sync::Arc;

/// Packed FP4 values per stored element.
const FP4_PER_ELEMENT: usize = 2;
/// Tile geometry of one eager projection.
#[derive(Clone, Copy, Debug)]
pub struct MatvecGeometry {
    pub rows: usize,
    pub columns: usize,
    pub tile_rows: usize,
    pub tile_columns: usize,
}

/// # Errors
/// Reports a rejected launch.
pub fn dense_matvec<T: DType>(
    stream: &Arc<Stream>,
    x: Arc<Tensor<f32>>,
    w: Arc<Tensor<T>>,
    geometry: MatvecGeometry,
) -> Result<Arc<Tensor<f32>>> {
    let block = geometry
        .columns
        .min(geometry.tile_columns)
        .next_power_of_two();
    let out = api::zeros::<f32>(&[geometry.rows]).partition([geometry.tile_rows]);
    dense(out, x, w)
        .generics(vec![
            T::DTYPE.as_str().into(),
            geometry.tile_rows.to_string(),
            block.to_string(),
            geometry.columns.to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(stream)
        .map(Arc::new)
        .map_err(backend_error)
}

/// FP8 channel-scaled weights with F32 activations and accumulation.
/// # Errors
/// Reports a rejected launch.
pub fn fp8_matvec(
    stream: &Arc<Stream>,
    x: Arc<Tensor<f32>>,
    w: Arc<Tensor<f8e4m3fn>>,
    scales: Arc<Tensor<f32>>,
    geometry: MatvecGeometry,
) -> Result<Arc<Tensor<f32>>> {
    let out = api::zeros::<f32>(&[geometry.rows]).partition([geometry.tile_rows]);
    fp8(out, x, w, scales)
        .generics(vec![
            geometry.tile_rows.to_string(),
            geometry.tile_columns.to_string(),
            geometry.columns.to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(stream)
        .map(Arc::new)
        .map_err(backend_error)
}

/// Packed NVFP4 weights with F32 activations and accumulation; `group` columns share a scale.
/// # Errors
/// Reports a rejected launch.
pub fn nvfp4_matvec(
    stream: &Arc<Stream>,
    x: Arc<Tensor<f32>>,
    (w, scales): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>),
    (inverse_scale, group): (f32, usize),
    geometry: MatvecGeometry,
) -> Result<Arc<Tensor<f32>>> {
    let out = api::zeros::<f32>(&[geometry.rows]).partition([geometry.tile_rows]);
    nvfp4(out, x, w, scales, inverse_scale)
        .generics(vec![
            geometry.tile_rows.to_string(),
            geometry.tile_columns.to_string(),
            geometry.columns.to_string(),
            (geometry.tile_columns / FP4_PER_ELEMENT).to_string(),
            (geometry.tile_columns / group).to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(stream)
        .map(Arc::new)
        .map_err(backend_error)
}
