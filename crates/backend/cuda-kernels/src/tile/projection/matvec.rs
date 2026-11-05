//! Eager host-validated projections: weight self-checks and reference paths, not captured graphs.
use super::linear::{dense, fp8, nvfp4};
use crate::contract::buffer::Buffer;
use crate::tile::runtime::{Allocation, Device};
use crate::tile::{backend_error, borrowed};
use cuda_core::{DType, f4e2m1fnx2, f8e4m3fn};
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

/// `w · x` for a row-major `[rows, columns]` weight, complete when this returns; the output is
/// a fresh allocation, zeroed in the same launch sequence.
/// # Errors
/// Rejects operands whose lengths disagree with `geometry`, or a rejected launch.
pub fn dense_matvec<T: DType>(
    device: &Device,
    x: Buffer<'_, f32>,
    w: Buffer<'_, T>,
    geometry: MatvecGeometry,
) -> Result<Allocation<f32>> {
    let MatvecGeometry { rows, columns, .. } = geometry;
    let ordinal = device.ordinal();
    let x = borrowed(x.require("matvec input", columns)?, ordinal, &[columns])?;
    let w = borrowed(
        w.require("matvec weight", rows * columns)?,
        ordinal,
        &[rows, columns],
    )?;
    let block = columns.min(geometry.tile_columns).next_power_of_two();
    let out = api::zeros::<f32>(&[rows]).partition([geometry.tile_rows]);
    let out = dense(out, &x, &w)
        .generics(vec![
            T::DTYPE.as_str().into(),
            geometry.tile_rows.to_string(),
            block.to_string(),
            columns.to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(device.stream())
        .map_err(backend_error)?;
    Allocation::adopt(Arc::new(out))
}

/// FP8 channel-scaled weights with F32 activations and accumulation.
/// # Errors
/// Rejects operands whose lengths disagree with `geometry`, or a rejected launch.
pub fn fp8_matvec(
    device: &Device,
    x: Buffer<'_, f32>,
    (w, scales): (Buffer<'_, f8e4m3fn>, Buffer<'_, f32>),
    geometry: MatvecGeometry,
) -> Result<Allocation<f32>> {
    let MatvecGeometry { rows, columns, .. } = geometry;
    let ordinal = device.ordinal();
    let x = borrowed(x.require("matvec input", columns)?, ordinal, &[columns])?;
    let w = borrowed(
        w.require("FP8 matvec weight", rows * columns)?,
        ordinal,
        &[rows, columns],
    )?;
    let scales = borrowed(
        scales.require("FP8 channel scales", rows)?,
        ordinal,
        &[rows],
    )?;
    let out = api::zeros::<f32>(&[rows]).partition([geometry.tile_rows]);
    let out = fp8(out, &x, &w, &scales)
        .generics(vec![
            geometry.tile_rows.to_string(),
            geometry.tile_columns.to_string(),
            columns.to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(device.stream())
        .map_err(backend_error)?;
    Allocation::adopt(Arc::new(out))
}

/// Packed NVFP4 weights with F32 activations and accumulation; `group` columns share a scale.
/// # Errors
/// Rejects operands whose lengths disagree with `geometry`, or a rejected launch.
pub fn nvfp4_matvec(
    device: &Device,
    x: Buffer<'_, f32>,
    (w, scales): (Buffer<'_, f4e2m1fnx2>, Buffer<'_, f8e4m3fn>),
    (inverse_scale, group): (f32, usize),
    geometry: MatvecGeometry,
) -> Result<Allocation<f32>> {
    let MatvecGeometry { rows, columns, .. } = geometry;
    let ordinal = device.ordinal();
    let (packed, groups) = (columns / FP4_PER_ELEMENT, columns / group);
    let x = borrowed(x.require("matvec input", columns)?, ordinal, &[columns])?;
    let w = borrowed(
        w.require("NVFP4 matvec weight", rows * packed)?,
        ordinal,
        &[rows, packed],
    )?;
    let scales = borrowed(
        scales.require("NVFP4 block scales", rows * groups)?,
        ordinal,
        &[rows, groups],
    )?;
    let out = api::zeros::<f32>(&[rows]).partition([geometry.tile_rows]);
    let out = nvfp4(out, &x, &w, &scales, inverse_scale)
        .generics(vec![
            geometry.tile_rows.to_string(),
            geometry.tile_columns.to_string(),
            columns.to_string(),
            (geometry.tile_columns / FP4_PER_ELEMENT).to_string(),
            (geometry.tile_columns / group).to_string(),
        ])
        .first()
        .unpartition()
        .sync_on(device.stream())
        .map_err(backend_error)?;
    Allocation::adopt(Arc::new(out))
}
