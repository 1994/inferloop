//! Fill a vendor FP4 probe's activation and its packed codes with real values before measuring.
use super::entries::probe_input;
use super::nvfp4_gemm;
use crate::contract::buffer::Buffer;
use crate::tile::runtime::{Device, Graph};
use crate::tile::{backend_error, borrowed};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::{Error, Result};

/// Rows and tile widths of the probe.
#[derive(Clone, Copy, Debug)]
pub struct ProbeGeometry {
    pub rows: usize,
    pub columns: usize,
    pub code_tile: usize,
    pub scale_tile: usize,
}

/// Packed FP4 values per stored element.
const FP4_PER_ELEMENT: usize = 2;

/// Fill `input`, `[rows, columns]` floats, then quantize it into the `q` codes and `qs` scales,
/// and wait for both. Every buffer must stay alive until this returns.
/// # Errors
/// Rejects buffers that do not divide into the geometry's rows, and a rejected capture or launch.
pub fn initialize(
    device: &Device,
    input: Buffer<'_, f32>,
    (q, qs): (Buffer<'_, f4e2m1fnx2>, Buffer<'_, f8e4m3fn>),
    scale: f32,
    geometry: ProbeGeometry,
) -> Result<()> {
    let ProbeGeometry {
        rows,
        columns,
        code_tile,
        scale_tile,
    } = geometry;
    let half = columns / FP4_PER_ELEMENT;
    let matrix = |len: usize, width: usize| {
        if width == 0 || !len.is_multiple_of(width) {
            return Err(Error::invalid("FP4 probe operand rows"));
        }
        Ok([len / width, width])
    };
    let ordinal = device.ordinal();
    let mut input = borrowed(input, ordinal, &matrix(input.len(), columns)?)?;
    let [code_rows, _] = matrix(q.len(), half)?;
    let mut q = borrowed(q, ordinal, &[code_rows, half])?;
    if code_rows == 0 || !qs.len().is_multiple_of(code_rows) {
        return Err(Error::invalid("FP4 probe scale rows"));
    }
    let mut qs = borrowed(qs, ordinal, &[code_rows, qs.len() / code_rows])?;
    let stream = device.stream();
    let graph = Graph::scope(stream, |scope| {
        scope.record(
            probe_input((&mut input).partition([rows, code_tile * FP4_PER_ELEMENT]))
                .generics(vec![columns.to_string()]),
        )?;
        scope.record(
            nvfp4_gemm::quantize_rows(
                (&mut q).partition([rows, code_tile]),
                (&mut qs).partition([rows, scale_tile]),
                &input,
                scale,
            )
            .generics(vec![columns.to_string()]),
        )?;
        Ok(())
    })
    .map_err(backend_error)?;
    graph.launch().sync_on(stream).map_err(backend_error)?;
    Ok(())
}
