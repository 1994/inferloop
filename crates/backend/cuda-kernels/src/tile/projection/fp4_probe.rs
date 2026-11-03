//! Fill a vendor FP4 probe's activation and its packed codes with real values before measuring.
use super::entries::probe_input;
use super::nvfp4_gemm;
use crate::tile::backend_error;
use crate::tile::runtime::Graph;
use cuda_core::{Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use infer_core::Result;
use std::sync::Arc;

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

/// # Errors
/// Reports a rejected capture or launch.
pub fn initialize(
    stream: &Arc<Stream>,
    input: &mut Tensor<f32>,
    (q, qs): (&mut Tensor<f4e2m1fnx2>, &mut Tensor<f8e4m3fn>),
    scale: f32,
    geometry: ProbeGeometry,
) -> Result<()> {
    let ProbeGeometry {
        rows,
        columns,
        code_tile,
        scale_tile,
    } = geometry;
    let graph = Graph::scope(stream, |scope| {
        scope.record(
            probe_input((&mut *input).partition([rows, code_tile * FP4_PER_ELEMENT]))
                .generics(vec![columns.to_string()]),
        )?;
        scope.record(
            nvfp4_gemm::quantize_rows(
                (&mut *q).partition([rows, code_tile]),
                (&mut *qs).partition([rows, scale_tile]),
                &*input,
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
