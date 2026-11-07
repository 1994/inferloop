//! Device work outside the kernel families: eager projections and probes, parameter packing, and
//! metadata writes before a graph's replay.
//!
//! Each of these is a cuTile kernel launch in the cuTile build. Until an AOT artifact provides
//! one, it reports it is not provided, like the families do.
use super::runtime::{Allocation, DType, Device, Graph, f4e2m1fnx2, f8e4m3fn};
use crate::contract::{Buffer, MatvecGeometry, ProbeGeometry};
use infer_core::{Error, Result};

/// Fields of one control-metadata row, as the cuTile build writes it.
const METADATA_FIELDS: usize = 4;
/// Axes of a rotary position: time, height and width.
const ROTARY_AXES: usize = 3;

fn unsupported<T>(operation: &str) -> Result<T> {
    Err(Error::unsupported(format!(
        "{operation} is not provided by this kernel source"
    )))
}

/// `w · x` for a row-major `[rows, columns]` weight, as a fresh allocation.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn dense_matvec<T: DType>(
    _device: &Device,
    _x: Buffer<'_, f32>,
    _w: Buffer<'_, T>,
    _geometry: MatvecGeometry,
) -> Result<Allocation<f32>> {
    unsupported("dense_matvec")
}

/// FP8 channel-scaled weights with F32 activations and accumulation.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn fp8_matvec(
    _device: &Device,
    _x: Buffer<'_, f32>,
    _weights: (Buffer<'_, f8e4m3fn>, Buffer<'_, f32>),
    _geometry: MatvecGeometry,
) -> Result<Allocation<f32>> {
    unsupported("fp8_matvec")
}

/// Packed NVFP4 weights with F32 activations and accumulation; `group` columns share a scale.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn nvfp4_matvec(
    _device: &Device,
    _x: Buffer<'_, f32>,
    _weights: (Buffer<'_, f4e2m1fnx2>, Buffer<'_, f8e4m3fn>),
    _scaling: (f32, usize),
    _geometry: MatvecGeometry,
) -> Result<Allocation<f32>> {
    unsupported("nvfp4_matvec")
}

/// Fill a probe input and quantize it into FP4 codes and scales.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn initialize_fp4_probe(
    _device: &Device,
    _input: Buffer<'_, f32>,
    _quantized: (Buffer<'_, f4e2m1fnx2>, Buffer<'_, f8e4m3fn>),
    _scale: f32,
    _geometry: ProbeGeometry,
) -> Result<()> {
    unsupported("initialize_fp4_probe")
}

/// Pack the immutable decay rate and bias of a delta node into one tensor.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn pack_parameters(
    _device: &Device,
    _parameters: Buffer<'_, f32>,
    _rate: Buffer<'_, f32>,
    _bias: Buffer<'_, f32>,
) -> Result<()> {
    unsupported("pack_parameters")
}

/// Write control metadata before the graph's next replay, on the graph's stream.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn update_metadata(
    _graph: &Graph,
    _output: Buffer<'_, i32>,
    _values: [i32; METADATA_FIELDS],
) -> Result<()> {
    unsupported("update_metadata")
}

/// Write control metadata and independent rotary coordinates on the graph's stream.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn update_rotary(
    _graph: &Graph,
    _output: Buffer<'_, i32>,
    _values: [i32; METADATA_FIELDS],
    _position: [usize; ROTARY_AXES],
) -> Result<()> {
    unsupported("update_rotary")
}

/// Load the rotary metadata write before any submission.
/// # Errors
/// Always: no AOT artifact provides it yet.
pub fn load_rotary(_device: &Device, _output: Buffer<'_, i32>) -> Result<()> {
    unsupported("load_rotary")
}
