//! cuTile implementation of the recurrent prompt family.
use super::{recurrent_chunk, recurrent_prefill};
use crate::contract::buffer::{Bf16, Buffer};
use crate::contract::kernels::{
    ChunkBuffers, DeltaBuffers, GateBuffers, PreparedDeltaBuffers, Recurrent,
};
use crate::contract::plan::{DeltaKernel, RecurrentShape};
use crate::contract::recording::Recording;
use crate::tile::projection::retyped_bf16;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::DType;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::Result;

/// Value columns in the general scan tile.
pub const SCAN_COLUMNS: usize = 32;
/// Wider tiles amortize factor loads for this measured value-head geometry.
pub const WIDE_SCAN_COLUMNS: usize = 64;
/// Value-head count whose prompt widths benefit from the wider tile.
pub const WIDE_SCAN_VALUE_HEADS: usize = 48;

/// Recurrent kernels compiled with cuTile.
pub struct TileRecurrent {
    ordinal: usize,
}

impl TileRecurrent {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    fn tensor<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
        name: &str,
        shape: &[usize],
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        borrowed(
            buffer.require(name, shape.iter().product())?,
            self.ordinal,
            shape,
        )
    }

    fn halves(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, Bf16>,
        name: &str,
        shape: &[usize],
    ) -> Result<Tensor<bf16>> {
        self.tensor(recording, retyped_bf16(buffer), name, shape)
    }

    fn flat<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        borrowed(buffer, self.ordinal, &[buffer.len()])
    }
}

impl Recurrent for TileRecurrent {
    fn chunk(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        chunk: usize,
        buffers: ChunkBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let RecurrentShape {
            key_heads: kh,
            value_heads: vh,
            dim,
            lanes,
        } = shape;
        let mut queries = self.halves(recording, buffers.queries, "queries", &[kh, lanes, dim])?;
        let mut keys = self.halves(recording, buffers.keys, "keys", &[kh, lanes, dim])?;
        let mut u = self.halves(recording, buffers.u, "u", &[vh, lanes, dim])?;
        let mut w = self.halves(recording, buffers.w, "w", &[vh, lanes, dim])?;
        let mut p = self.halves(recording, buffers.p, "p", &[vh, lanes, chunk])?;
        let mut cumulative =
            self.tensor(recording, buffers.cumulative, "cumulative", &[vh, lanes])?;
        let qkv = self.tensor(recording, buffers.qkv, "qkv", &[lanes, 2 * kh + vh, dim])?;
        let betas = self.tensor(recording, buffers.betas, "betas", &[lanes, vh])?;
        let decays = self.tensor(recording, buffers.decays, "decays", &[lanes, vh])?;
        let mut state = self.tensor(recording, buffers.state, "state", &[vh, dim, dim])?;
        let mut output = self.tensor(recording, buffers.output, "output", &[vh, lanes, dim])?;
        let metadata = self.flat(recording, buffers.metadata)?;
        scope
            .record(
                recurrent_chunk::chunk_normalize(
                    (&mut queries).partition([1, chunk, dim]),
                    (&mut keys).partition([1, chunk, dim]),
                    &qkv,
                    &metadata,
                )
                .generics(vec![kh.to_string(), dim.to_string()]),
            )
            .map_err(backend_error)?;
        scope
            .record(
                recurrent_chunk::chunk_prepare(
                    (&mut u).partition([1, chunk, dim]),
                    (&mut w).partition([1, chunk, dim]),
                    (&mut p).partition([1, chunk, chunk]),
                    (&mut cumulative).partition([1, chunk]),
                    &qkv,
                    &queries,
                    &keys,
                    &betas,
                    &decays,
                    &metadata,
                )
                .generics(vec![kh.to_string(), vh.to_string(), dim.to_string()]),
            )
            .map_err(backend_error)?;
        let generics = vec![kh.to_string(), vh.to_string(), lanes.to_string()];
        let recorded = if vh == WIDE_SCAN_VALUE_HEADS {
            scope.record(
                recurrent_chunk::chunk_scan_wide(
                    (&mut state).partition([1, dim, WIDE_SCAN_COLUMNS]),
                    (&mut output).partition([1, lanes, WIDE_SCAN_COLUMNS]),
                    &u,
                    &w,
                    &queries,
                    &keys,
                    &p,
                    &cumulative,
                    &metadata,
                )
                .generics(generics),
            )
        } else {
            scope.record(
                recurrent_chunk::chunk_scan(
                    (&mut state).partition([1, dim, SCAN_COLUMNS]),
                    (&mut output).partition([1, lanes, SCAN_COLUMNS]),
                    &u,
                    &w,
                    &queries,
                    &keys,
                    &p,
                    &cumulative,
                    &metadata,
                )
                .generics(generics),
            )
        };
        recorded.map(drop).map_err(backend_error)
    }

    fn normalize(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        normalized: Buffer<'_, f32>,
        qkv: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let RecurrentShape {
            key_heads: kh,
            value_heads: vh,
            dim,
            lanes,
        } = shape;
        let mut normalized =
            self.tensor(recording, normalized, "normalized", &[lanes, 2 * kh, dim])?;
        let qkv = self.tensor(recording, qkv, "qkv", &[lanes, 2 * kh + vh, dim])?;
        scope
            .record(
                recurrent_prefill::normalize_shared((&mut normalized).partition([1, 1, dim]), &qkv)
                    .generics(vec![kh.to_string(), dim.to_string()]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn gates(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        buffers: GateBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let (lanes, vh) = (shape.lanes, shape.value_heads);
        let mut betas = self.tensor(recording, buffers.betas, "betas", &[lanes, vh])?;
        let mut decays = self.tensor(recording, buffers.decays, "decays", &[lanes, vh])?;
        let beta = self.tensor(recording, buffers.beta, "beta", &[lanes, vh])?;
        let alpha = self.tensor(recording, buffers.alpha, "alpha", &[lanes, vh])?;
        let decay = self.tensor(recording, buffers.decay, "decay", &[vh])?;
        let bias = self.tensor(recording, buffers.bias, "bias", &[vh])?;
        scope
            .record(recurrent_prefill::gates_shared(
                (&mut betas).partition([1, 1]),
                (&mut decays).partition([1, 1]),
                &beta,
                &alpha,
                &decay,
                &bias,
            ))
            .map(drop)
            .map_err(backend_error)
    }

    fn delta_prepared(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        block: usize,
        buffers: PreparedDeltaBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let RecurrentShape {
            key_heads: kh,
            value_heads: vh,
            dim,
            lanes,
        } = shape;
        let mut state = self.tensor(recording, buffers.state, "state", &[vh, dim, dim])?;
        let mut output = self.tensor(recording, buffers.output, "output", &[vh, lanes, dim])?;
        let qkv = self.tensor(recording, buffers.qkv, "qkv", &[lanes, 2 * kh + vh, dim])?;
        let normalized = self.tensor(
            recording,
            buffers.normalized,
            "normalized",
            &[lanes, 2 * kh, dim],
        )?;
        let betas = self.tensor(recording, buffers.betas, "betas", &[lanes, vh])?;
        let decays = self.tensor(recording, buffers.decays, "decays", &[lanes, vh])?;
        let metadata = self.flat(recording, buffers.metadata)?;
        scope
            .record(
                recurrent_prefill::delta_precomputed(
                    (&mut state).partition([1, dim, block]),
                    (&mut output).partition([1, lanes, block]),
                    &qkv,
                    &normalized,
                    &betas,
                    &decays,
                    &metadata,
                )
                .generics(vec![
                    kh.to_string(),
                    vh.to_string(),
                    dim.to_string(),
                    lanes.to_string(),
                    block.to_string(),
                ]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn delta(
        &self,
        recording: &Recording<'_>,
        kernel: DeltaKernel,
        shape: RecurrentShape,
        block: usize,
        buffers: DeltaBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let RecurrentShape {
            key_heads: kh,
            value_heads: vh,
            dim,
            lanes,
        } = shape;
        let mut state = self.tensor(recording, buffers.state, "state", &[vh, dim, dim])?;
        let mut output = self.tensor(recording, buffers.output, "output", &[vh, lanes, dim])?;
        let qkv = self.tensor(recording, buffers.qkv, "qkv", &[lanes, 2 * kh + vh, dim])?;
        let beta = self.tensor(recording, buffers.beta, "beta", &[lanes, vh])?;
        let alpha = self.tensor(recording, buffers.alpha, "alpha", &[lanes, vh])?;
        let decay = self.tensor(recording, buffers.decay, "decay", &[vh])?;
        let bias = self.tensor(recording, buffers.bias, "bias", &[vh])?;
        let metadata = self.flat(recording, buffers.metadata)?;
        let generics = vec![
            kh.to_string(),
            vh.to_string(),
            dim.to_string(),
            lanes.to_string(),
            block.to_string(),
        ];
        let recorded = match kernel {
            DeltaKernel::Scan => scope.record(
                recurrent_prefill::delta(
                    (&mut state).partition([1, dim, block]),
                    (&mut output).partition([1, lanes, block]),
                    &qkv,
                    &beta,
                    &alpha,
                    &decay,
                    &bias,
                    &metadata,
                )
                .generics(generics),
            ),
            DeltaKernel::Wy => scope.record(
                recurrent_prefill::delta_wy(
                    (&mut state).partition([1, dim, block]),
                    (&mut output).partition([1, lanes, block]),
                    &qkv,
                    &beta,
                    &alpha,
                    &decay,
                    &bias,
                    &metadata,
                )
                .generics(generics),
            ),
        };
        recorded.map(drop).map_err(backend_error)
    }

    fn transpose(
        &self,
        recording: &Recording<'_>,
        shape: RecurrentShape,
        output: Buffer<'_, f32>,
        scratch: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let RecurrentShape {
            value_heads: vh,
            dim,
            lanes,
            ..
        } = shape;
        let mut output = self.tensor(recording, output, "output", &[lanes, vh, dim])?;
        let scratch = self.tensor(recording, scratch, "scratch", &[vh, lanes, dim])?;
        scope
            .record(
                recurrent_prefill::transpose((&mut output).partition([1, 1, dim]), &scratch)
                    .generics(vec![dim.to_string()]),
            )
            .map(drop)
            .map_err(backend_error)
    }
}
