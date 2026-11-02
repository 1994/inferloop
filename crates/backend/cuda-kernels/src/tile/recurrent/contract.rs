//! cuTile implementation of the recurrent prompt family.
use super::{conv_prefill, recurrent, recurrent_chunk, recurrent_prefill, slot_convolution};
use crate::contract::buffer::{Bf16, Buffer};
use crate::contract::kernels::{
    ChunkBuffers, ConvBuffers, DeltaBuffers, GateBuffers, PreparedDeltaBuffers, Recurrent,
};
use crate::contract::plan::{ConvKernel, DeltaKernel, DeltaStepPlan, RecurrentShape};
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

/// Elements of a channel axis in one convolution tile.
const CONV_TILE: usize = 128;
/// Taps of the convolution weight.
const CONV_TAPS: usize = 4;
/// Prompt rows in one forward tile when the chunk divides evenly.
const CONV_ROWS: usize = 16;
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

type ConvFixed<'t> = (&'t Tensor<f32>, &'t Tensor<i32>);

impl TileRecurrent {
    fn conv_step(
        &self,
        recording: &Recording<'_>,
        scope: &Scope,
        buffers: ConvBuffers<'_>,
        (weight, metadata): ConvFixed<'_>,
    ) -> Result<()> {
        let [h0, h1, h2] = buffers.history;
        let mut output = self.flat(recording, buffers.output)?;
        let mut h0 = self.flat(recording, h0)?;
        let mut h1 = self.flat(recording, h1)?;
        let mut h2 = self.flat(recording, h2)?;
        let input = self.flat(recording, buffers.input)?;
        scope
            .record(recurrent::conv4(
                (&mut output).partition([CONV_TILE]),
                (&mut h0).partition([CONV_TILE]),
                (&mut h1).partition([CONV_TILE]),
                (&mut h2).partition([CONV_TILE]),
                &input,
                weight,
                metadata,
            ))
            .map(drop)
            .map_err(backend_error)
    }

    fn conv_lanes(
        &self,
        recording: &Recording<'_>,
        scope: &Scope,
        shape: [usize; 2],
        buffers: ConvBuffers<'_>,
        (weight, metadata): ConvFixed<'_>,
    ) -> Result<()> {
        let [h0, h1, h2] = buffers.history;
        let mut output = self.tensor(recording, buffers.output, "conv output", &shape)?;
        let mut h0 = self.tensor(recording, h0, "conv history", &shape)?;
        let mut h1 = self.tensor(recording, h1, "conv history", &shape)?;
        let mut h2 = self.tensor(recording, h2, "conv history", &shape)?;
        let input = self.tensor(recording, buffers.input, "conv input", &shape)?;
        scope
            .record(recurrent::conv4_lanes(
                (&mut output).partition([1, CONV_TILE]),
                (&mut h0).partition([1, CONV_TILE]),
                (&mut h1).partition([1, CONV_TILE]),
                (&mut h2).partition([1, CONV_TILE]),
                &input,
                weight,
                metadata,
            ))
            .map(drop)
            .map_err(backend_error)
    }

    fn conv_prefill(
        &self,
        recording: &Recording<'_>,
        scope: &Scope,
        shape: [usize; 2],
        buffers: ConvBuffers<'_>,
        (weight, metadata): ConvFixed<'_>,
    ) -> Result<()> {
        let [h0, h1, h2] = buffers.history;
        let mut output = self.tensor(recording, buffers.output, "conv output", &shape)?;
        let mut h0 = self.flat(recording, h0)?;
        let mut h1 = self.flat(recording, h1)?;
        let mut h2 = self.flat(recording, h2)?;
        let input = self.tensor(recording, buffers.input, "conv input", &shape)?;
        if shape[0].is_multiple_of(CONV_ROWS) {
            scope
                .record(conv_prefill::forward_rows(
                    (&mut output).partition([CONV_ROWS, CONV_TILE]),
                    &h0,
                    &h1,
                    &h2,
                    &input,
                    weight,
                    metadata,
                ))
                .map_err(backend_error)?;
        } else {
            scope
                .record(conv_prefill::forward(
                    (&mut output).partition([1, CONV_TILE]),
                    &h0,
                    &h1,
                    &h2,
                    &input,
                    weight,
                    metadata,
                ))
                .map_err(backend_error)?;
        }
        scope
            .record(conv_prefill::commit(
                (&mut h0).partition([CONV_TILE]),
                (&mut h1).partition([CONV_TILE]),
                (&mut h2).partition([CONV_TILE]),
                &input,
                metadata,
            ))
            .map(drop)
            .map_err(backend_error)
    }

    fn conv_verify(
        &self,
        recording: &Recording<'_>,
        scope: &Scope,
        ([slots, lanes, channels], fields, fused): ([usize; 3], usize, bool),
        buffers: ConvBuffers<'_>,
        (weight, metadata): ConvFixed<'_>,
    ) -> Result<()> {
        let [h0, h1, h2] = buffers.history;
        let shape = [slots, lanes, channels];
        let history = [slots, channels];
        let mut output = self.tensor(recording, buffers.output, "conv output", &shape)?;
        let h0 = self.tensor(recording, h0, "conv history", &history)?;
        let h1 = self.tensor(recording, h1, "conv history", &history)?;
        let h2 = self.tensor(recording, h2, "conv history", &history)?;
        let input = self.tensor(recording, buffers.input, "conv input", &shape)?;
        if fused {
            scope
                .record(
                    slot_convolution::verify(
                        (&mut output).partition([1, lanes, CONV_TILE]),
                        &input,
                        &h0,
                        &h1,
                        &h2,
                        weight,
                        metadata,
                    )
                    .generics(vec![channels.to_string(), fields.to_string()]),
                )
                .map(drop)
                .map_err(backend_error)
        } else {
            scope
                .record(
                    slot_convolution::verify_lanes(
                        (&mut output).partition([1, lanes, CONV_TILE]),
                        &input,
                        &h0,
                        &h1,
                        &h2,
                        weight,
                        metadata,
                    )
                    .generics(vec![
                        channels.to_string(),
                        fields.to_string(),
                        lanes.to_string(),
                    ]),
                )
                .map(drop)
                .map_err(backend_error)
        }
    }
}

impl Recurrent for TileRecurrent {
    fn delta_step(
        &self,
        recording: &Recording<'_>,
        plan: &DeltaStepPlan,
        buffers: DeltaBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DeltaStepPlan {
            key_heads,
            value_heads,
            dim,
            block,
            lanes,
        } = *plan;
        let blocks = lanes.unwrap_or(1) * value_heads;
        let mut output = self.tensor(recording, buffers.output, "output", &[blocks, 1, dim])?;
        let mut state = self.tensor(recording, buffers.state, "state", &[blocks, dim, dim])?;
        let qkv = self.flat(recording, buffers.qkv)?;
        let beta = self.flat(recording, buffers.beta)?;
        let alpha = self.flat(recording, buffers.alpha)?;
        let decay = self.flat(recording, buffers.decay)?;
        let bias = self.flat(recording, buffers.bias)?;
        let metadata = self.flat(recording, buffers.metadata)?;
        let generics = vec![
            key_heads.to_string(),
            value_heads.to_string(),
            dim.to_string(),
            block.to_string(),
        ];
        let recorded = if lanes.is_some() {
            scope.record(
                recurrent::delta_lanes(
                    (&mut output).partition([1, 1, block]),
                    (&mut state).partition([1, dim, block]),
                    &qkv,
                    &beta,
                    &alpha,
                    &decay,
                    &bias,
                    &metadata,
                )
                .generics(generics),
            )
        } else {
            scope.record(
                recurrent::delta(
                    (&mut output).partition([1, 1, block]),
                    (&mut state).partition([1, dim, block]),
                    &qkv,
                    &beta,
                    &alpha,
                    &decay,
                    &bias,
                    &metadata,
                )
                .generics(generics),
            )
        };
        recorded.map(drop).map_err(backend_error)
    }

    fn conv(
        &self,
        recording: &Recording<'_>,
        kernel: ConvKernel,
        channels: usize,
        buffers: ConvBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let weight = self.tensor(
            recording,
            buffers.weight,
            "conv weight",
            &[channels, CONV_TAPS],
        )?;
        let metadata = self.flat(recording, buffers.metadata)?;
        let fixed = (&weight, &metadata);
        match kernel {
            ConvKernel::Step => self.conv_step(recording, scope, buffers, fixed),
            ConvKernel::Lanes { lanes } => {
                self.conv_lanes(recording, scope, [lanes, channels], buffers, fixed)
            }
            ConvKernel::Prefill { lanes } => {
                self.conv_prefill(recording, scope, [lanes, channels], buffers, fixed)
            }
            ConvKernel::Verify {
                slots,
                lanes,
                fields,
                fused,
            } => self.conv_verify(
                recording,
                scope,
                ([slots, lanes, channels], fields, fused),
                buffers,
                fixed,
            ),
        }
    }

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
