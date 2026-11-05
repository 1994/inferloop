//! cuTile implementation of fused and pool-wide recurrent verification.
use super::slot_recurrent as fused;
use crate::contract::buffer::Buffer;
use crate::contract::kernels::{
    CONV_HISTORY, FusedConvBuffers, FusedDeltaBuffers, LaneBuffers, PoolDeltaBuffers, RecurrentPool,
};
use crate::contract::plan::{DeltaStepPlan, PoolConvPlan, PoolDeltaPlan};
use crate::contract::recording::Recording;
use crate::tile::runtime::Device;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::DType;
use cutile::prelude::*;
use infer_core::Result;

/// Elements of a channel axis in one convolution tile.
const CONV_TILE: usize = 128;
/// Taps of the convolution weight.
const CONV_TAPS: usize = 4;
/// Candidates the specialized pool kernels cover.
const FUSED_CANDIDATES: usize = 3;

/// Pool recurrent kernels compiled with cuTile.
pub struct TileRecurrentPool {
    ordinal: usize,
}

impl TileRecurrentPool {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    fn flat<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        borrowed(buffer, self.ordinal, &[buffer.len()])
    }

    fn shaped<T: DType>(
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

    fn lanes<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffers: LaneBuffers<'_, T>,
    ) -> Result<[Tensor<T>; FUSED_CANDIDATES]> {
        let [a, b, c] = buffers;
        Ok([
            self.flat(recording, a)?,
            self.flat(recording, b)?,
            self.flat(recording, c)?,
        ])
    }

    fn history(
        &self,
        recording: &Recording<'_>,
        buffers: [Buffer<'_, f32>; CONV_HISTORY],
    ) -> Result<[Tensor<f32>; CONV_HISTORY]> {
        self.lanes(recording, buffers)
    }
}

impl RecurrentPool for TileRecurrentPool {
    fn conv_verify_fused(
        &self,
        recording: &Recording<'_>,
        channels: usize,
        outputs: LaneBuffers<'_, f32>,
        buffers: FusedConvBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let [mut o0, mut o1, mut o2] = self.lanes(recording, outputs)?;
        let [h0, h1, h2] = self.history(recording, buffers.history)?;
        let [x0, x1, x2] = self.lanes(recording, buffers.inputs)?;
        let weight = self.shaped(
            recording,
            buffers.weight,
            "conv weight",
            &[channels, CONV_TAPS],
        )?;
        let [m0, m1, m2] = self.lanes(recording, buffers.metadata)?;
        scope
            .record(fused::conv_verify3(
                (&mut o0).partition([CONV_TILE]),
                (&mut o1).partition([CONV_TILE]),
                (&mut o2).partition([CONV_TILE]),
                &h0,
                &h1,
                &h2,
                &x0,
                &x1,
                &x2,
                &weight,
                &m0,
                &m1,
                &m2,
            ))
            .map(drop)
            .map_err(backend_error)
    }

    fn conv_commit_fused(
        &self,
        recording: &Recording<'_>,
        count: i32,
        history: [Buffer<'_, f32>; CONV_HISTORY],
        inputs: LaneBuffers<'_, f32>,
        metadata: LaneBuffers<'_, i32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let [mut h0, mut h1, mut h2] = self.history(recording, history)?;
        let [x0, x1, x2] = self.lanes(recording, inputs)?;
        let [m0, m1, m2] = self.lanes(recording, metadata)?;
        scope
            .record(fused::conv_commit3(
                (&mut h0).partition([CONV_TILE]),
                (&mut h1).partition([CONV_TILE]),
                (&mut h2).partition([CONV_TILE]),
                &x0,
                &x1,
                &x2,
                &m0,
                &m1,
                &m2,
                count,
            ))
            .map(drop)
            .map_err(backend_error)
    }

    fn delta_verify_fused(
        &self,
        recording: &Recording<'_>,
        plan: &DeltaStepPlan,
        outputs: LaneBuffers<'_, f32>,
        buffers: FusedDeltaBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DeltaStepPlan {
            key_heads,
            value_heads,
            dim,
            block,
            ..
        } = *plan;
        let shape = [value_heads, 1, dim];
        let [o0, o1, o2] = outputs;
        let mut o0 = self.shaped(recording, o0, "verify output", &shape)?;
        let mut o1 = self.shaped(recording, o1, "verify output", &shape)?;
        let mut o2 = self.shaped(recording, o2, "verify output", &shape)?;
        let state = self.shaped(recording, buffers.state, "state", &[value_heads, dim, dim])?;
        let [q0, q1, q2] = self.lanes(recording, buffers.qkv)?;
        let [b0, b1, b2] = self.lanes(recording, buffers.beta)?;
        let [a0, a1, a2] = self.lanes(recording, buffers.alpha)?;
        let decay = self.flat(recording, buffers.decay)?;
        let bias = self.flat(recording, buffers.bias)?;
        let [m0, m1, m2] = self.lanes(recording, buffers.metadata)?;
        scope
            .record(
                fused::delta_verify3(
                    (&mut o0).partition([1, 1, block]),
                    (&mut o1).partition([1, 1, block]),
                    (&mut o2).partition([1, 1, block]),
                    &state,
                    &q0,
                    &q1,
                    &q2,
                    &b0,
                    &b1,
                    &b2,
                    &a0,
                    &a1,
                    &a2,
                    &decay,
                    &bias,
                    &m0,
                    &m1,
                    &m2,
                )
                .generics(vec![
                    key_heads.to_string(),
                    value_heads.to_string(),
                    dim.to_string(),
                    block.to_string(),
                ]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn delta_commit_fused(
        &self,
        recording: &Recording<'_>,
        plan: &DeltaStepPlan,
        count: i32,
        buffers: FusedDeltaBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DeltaStepPlan {
            key_heads,
            value_heads,
            dim,
            block,
            ..
        } = *plan;
        let mut state = self.shaped(recording, buffers.state, "state", &[value_heads, dim, dim])?;
        let [q0, q1, q2] = self.lanes(recording, buffers.qkv)?;
        let [b0, b1, b2] = self.lanes(recording, buffers.beta)?;
        let [a0, a1, a2] = self.lanes(recording, buffers.alpha)?;
        let decay = self.flat(recording, buffers.decay)?;
        let bias = self.flat(recording, buffers.bias)?;
        let [m0, m1, m2] = self.lanes(recording, buffers.metadata)?;
        scope
            .record(
                fused::delta_commit3(
                    (&mut state).partition([1, dim, block]),
                    &q0,
                    &q1,
                    &q2,
                    &b0,
                    &b1,
                    &b2,
                    &a0,
                    &a1,
                    &a2,
                    &decay,
                    &bias,
                    &m0,
                    &m1,
                    &m2,
                    count,
                )
                .generics(vec![
                    key_heads.to_string(),
                    value_heads.to_string(),
                    dim.to_string(),
                    block.to_string(),
                ]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn delta_verify_pool(
        &self,
        recording: &Recording<'_>,
        plan: &PoolDeltaPlan,
        output: Buffer<'_, f32>,
        buffers: PoolDeltaBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PoolDeltaPlan {
            key_heads,
            value_heads,
            dim,
            block,
            slots,
            lanes,
            fields,
            fused: specialized,
        } = *plan;
        let mut output = self.shaped(
            recording,
            output,
            "verify output",
            &[slots, lanes, value_heads * dim],
        )?;
        let state = self.shaped(
            recording,
            buffers.state,
            "state",
            &[slots * value_heads, dim, dim],
        )?;
        let qkv = self.flat(recording, buffers.qkv)?;
        let beta = self.flat(recording, buffers.beta)?;
        let alpha = self.flat(recording, buffers.alpha)?;
        let parameters = self.flat(recording, buffers.parameters)?;
        let metadata = self.flat(recording, buffers.metadata)?;
        let mut generics = vec![
            key_heads.to_string(),
            value_heads.to_string(),
            dim.to_string(),
            block.to_string(),
            fields.to_string(),
        ];
        if specialized {
            scope
                .record(
                    fused::delta_verify_batch(
                        (&mut output).partition([1, FUSED_CANDIDATES, block]),
                        &state,
                        &qkv,
                        &beta,
                        &alpha,
                        &parameters,
                        &metadata,
                    )
                    .generics(generics),
                )
                .map(drop)
                .map_err(backend_error)
        } else {
            generics.push(lanes.to_string());
            scope
                .record(
                    fused::delta_verify_lanes(
                        (&mut output).partition([1, lanes, block]),
                        &state,
                        &qkv,
                        &beta,
                        &alpha,
                        &parameters,
                        &metadata,
                    )
                    .generics(generics),
                )
                .map(drop)
                .map_err(backend_error)
        }
    }

    fn delta_commit_pool(
        &self,
        recording: &Recording<'_>,
        plan: &PoolDeltaPlan,
        counts: Buffer<'_, i32>,
        buffers: PoolDeltaBuffers<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PoolDeltaPlan {
            key_heads,
            value_heads,
            dim,
            block,
            slots,
            lanes,
            fields,
            fused: specialized,
        } = *plan;
        // The slot axis stays first: a block addresses one slot's one head.
        let mut state = self.shaped(
            recording,
            buffers.state,
            "state",
            &[slots, value_heads * dim, dim],
        )?;
        let qkv = self.flat(recording, buffers.qkv)?;
        let beta = self.flat(recording, buffers.beta)?;
        let alpha = self.flat(recording, buffers.alpha)?;
        let parameters = self.flat(recording, buffers.parameters)?;
        let metadata = self.flat(recording, buffers.metadata)?;
        let counts = self.flat(recording, counts)?;
        let mut generics = vec![
            key_heads.to_string(),
            value_heads.to_string(),
            dim.to_string(),
            block.to_string(),
            fields.to_string(),
        ];
        if specialized {
            scope
                .record(
                    fused::delta_commit_batch(
                        (&mut state).partition([1, dim, block]),
                        &qkv,
                        &beta,
                        &alpha,
                        &parameters,
                        &metadata,
                        &counts,
                    )
                    .generics(generics),
                )
                .map(drop)
                .map_err(backend_error)
        } else {
            generics.push(lanes.to_string());
            scope
                .record(
                    fused::delta_commit_lanes(
                        (&mut state).partition([1, dim, block]),
                        &qkv,
                        &beta,
                        &alpha,
                        &parameters,
                        &metadata,
                        &counts,
                    )
                    .generics(generics),
                )
                .map(drop)
                .map_err(backend_error)
        }
    }

    fn conv_commit_pool(
        &self,
        recording: &Recording<'_>,
        plan: &PoolConvPlan,
        history: [Buffer<'_, f32>; CONV_HISTORY],
        rows: Buffer<'_, f32>,
        metadata: Buffer<'_, i32>,
        counts: Buffer<'_, i32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PoolConvPlan {
            channels,
            lanes,
            fields,
            fused: specialized,
        } = *plan;
        let [h0, h1, h2] = history;
        let slots = h0.len() / channels.max(1);
        let shape = [slots, channels];
        let mut h0 = self.shaped(recording, h0, "conv history", &shape)?;
        let mut h1 = self.shaped(recording, h1, "conv history", &shape)?;
        let mut h2 = self.shaped(recording, h2, "conv history", &shape)?;
        let rows = self.shaped(recording, rows, "conv rows", &[slots * lanes, channels])?;
        let metadata = self.flat(recording, metadata)?;
        let counts = self.flat(recording, counts)?;
        if specialized {
            scope
                .record(
                    fused::conv_commit_batch(
                        (&mut h0).partition([1, CONV_TILE]),
                        (&mut h1).partition([1, CONV_TILE]),
                        (&mut h2).partition([1, CONV_TILE]),
                        &rows,
                        &metadata,
                        &counts,
                    )
                    .generics(vec![fields.to_string()]),
                )
                .map(drop)
                .map_err(backend_error)
        } else {
            scope
                .record(
                    fused::conv_commit_lanes(
                        (&mut h0).partition([1, CONV_TILE]),
                        (&mut h1).partition([1, CONV_TILE]),
                        (&mut h2).partition([1, CONV_TILE]),
                        &rows,
                        &metadata,
                        &counts,
                    )
                    .generics(vec![fields.to_string(), lanes.to_string()]),
                )
                .map(drop)
                .map_err(backend_error)
        }
    }
}

/// Pack the immutable decay rate and bias of a delta node into one tensor.
/// # Errors
/// Reports a rejected launch.
pub fn pack_parameters(
    device: &Device,
    parameters: Buffer<'_, f32>,
    rate: Buffer<'_, f32>,
    bias: Buffer<'_, f32>,
) -> Result<()> {
    let value_heads = rate.len();
    let ordinal = device.ordinal();
    let parameters = parameters.require("packed delta parameters", 2 * value_heads)?;
    let bias = bias.require("delta bias", value_heads)?;
    let mut parameters = borrowed(parameters, ordinal, &[2, value_heads])?;
    let rate = borrowed(rate, ordinal, &[value_heads])?;
    let bias = borrowed(bias, ordinal, &[value_heads])?;
    fused::pack_parameters((&mut parameters).partition([2, 1]), &rate, &bias)
        .generics(vec![value_heads.to_string()])
        .sync_on(device.stream())
        .map(drop)
        .map_err(backend_error)
}
