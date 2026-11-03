//! cuTile implementation of the vision family.
use super::encoder;
use super::vision_position_kernels;
use crate::contract::buffer::Buffer;
use crate::contract::recording::Recording;
use crate::contract::vision::{
    Gelu, Pack, PositionTaps, Vision, VisionDense, VisionNorm, VisionRope,
};
use crate::tile::projection::retyped_bf16;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::DType;
use cutile::prelude::*;
use infer_core::Result;

/// Rows of the encoder's output tile.
const TOKEN_TILE: usize = 32;
/// Columns of the encoder's output tile.
const COLUMN_TILE: usize = 64;
/// Columns of a compact head's segment.
const PACK_SEGMENT: usize = 16;

/// Vision kernels compiled with cuTile.
pub struct TileVision {
    ordinal: usize,
}

impl TileVision {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    fn rows<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
        name: &str,
        columns: usize,
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        if columns == 0 || !buffer.len().is_multiple_of(columns) {
            return Err(backend_error(format!("{name}: not rows of {columns}")));
        }
        borrowed(buffer, self.ordinal, &[buffer.len() / columns, columns])
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

impl Vision for TileVision {
    fn dense_bias(&self, recording: &Recording<'_>, dense: VisionDense<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let VisionDense {
            out,
            input,
            weight,
            bias,
            columns,
        } = dense;
        let rows = input.len() / columns;
        let mut out = self.rows(recording, out, "dense output", out.len() / rows)?;
        scope
            .record(
                encoder::dense_bias(
                    (&mut out).partition([TOKEN_TILE, COLUMN_TILE]),
                    &self.rows(recording, input, "dense input", columns)?,
                    &self.rows(recording, retyped_bf16(weight), "dense weight", columns)?,
                    &self.flat(recording, retyped_bf16(bias))?,
                )
                .generics(vec![columns.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn add(
        &self,
        recording: &Recording<'_>,
        columns: usize,
        out: Buffer<'_, f32>,
        left: Buffer<'_, f32>,
        right: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.rows(recording, out, "sum", columns)?;
        scope
            .record(
                encoder::add(
                    (&mut out).partition([TOKEN_TILE, COLUMN_TILE]),
                    &self.rows(recording, left, "left", columns)?,
                    &self.rows(recording, right, "right", columns)?,
                )
                .generics(vec![columns.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn gelu(
        &self,
        recording: &Recording<'_>,
        form: Gelu,
        columns: usize,
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.rows(recording, out, "activation", columns)?;
        let input = self.rows(recording, input, "activation input", columns)?;
        let generics = vec![columns.to_string()];
        let tile = (&mut out).partition([TOKEN_TILE, COLUMN_TILE]);
        match form {
            Gelu::Tanh => scope
                .record(encoder::gelu_tanh(tile, &input).generics(generics))
                .map_err(backend_error)?,
            Gelu::Erf => scope
                .record(encoder::gelu_erf(tile, &input).generics(generics))
                .map_err(backend_error)?,
        };
        Ok(())
    }

    fn layernorm(&self, recording: &Recording<'_>, norm: VisionNorm<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.flat(recording, norm.out)?;
        scope
            .record(
                encoder::layernorm(
                    (&mut out).partition([norm.tile]),
                    &self.rows(recording, norm.input, "norm input", norm.tile)?,
                    &self.flat(recording, retyped_bf16(norm.weight))?,
                    &self.flat(recording, retyped_bf16(norm.bias))?,
                    norm.epsilon,
                )
                .generics(vec![norm.dim.to_string(), norm.tile.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn rope(&self, recording: &Recording<'_>, rope: VisionRope<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let columns = rope.input.len() / rope.rows;
        let table = rope.cosines.len() / rope.rows;
        let mut out = self.rows(recording, rope.out, "rope output", columns)?;
        let input = self.rows(recording, rope.input, "rope input", columns)?;
        let cosines = self.rows(recording, rope.cosines, "cosines", table)?;
        let sines = self.rows(recording, rope.sines, "sines", table)?;
        match rope.real_half {
            Some(real) => scope
                .record(
                    encoder::rope_indexed(
                        (&mut out).partition([TOKEN_TILE, PACK_SEGMENT]),
                        &input,
                        &cosines,
                        &sines,
                    )
                    .generics(vec![rope.half.to_string(), real.to_string()]),
                )
                .map_err(backend_error)?,
            None => scope
                .record(
                    encoder::rope(
                        (&mut out).partition([TOKEN_TILE, rope.half]),
                        &input,
                        &cosines,
                        &sines,
                    )
                    .generics(vec![rope.half.to_string()]),
                )
                .map_err(backend_error)?,
        };
        Ok(())
    }

    fn pack(
        &self,
        recording: &Recording<'_>,
        direction: Pack,
        rows: usize,
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.rows(recording, out, "packed", out.len() / rows)?;
        let input = self.rows(recording, input, "pack input", input.len() / rows)?;
        let tile = (&mut out).partition([TOKEN_TILE, PACK_SEGMENT]);
        match direction {
            Pack::Wide => scope.record(encoder::pack_wide(tile, &input)),
            Pack::Compact => scope.record(encoder::pack_compact(tile, &input)),
        }
        .map_err(backend_error)?;
        Ok(())
    }

    fn add_positions(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        embedded: Buffer<'_, f32>,
        taps: PositionTaps<'_>,
    ) -> Result<()> {
        vision_position_kernels::add_positions(recording, self.ordinal, out, embedded, taps)
    }
}
