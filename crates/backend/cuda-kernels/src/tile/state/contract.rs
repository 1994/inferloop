//! cuTile implementation of the element-wise and slicing family.
use super::aux;
use crate::contract::buffer::Buffer;
use crate::contract::kernels::State;
use crate::contract::plan::{BinaryOp, SplitPlan, UnaryOp};
use crate::contract::recording::Recording;
use crate::tile::moe::moe as moe_kernels;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::DType;
use cutile::prelude::*;
use infer_core::{Error, Result};

/// Elements in one tile of these kernels.
const TILE: usize = 256;

/// State kernels compiled with cuTile.
pub struct TileState {
    ordinal: usize,
}

impl TileState {
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
}

impl State for TileState {
    fn unary(
        &self,
        recording: &Recording<'_>,
        op: UnaryOp,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.flat(recording, out)?;
        let x = self.flat(recording, x)?;
        scope
            .record(
                aux::unary((&mut out).partition([TILE]), &x)
                    .generics(vec![i32::from(op == UnaryOp::Sigmoid).to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn binary(
        &self,
        recording: &Recording<'_>,
        op: BinaryOp,
        out: Buffer<'_, f32>,
        a: Buffer<'_, f32>,
        b: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.flat(recording, out)?;
        let a = self.flat(recording, a)?;
        let b = self.flat(recording, b)?;
        scope
            .record(
                aux::binary((&mut out).partition([TILE]), &a, &b)
                    .generics(vec![i32::from(op == BinaryOp::Multiply).to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn scale(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        scalar: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.flat(recording, out)?;
        let x = self.flat(recording, x)?;
        let scalar = self.flat(recording, scalar)?;
        scope
            .record(moe_kernels::scale(
                (&mut out).partition([TILE]),
                &x,
                &scalar,
            ))
            .map_err(backend_error)?;
        Ok(())
    }

    fn split(
        &self,
        recording: &Recording<'_>,
        plan: &SplitPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let SplitPlan {
            rows,
            stride,
            width,
            part,
        } = *plan;
        recording.retain(out);
        recording.retain(x);
        let mut out = borrowed(
            out.require("split output", rows * width)?,
            self.ordinal,
            &[rows, width],
        )?;
        let x = borrowed(
            x.require("split input", rows * stride)?,
            self.ordinal,
            &[rows, stride],
        )?;
        let part = i32::try_from(part).map_err(|_| Error::invalid("split part exceeds i32"))?;
        scope
            .record(
                aux::split((&mut out).partition([1, width]), &x, part)
                    .generics(vec![stride.to_string(), width.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }
}
