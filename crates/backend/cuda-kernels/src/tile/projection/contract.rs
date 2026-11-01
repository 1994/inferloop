//! cuTile implementation of the projection family.
use super::pool_gemv;
use crate::contract::buffer::{Bf16, Buffer};
use crate::contract::kernels::Projection;
use crate::contract::plan::PoolGemvPlan;
use crate::contract::recording::Recording;
use crate::tile::{backend_error, borrowed};
use cuda_core::DType;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};

/// Projection kernels compiled with cuTile.
pub struct TileProjection {
    ordinal: usize,
}

impl TileProjection {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }
}

/// The same memory, typed for cuTile.
#[expect(
    unsafe_code,
    reason = "Bf16 and bf16 are the same two bytes; only the marker changes"
)]
const fn weight(buffer: Buffer<'_, Bf16>) -> Buffer<'_, bf16> {
    // SAFETY: the address and element count are unchanged and carry the caller's guarantee.
    unsafe { Buffer::new(buffer.address(), buffer.len()) }
}

impl Projection for TileProjection {
    fn pool_gemv_bf16(
        &self,
        recording: &Recording<'_>,
        plan: &PoolGemvPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        w: Buffer<'_, Bf16>,
    ) -> Result<()> {
        let Some(scope) = recording.cutile_scope() else {
            return Err(Error::unsupported("cuTile launches record through a scope"));
        };
        recording.retain(out);
        recording.retain(x);
        recording.retain(w);
        let (rows, outputs, columns) = (plan.rows, plan.outputs, plan.columns);
        let out = out.require("projection output", rows * outputs)?;
        let x = x.require("projection input", rows * columns)?;
        let w = w.require("projection weight", outputs * columns)?;
        let mut out = borrowed(out, self.ordinal, &[rows, outputs])?;
        let x = borrowed(x, self.ordinal, &[rows, columns])?;
        let w = borrowed(weight(w), self.ordinal, &[outputs, columns])?;
        scope
            .record(
                pool_gemv::dense((&mut out).partition([1, plan.tile_rows]), &x, &w).generics(vec![
                    bf16::DTYPE.as_str().into(),
                    plan.tile_rows.to_string(),
                    plan.tile_columns.to_string(),
                    plan.columns.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }
}
