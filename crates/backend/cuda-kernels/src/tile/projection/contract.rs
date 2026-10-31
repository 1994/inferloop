//! cuTile implementation of the projection family.
//!
//! The host owns every allocation; this module wraps its device pointers in borrowed tensors for
//! exactly one recording, and the captured graph keeps only the addresses.
#![expect(
    unsafe_code,
    reason = "Wraps host-owned device pointers in borrowed cuTile tensors for one recording"
)]

use super::pool_gemv;
use crate::contract::kernels::Projection;
use crate::contract::plan::PoolGemvPlan;
use crate::contract::ptr::DevicePtr;
use crate::contract::recording::Recording;
use cuda_core::DType;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, ErrorCode, Result};

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

impl Projection for TileProjection {
    unsafe fn pool_gemv_bf16(
        &self,
        recording: &Recording<'_>,
        plan: &PoolGemvPlan,
        out: DevicePtr,
        x: DevicePtr,
        w: DevicePtr,
    ) -> Result<()> {
        let Recording::Scope(scope) = recording else {
            return Err(Error::unsupported("cuTile launches record through a scope"));
        };
        let rows = i32::try_from(plan.rows).map_err(|_| geometry("rows"))?;
        let outputs = i32::try_from(plan.outputs).map_err(|_| geometry("outputs"))?;
        let columns = i32::try_from(plan.columns).map_err(|_| geometry("columns"))?;
        // SAFETY: the caller guarantees a contiguous `[rows, outputs]` allocation that outlives
        // every replay; the borrowed tensor frees nothing when it drops.
        let mut out = unsafe {
            Tensor::<f32>::borrow_raw_parts(
                out.0,
                self.ordinal,
                vec![rows, outputs],
                vec![outputs, 1],
            )
        };
        // SAFETY: as for `out`, with the `[rows, columns]` input.
        let x = unsafe {
            Tensor::<f32>::borrow_raw_parts(
                x.0,
                self.ordinal,
                vec![rows, columns],
                vec![columns, 1],
            )
        };
        // SAFETY: as for `out`, with the `[outputs, columns]` weight.
        let w = unsafe {
            Tensor::<bf16>::borrow_raw_parts(
                w.0,
                self.ordinal,
                vec![outputs, columns],
                vec![columns, 1],
            )
        };
        scope
            .record(
                pool_gemv::dense((&mut out).partition([1, plan.tile_rows]), &x, &w).generics(vec![
                    bf16::DTYPE.as_str().into(),
                    plan.tile_rows.to_string(),
                    plan.tile_columns.to_string(),
                    plan.columns.to_string(),
                ]),
            )
            .map_err(|error| Error::new(ErrorCode::Backend, error.to_string()))?;
        Ok(())
    }
}

fn geometry(field: &str) -> Error {
    Error::invalid(format!("projection {field} exceeds i32"))
}
