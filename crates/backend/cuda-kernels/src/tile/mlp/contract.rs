//! cuTile implementation of the fused MLP family.
use super::{fused, pdl, pdl_consumers as consumers};
use crate::contract::buffer::Buffer;
use crate::contract::kernels::{Mlp, ProductBuffers, WeightRef};
use crate::contract::plan::{DownPlan, RowNormPlan};
use crate::contract::recording::Recording;
use crate::tile::{backend_error, borrowed};
use cuda_core::DType;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};

/// Elements in one tile of the auxiliary kernels.
const AUX_TILE: usize = 256;
/// Elements one FP4 scale covers.
const FP4_GROUP: usize = 16;

/// MLP kernels compiled with cuTile.
pub struct TileMlp {
    ordinal: usize,
}

impl TileMlp {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    fn flat<T: DType>(&self, buffer: Buffer<'_, T>) -> Result<Tensor<T>> {
        borrowed(buffer, self.ordinal, &[buffer.len()])
    }
}

fn capture<'a>(recording: &Recording<'a>) -> Result<&'a Scope> {
    recording
        .cutile_scope()
        .ok_or_else(|| Error::unsupported("cuTile launches record through a scope"))
}

impl Mlp for TileMlp {
    fn row_norm(
        &self,
        recording: &Recording<'_>,
        plan: &RowNormPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        recording.retain(out);
        recording.retain(x);
        recording.retain(weight);
        let block = plan.dimension.next_power_of_two();
        let mut out = self.flat(out.require("norm output", plan.dimension)?)?;
        let (x, weight) = (self.flat(x)?, self.flat(weight)?);
        scope
            .record(
                fused::norm(
                    (&mut out).partition([block]),
                    &x,
                    &weight,
                    plan.epsilon,
                    plan.offset,
                )
                .generics(vec![plan.dimension.to_string(), block.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn product(&self, recording: &Recording<'_>, buffers: ProductBuffers<'_>) -> Result<()> {
        let scope = capture(recording)?;
        recording.retain(buffers.out);
        recording.retain(buffers.gate);
        recording.retain(buffers.up);
        let mut out = self.flat(buffers.out)?;
        let (gate, up) = (self.flat(buffers.gate)?, self.flat(buffers.up)?);
        scope
            .record(
                fused::silu_mul((&mut out).partition([AUX_TILE]), &gate, &up)
                    .generics(vec![AUX_TILE.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn residual(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        down: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        recording.retain(out);
        recording.retain(down);
        recording.retain(x);
        let mut out = self.flat(out)?;
        let (down, x) = (self.flat(down)?, self.flat(x)?);
        scope
            .record(
                fused::residual((&mut out).partition([AUX_TILE]), &down, &x)
                    .generics(vec![AUX_TILE.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn signalling_product(
        &self,
        recording: &Recording<'_>,
        buffers: ProductBuffers<'_>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        recording.retain(buffers.out);
        recording.retain(buffers.gate);
        recording.retain(buffers.up);
        let mut out = self.flat(buffers.out)?;
        let (gate, up) = (self.flat(buffers.gate)?, self.flat(buffers.up)?);
        scope
            .record(pdl::silu_mul((&mut out).partition([AUX_TILE]), &gate, &up))
            .map_err(backend_error)?;
        Ok(())
    }

    #[expect(
        unsafe_code,
        reason = "Audited PDL edge: the consumer waits on the producer's token before each read"
    )]
    fn down_after_product(
        &self,
        recording: &Recording<'_>,
        plan: &DownPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        recording.retain(out);
        recording.retain(x);
        let columns = plan.columns;
        let mut out = self.flat(out)?;
        let x = borrowed(x.require("down input", columns)?, self.ordinal, &[columns])?;
        let mut generics = vec![
            plan.tile_rows.to_string(),
            plan.tile_columns.to_string(),
            columns.to_string(),
        ];
        let tile = (&mut out).partition([plan.tile_rows]);
        match weight {
            WeightRef::Dense(w) => {
                recording.retain(w);
                // SAFETY: `Bf16` marks `bf16` memory.
                let w: Buffer<'_, bf16> = unsafe { w.retyped() };
                let w = borrowed(w, self.ordinal, &[w.len() / columns, columns])?;
                generics.insert(0, bf16::DTYPE.as_str().into());
                let launch = consumers::dense(tile, &x, &w).generics(generics);
                // SAFETY: this kernel waits before reading the producer's activation buffer.
                scope
                    .record(unsafe { launch.programmatic_dependent_launch() })
                    .map(drop)
            }
            WeightRef::Fp8 { weight, scales } => {
                recording.retain(weight);
                recording.retain(scales);
                // SAFETY: `Fp8` marks `f8e4m3fn` memory.
                let w: Buffer<'_, cuda_core::f8e4m3fn> = unsafe { weight.retyped() };
                let w = borrowed(w, self.ordinal, &[w.len() / columns, columns])?;
                let s = self.flat(scales)?;
                let launch = consumers::fp8(tile, &x, &w, &s).generics(generics);
                // SAFETY: same token-ordered activation reads; weights and scales are immutable.
                scope
                    .record(unsafe { launch.programmatic_dependent_launch() })
                    .map(drop)
            }
            WeightRef::Fp4 {
                weight,
                scales,
                inverse_global,
            } => {
                recording.retain(weight);
                recording.retain(scales);
                // SAFETY: `Fp4Pair` marks `f4e2m1fnx2` memory.
                let w: Buffer<'_, cuda_core::f4e2m1fnx2> = unsafe { weight.retyped() };
                // SAFETY: `Fp8` marks `f8e4m3fn` memory.
                let s: Buffer<'_, cuda_core::f8e4m3fn> = unsafe { scales.retyped() };
                let (code_columns, scale_columns) = (columns / 2, columns / FP4_GROUP);
                let rows = w.len() / code_columns;
                let w = borrowed(w, self.ordinal, &[rows, code_columns])?;
                let s = borrowed(s, self.ordinal, &[rows, scale_columns])?;
                generics.extend([
                    (plan.tile_columns / 2).to_string(),
                    (plan.tile_columns / FP4_GROUP).to_string(),
                ]);
                let launch = consumers::nvfp4(tile, &x, &w, &s, inverse_global).generics(generics);
                // SAFETY: same token-ordered activation reads; the graph retains all storage.
                scope
                    .record(unsafe { launch.programmatic_dependent_launch() })
                    .map(drop)
            }
        }
        .map_err(backend_error)
    }
}
