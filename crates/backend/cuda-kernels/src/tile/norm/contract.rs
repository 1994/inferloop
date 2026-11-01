//! cuTile implementation of the normalization family.
use super::{norm_fp4, norm_maximum};
use crate::contract::buffer::{Buffer, Fp4Pair, Fp8};
use crate::contract::kernels::{
    FusionNormBuffers, Norm, NormBuffers, NormFp4Buffers, NormMaximumBuffers,
};
use crate::contract::plan::{FusionNormPlan, FusionShape, NormLayout, NormPlan, QuantNormPlan};
use crate::contract::recording::Recording;
use crate::tile::state::aux;
use crate::tile::{backend_error, borrowed};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use infer_core::{Error, Result};

/// Elements one FP4 scale covers.
const FP4_GROUP: usize = 16;

/// Normalization kernels compiled with cuTile.
pub struct TileNorm {
    ordinal: usize,
}

impl TileNorm {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }
}

fn capture<'a>(recording: &Recording<'a>) -> Result<&'a Scope> {
    recording
        .cutile_scope()
        .ok_or_else(|| Error::unsupported("cuTile launches record through a scope"))
}

fn dimensions(values: &[usize]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

/// Host markers name these elements; cuTile needs its own types.
#[expect(
    unsafe_code,
    reason = "Each marker is declared for exactly this element's size and layout"
)]
const fn typed<'a>(
    codes: Buffer<'a, Fp4Pair>,
    scales: Buffer<'a, Fp8>,
) -> (Buffer<'a, f4e2m1fnx2>, Buffer<'a, f8e4m3fn>) {
    // SAFETY: `Fp4Pair` marks `f4e2m1fnx2` memory.
    let codes = unsafe { codes.retyped() };
    // SAFETY: `Fp8` marks `f8e4m3fn` memory.
    let scales = unsafe { scales.retyped() };
    (codes, scales)
}

impl Norm for TileNorm {
    fn norm(
        &self,
        recording: &Recording<'_>,
        plan: &NormPlan,
        buffers: NormBuffers<'_>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        let (rows, dimension) = (plan.rows, plan.dimension);
        for buffer in [buffers.out, buffers.x, buffers.weight, buffers.gate] {
            recording.retain(buffer);
        }
        let elements = rows * dimension;
        let shape = [rows, dimension];
        let mut out = borrowed(
            buffers.out.require("norm output", elements)?,
            self.ordinal,
            &shape,
        )?;
        let x = borrowed(
            buffers.x.require("norm input", elements)?,
            self.ordinal,
            &shape,
        )?;
        let gate = borrowed(
            buffers.gate.require("norm gate", elements)?,
            self.ordinal,
            &shape,
        )?;
        let weight = borrowed(buffers.weight, self.ordinal, &[buffers.weight.len()])?;
        match plan.layout {
            NormLayout::Rows { tile } => scope
                .record(
                    aux::norm_rows(
                        (&mut out).partition([tile, dimension]),
                        &x,
                        &weight,
                        &gate,
                        plan.epsilon,
                        plan.offset,
                    )
                    .generics(vec![
                        dimension.to_string(),
                        i32::from(plan.gated).to_string(),
                    ]),
                )
                .map(drop),
            NormLayout::Block => {
                let block = dimension.next_power_of_two();
                let options = plan
                    .occupancy
                    .map_or_else(CompileOptions::default, |occupancy| {
                        CompileOptions::default().occupancy(occupancy)
                    });
                scope
                    .record(
                        aux::norm(
                            (&mut out).partition([1, block]),
                            &x,
                            &weight,
                            &gate,
                            plan.epsilon,
                            plan.offset,
                        )
                        .generics(vec![
                            dimension.to_string(),
                            block.to_string(),
                            i32::from(plan.gated).to_string(),
                        ])
                        .compile_options(options),
                    )
                    .map(drop)
            }
        }
        .map_err(backend_error)
    }

    fn fusion(
        &self,
        recording: &Recording<'_>,
        plan: &FusionNormPlan,
        buffers: FusionNormBuffers<'_>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        let dimension = plan.dimension;
        let block = dimension.next_power_of_two();
        for buffer in [
            buffers.out,
            buffers.embedding,
            buffers.hidden,
            buffers.norms,
        ] {
            recording.retain(buffer);
        }
        let norms = borrowed(
            buffers.norms.require("fusion norms", 2 * dimension)?,
            self.ordinal,
            &[2, dimension],
        )?;
        let generics = dimensions(&[dimension, block]);
        match plan.shape {
            FusionShape::Single => {
                let mut out = borrowed(
                    buffers.out.require("fusion output", 2 * dimension)?,
                    self.ordinal,
                    &[2, dimension],
                )?;
                let embedding =
                    borrowed(buffers.embedding, self.ordinal, &[buffers.embedding.len()])?;
                let hidden = borrowed(buffers.hidden, self.ordinal, &[buffers.hidden.len()])?;
                scope
                    .record(
                        aux::fusion_norm(
                            (&mut out).partition([1, block]),
                            &embedding,
                            &hidden,
                            &norms,
                            plan.epsilon,
                            plan.offset,
                        )
                        .generics(generics),
                    )
                    .map(drop)
            }
            FusionShape::Lanes { lanes } => {
                let mut out = borrowed(
                    buffers
                        .out
                        .require("fusion output", 2 * lanes * dimension)?,
                    self.ordinal,
                    &[2 * lanes, dimension],
                )?;
                let embedding = borrowed(
                    buffers
                        .embedding
                        .require("fusion embedding", lanes * dimension)?,
                    self.ordinal,
                    &[lanes, dimension],
                )?;
                let hidden = borrowed(
                    buffers.hidden,
                    self.ordinal,
                    &[buffers.hidden.len() / dimension, dimension],
                )?;
                scope
                    .record(
                        aux::fusion_norm_rows(
                            (&mut out).partition([1, block]),
                            &embedding,
                            &hidden,
                            &norms,
                            plan.epsilon,
                            plan.offset,
                        )
                        .generics(generics),
                    )
                    .map(drop)
            }
        }
        .map_err(backend_error)
    }

    fn maximum(
        &self,
        recording: &Recording<'_>,
        plan: &QuantNormPlan,
        buffers: NormMaximumBuffers<'_>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        let (rows, dimension) = (plan.rows, plan.dimension);
        for buffer in [buffers.out, buffers.maximum, buffers.x, buffers.weight] {
            recording.retain(buffer);
        }
        let shape = [rows, dimension];
        let elements = rows * dimension;
        let mut out = borrowed(
            buffers.out.require("norm output", elements)?,
            self.ordinal,
            &shape,
        )?;
        let mut maximum = borrowed(
            buffers.maximum.require("norm maximum", rows)?,
            self.ordinal,
            &[rows, 1],
        )?;
        let x = borrowed(
            buffers.x.require("norm input", elements)?,
            self.ordinal,
            &shape,
        )?;
        let weight = borrowed(buffers.weight, self.ordinal, &[buffers.weight.len()])?;
        let block = dimension.next_power_of_two();
        scope
            .record(
                norm_maximum::norm_max(
                    (&mut out).partition([1, block]),
                    (&mut maximum).partition([1, 1]),
                    &x,
                    &weight,
                    plan.epsilon,
                    plan.offset,
                )
                .generics(dimensions(&[dimension, block])),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn fp4(
        &self,
        recording: &Recording<'_>,
        plan: &QuantNormPlan,
        input_global: f32,
        buffers: NormFp4Buffers<'_>,
    ) -> Result<()> {
        let scope = capture(recording)?;
        let (rows, dimension) = (plan.rows, plan.dimension);
        for buffer in [buffers.out, buffers.x, buffers.weight] {
            recording.retain(buffer);
        }
        recording.retain(buffers.codes);
        recording.retain(buffers.scales);
        let block = dimension.next_power_of_two();
        let (group, half) = (block / FP4_GROUP, block / 2);
        let shape = [rows, dimension];
        let elements = rows * dimension;
        let (codes, scales) = typed(buffers.codes, buffers.scales);
        let mut out = borrowed(
            buffers.out.require("norm output", elements)?,
            self.ordinal,
            &shape,
        )?;
        // Storage follows the dimension; the partition tile follows its power-of-two block.
        let (code_columns, scale_columns) = (dimension / 2, dimension / FP4_GROUP);
        let mut codes = borrowed(
            codes.require("norm codes", rows * code_columns)?,
            self.ordinal,
            &[rows, code_columns],
        )?;
        let mut scales = borrowed(
            scales.require("norm scales", rows * scale_columns)?,
            self.ordinal,
            &[rows, scale_columns],
        )?;
        let x = borrowed(
            buffers.x.require("norm input", elements)?,
            self.ordinal,
            &shape,
        )?;
        let weight = borrowed(buffers.weight, self.ordinal, &[buffers.weight.len()])?;
        scope
            .record(
                norm_fp4::norm_fp4(
                    (&mut out).partition([1, block]),
                    (&mut codes).partition([1, half]),
                    (&mut scales).partition([1, group]),
                    &x,
                    &weight,
                    plan.epsilon,
                    plan.offset,
                    input_global,
                )
                .generics(dimensions(&[dimension, block, group, half])),
            )
            .map_err(backend_error)?;
        Ok(())
    }
}
