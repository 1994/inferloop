//! cuTile implementation of the mixture-of-experts family.
use super::moe;
use crate::contract::buffer::{Bf16, Buffer, Fp8};
use crate::contract::kernels::{ExpertWeight, Moe};
use crate::contract::plan::{ExpertPlan, RoutePlan};
use crate::contract::recording::Recording;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::{DType, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::Result;

/// Elements in one tile of the combine kernel.
const COMBINE_TILE: usize = 256;
/// Output tile of an expert projection.
const PROJECTION_TILE: [usize; 2] = [1, 4];

/// `CHANNEL` generic of the FP8 kernel for per-row scales.
const CHANNEL_SCALES: &str = "1";
/// `CHANNEL` generic of the FP8 kernel for per-block scales.
const BLOCK_SCALES: &str = "0";

/// Mixture-of-experts kernels compiled with cuTile.
pub struct TileMoe {
    ordinal: usize,
}

impl TileMoe {
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

/// The same memory, typed for cuTile.
#[expect(
    unsafe_code,
    reason = "Each marker is declared for exactly this element's size and layout"
)]
const fn fp8(weight: Buffer<'_, Fp8>) -> Buffer<'_, f8e4m3fn> {
    // SAFETY: `Fp8` marks `f8e4m3fn` memory.
    unsafe { weight.retyped() }
}

/// The same memory, typed for cuTile.
#[expect(
    unsafe_code,
    reason = "Each marker is declared for exactly this element's size and layout"
)]
const fn dense(weight: Buffer<'_, Bf16>) -> Buffer<'_, bf16> {
    // SAFETY: `Bf16` marks `bf16` memory.
    unsafe { weight.retyped() }
}

impl Moe for TileMoe {
    fn route(
        &self,
        recording: &Recording<'_>,
        plan: &RoutePlan,
        out: Buffer<'_, f32>,
        scores: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.flat(recording, out)?;
        let scores = self.flat(recording, scores)?;
        scope
            .record(
                moe::route((&mut out).partition([1]), &scores, f32::NEG_INFINITY).generics(vec![
                    plan.experts.to_string(),
                    plan.experts.next_power_of_two().to_string(),
                    plan.active.to_string(),
                    i32::from(plan.normalize).to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn combine(
        &self,
        recording: &Recording<'_>,
        (active, hidden): (usize, usize),
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        routes: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        recording.retain(x);
        let mut out = self.flat(recording, out)?;
        let x = borrowed(
            x.require("combine input", active * hidden)?,
            self.ordinal,
            &[active, hidden],
        )?;
        let routes = self.flat(recording, routes)?;
        scope
            .record(
                moe::combine((&mut out).partition([COMBINE_TILE]), &x, &routes)
                    .generics(vec![active.to_string(), hidden.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn experts(
        &self,
        recording: &Recording<'_>,
        plan: &ExpertPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        routes: Buffer<'_, f32>,
        weight: ExpertWeight<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let ExpertPlan {
            experts,
            active,
            rows,
            input,
            routed_input,
        } = *plan;
        recording.retain(out);
        recording.retain(x);
        let mut out = borrowed(
            out.require("expert output", active * rows)?,
            self.ordinal,
            &[active, rows],
        )?;
        let x_rows = if routed_input { active } else { 1 };
        let x = borrowed(
            x.require("expert input", x_rows * input)?,
            self.ordinal,
            &[x_rows, input],
        )?;
        let routes = self.flat(recording, routes)?;
        let bank = [experts, rows, input];
        let mut generics = vec![
            rows.to_string(),
            input.to_string(),
            i32::from(routed_input).to_string(),
        ];
        let tile = (&mut out).partition(PROJECTION_TILE);
        match weight {
            ExpertWeight::Dense(w) => {
                recording.retain(w);
                let w = borrowed(dense(w), self.ordinal, &bank)?;
                scope
                    .record(moe::dense(tile, &x, &routes, &w).generics(generics))
                    .map(drop)
            }
            ExpertWeight::Fp8Channel {
                weight,
                scales,
                quantized_inputs,
            } => {
                recording.retain(weight);
                recording.retain(scales);
                generics.extend([
                    CHANNEL_SCALES.into(),
                    i32::from(quantized_inputs).to_string(),
                    input.next_power_of_two().to_string(),
                ]);
                let w = borrowed(fp8(weight), self.ordinal, &bank)?;
                let s = borrowed(scales, self.ordinal, &[experts, rows, 1])?;
                scope
                    .record(moe::fp8(tile, &x, &routes, &w, &s).generics(generics))
                    .map(drop)
            }
            ExpertWeight::Fp8Block {
                weight,
                scales,
                block_columns,
                quantized_inputs,
            } => {
                recording.retain(weight);
                recording.retain(scales);
                generics.extend([
                    BLOCK_SCALES.into(),
                    i32::from(quantized_inputs).to_string(),
                    input.next_power_of_two().to_string(),
                ]);
                let w = borrowed(fp8(weight), self.ordinal, &bank)?;
                let s = borrowed(
                    scales,
                    self.ordinal,
                    &[experts, rows, input / block_columns],
                )?;
                scope
                    .record(moe::fp8(tile, &x, &routes, &w, &s).generics(generics))
                    .map(drop)
            }
        }
        .map_err(backend_error)
    }
}
