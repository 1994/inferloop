//! cuTile implementation of the sampling family.
use super::greedy;
use crate::contract::buffer::Buffer;
use crate::contract::kernels::{GreedyPartials, Sampling};
use crate::contract::plan::GreedyPlan;
use crate::contract::recording::Recording;
use crate::tile::{backend_error, borrowed, capture_scope};
use cutile::prelude::*;
use infer_core::Result;

/// Elements a token row holds: its token and validity.
const TOKEN_COLUMNS: usize = 2;

/// Sampling kernels compiled with cuTile.
pub struct TileSampling {
    ordinal: usize,
}

impl TileSampling {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    /// The reduction state as tensors of `[rows, blocks]`.
    fn partials(
        &self,
        recording: &Recording<'_>,
        plan: &GreedyPlan,
        partials: GreedyPartials<'_>,
    ) -> Result<(Tensor<u32>, Tensor<i32>, Tensor<i32>)> {
        let shape = [plan.rows, plan.blocks];
        let elements = plan.rows * plan.blocks;
        recording.retain(partials.keys);
        recording.retain(partials.indices);
        recording.retain(partials.invalid);
        Ok((
            borrowed(
                partials.keys.require("greedy keys", elements)?,
                self.ordinal,
                &shape,
            )?,
            borrowed(
                partials.indices.require("greedy indices", elements)?,
                self.ordinal,
                &shape,
            )?,
            borrowed(
                partials.invalid.require("greedy invalid", elements)?,
                self.ordinal,
                &shape,
            )?,
        ))
    }
}

impl Sampling for TileSampling {
    fn greedy_partials(
        &self,
        recording: &Recording<'_>,
        plan: &GreedyPlan,
        partials: GreedyPartials<'_>,
        logits: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let (mut keys, mut indices, mut invalid) = self.partials(recording, plan, partials)?;
        recording.retain(logits);
        let logits = borrowed(
            logits.require("greedy logits", plan.rows * plan.vocabulary)?,
            self.ordinal,
            &[plan.rows, plan.vocabulary],
        )?;
        scope
            .record(
                greedy::partials(
                    (&mut keys).partition([1, 1]),
                    (&mut indices).partition([1, 1]),
                    (&mut invalid).partition([1, 1]),
                    &logits,
                )
                .generics(vec![plan.vocabulary.to_string(), plan.tile.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn greedy_finish(
        &self,
        recording: &Recording<'_>,
        plan: &GreedyPlan,
        offset: usize,
        output: Buffer<'_, i32>,
        partials: GreedyPartials<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let (keys, indices, invalid) = self.partials(recording, plan, partials)?;
        recording.retain(output);
        let rows = output.len() / TOKEN_COLUMNS;
        let mut output = borrowed(output, self.ordinal, &[rows, TOKEN_COLUMNS])?;
        scope
            .record(
                greedy::finish(
                    (&mut output).partition([1, TOKEN_COLUMNS]),
                    &keys,
                    &indices,
                    &invalid,
                )
                .generics(vec![
                    plan.blocks.to_string(),
                    plan.blocks.next_power_of_two().to_string(),
                    offset.to_string(),
                    plan.rows.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }
}
