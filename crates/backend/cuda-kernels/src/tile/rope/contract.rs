//! cuTile implementation of the rotary family.
use super::slot_rope;
use crate::contract::buffer::Buffer;
use crate::contract::kernels::{Rope, RopeBuffers};
use crate::contract::plan::{RopeMode, RopePlan};
use crate::contract::recording::Recording;
use crate::tile::state::aux;
use crate::tile::{backend_error, borrowed};
use cutile::prelude::*;
use infer_core::{Error, Result};

/// Rotary kernels compiled with cuTile.
pub struct TileRope {
    ordinal: usize,
}

impl TileRope {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }
}

fn require_axes(axes: Option<Buffer<'_, i32>>, half: usize) -> Result<Buffer<'_, i32>> {
    axes.ok_or_else(|| Error::invalid("rotary mode reads per-axis positions"))?
        .require("rotary axes", half)
}

impl Rope for TileRope {
    fn rope(
        &self,
        recording: &Recording<'_>,
        plan: &RopePlan,
        buffers: RopeBuffers<'_>,
    ) -> Result<()> {
        let Recording::Scope(scope) = recording else {
            return Err(Error::unsupported("cuTile launches record through a scope"));
        };
        let RopePlan {
            mode,
            rows,
            heads,
            head_dim,
            half,
        } = *plan;
        let elements = rows * head_dim;
        let shape = [rows, head_dim];
        let mut out = borrowed(
            buffers.out.require("rotary output", elements)?,
            self.ordinal,
            &shape,
        )?;
        let x = borrowed(
            buffers.x.require("rotary input", elements)?,
            self.ordinal,
            &shape,
        )?;
        let frequencies = borrowed(
            buffers.frequencies.require("rotary frequencies", half)?,
            self.ordinal,
            &[half],
        )?;
        let metadata = borrowed(buffers.metadata, self.ordinal, &[buffers.metadata.len()])?;
        let out = (&mut out).partition([1, half]);
        let single = vec![head_dim.to_string(), half.to_string()];
        let prompt = vec![head_dim.to_string(), half.to_string(), heads.to_string()];
        match mode {
            RopeMode::Single => scope
                .record(aux::rope(out, &x, &frequencies, &metadata).generics(single))
                .map(drop),
            RopeMode::SingleAxes => {
                let axes = require_axes(buffers.axes, half)?;
                let axes = borrowed(axes, self.ordinal, &[half])?;
                scope
                    .record(aux::mrope(out, &x, &frequencies, &axes, &metadata).generics(single))
                    .map(drop)
            }
            RopeMode::Prompt => scope
                .record(aux::prefill_rope(out, &x, &frequencies, &metadata).generics(prompt))
                .map(drop),
            RopeMode::PromptAxes => {
                let axes = require_axes(buffers.axes, half)?;
                let axes = borrowed(axes, self.ordinal, &[half])?;
                scope
                    .record(
                        aux::prefill_mrope(out, &x, &frequencies, &axes, &metadata)
                            .generics(prompt),
                    )
                    .map(drop)
            }
            RopeMode::Pooled { fields } => scope
                .record(
                    slot_rope::rope(out, &x, &frequencies, &metadata).generics(
                        [head_dim, half, heads, fields]
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                    ),
                )
                .map(drop),
        }
        .map_err(backend_error)?;
        Ok(())
    }
}
