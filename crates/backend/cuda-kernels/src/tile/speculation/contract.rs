//! cuTile implementation of the block-draft family.
use super::dflash2 as draft;
use crate::contract::buffer::{Bf16, Buffer};
use crate::contract::draft::{
    DRAFT_KEY_TILE, DRAFT_ROW_TILE, Draft, DraftAddNorm, DraftAttention, DraftConv, DraftNorm,
    DraftProjection, DraftRingStore, DraftRope, DraftSplit, DraftTop, DraftWalk, DraftWeight,
};
use crate::contract::recording::Recording;
use crate::tile::projection::{prefill_gemm as gemm, retyped_bf16, retyped_fp8};
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::DType;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::Result;

/// Output tile of the activation product.
const PRODUCT_TILE: usize = 256;

/// Draft kernels compiled with cuTile.
pub struct TileDraft {
    ordinal: usize,
}

impl TileDraft {
    /// Build the implementation for one device ordinal.
    #[must_use]
    pub const fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    fn tensor<T: DType>(
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

    fn flat<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        borrowed(buffer, self.ordinal, &[buffer.len()])
    }

    fn rows<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
        name: &str,
        columns: usize,
    ) -> Result<Tensor<T>> {
        self.tensor(recording, buffer, name, &[buffer.len() / columns, columns])
    }

    fn halves(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, Bf16>,
        name: &str,
        shape: &[usize],
    ) -> Result<Tensor<bf16>> {
        self.tensor(recording, retyped_bf16(buffer), name, shape)
    }
}

impl Draft for TileDraft {
    fn project(
        &self,
        recording: &Recording<'_>,
        weight: DraftWeight<'_>,
        projection: DraftProjection<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DraftProjection {
            out,
            input,
            rows,
            columns,
        } = projection;
        let mut out = self.rows(recording, out, "projection output", out.len() / rows)?;
        let input = self.tensor(recording, input, "projection input", &[rows, columns])?;
        let tiles = [DRAFT_ROW_TILE, DRAFT_KEY_TILE];
        let generics = vec![columns.to_string()];
        match weight {
            DraftWeight::Dense(weight) => {
                let weight = self.rows(recording, retyped_bf16(weight), "weight", columns)?;
                scope
                    .record(
                        gemm::dense((&mut out).partition(tiles), &input, &weight)
                            .generics(generics),
                    )
                    .map_err(backend_error)?;
            }
            DraftWeight::Fp8 { weight, scales } => {
                let weight = self.rows(recording, retyped_fp8(weight), "weight", columns)?;
                let scales = self.flat(recording, scales)?;
                scope
                    .record(
                        gemm::fp8((&mut out).partition(tiles), &input, &weight, &scales)
                            .generics(generics),
                    )
                    .map_err(backend_error)?;
            }
            DraftWeight::Fp8Block {
                weight,
                scales,
                scale_columns,
            } => {
                let weight = self.rows(recording, retyped_fp8(weight), "weight", columns)?;
                let scales = self.rows(recording, scales, "block scales", scale_columns)?;
                scope
                    .record(
                        draft::fp8_block((&mut out).partition(tiles), &input, &weight, &scales)
                            .generics(generics),
                    )
                    .map_err(backend_error)?;
            }
        }
        Ok(())
    }

    fn project_split(&self, recording: &Recording<'_>, split: DraftSplit<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DraftProjection {
            out,
            input,
            rows,
            columns,
        } = split.projection;
        let outputs = out.len() / rows;
        let tiles = [DRAFT_ROW_TILE, DRAFT_KEY_TILE];
        let mut partials = self.tensor(
            recording,
            split.partials,
            "split partials",
            &[split.splits * DRAFT_ROW_TILE, outputs],
        )?;
        let input = self.tensor(recording, input, "projection input", &[rows, columns])?;
        let weight = self.rows(recording, retyped_fp8(split.weight), "weight", columns)?;
        let k_tiles = i32::try_from(split.k_tiles).map_err(backend_error)?;
        let splits = i32::try_from(split.splits).map_err(backend_error)?;
        scope
            .record(
                draft::fp8_split((&mut partials).partition(tiles), &input, &weight, k_tiles)
                    .generics(vec![columns.to_string()]),
            )
            .map_err(backend_error)?;
        let mut out = self.tensor(recording, out, "projection output", &[rows, outputs])?;
        scope
            .record(draft::reduce_scaled(
                (&mut out).partition(tiles),
                &partials,
                &self.flat(recording, split.scales)?,
                splits,
            ))
            .map_err(backend_error)?;
        Ok(())
    }

    fn norm(&self, recording: &Recording<'_>, norm: DraftNorm<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DraftNorm {
            out,
            input,
            weight,
            dim,
            tile,
            epsilon,
        } = norm;
        let mut out = self.rows(recording, out, "norm output", dim)?;
        scope
            .record(
                draft::norm(
                    (&mut out).partition([1, tile]),
                    &self.rows(recording, input, "norm input", dim)?,
                    &self.flat(recording, weight)?,
                    epsilon,
                )
                .generics(vec![dim.to_string(), tile.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn add_norm(&self, recording: &Recording<'_>, add: DraftAddNorm<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DraftNorm {
            out,
            input,
            weight,
            dim,
            tile,
            epsilon,
        } = add.norm;
        let mut out = self.rows(recording, out, "norm output", dim)?;
        let mut stream = self.rows(recording, add.stream, "stream", dim)?;
        scope
            .record(
                draft::add_norm(
                    (&mut out).partition([1, tile]),
                    (&mut stream).partition([1, tile]),
                    &self.rows(recording, input, "residual", dim)?,
                    &self.rows(recording, add.delta, "delta", dim)?,
                    &self.flat(recording, weight)?,
                    epsilon,
                )
                .generics(vec![dim.to_string(), tile.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn embed(
        &self,
        recording: &Recording<'_>,
        shape: [usize; 2],
        out: Buffer<'_, f32>,
        table: Buffer<'_, Bf16>,
        tokens: Buffer<'_, i32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let [columns, tile] = shape;
        let mut out = self.rows(recording, out, "embedding output", columns)?;
        let table = self.rows(recording, retyped_bf16(table), "embedding table", columns)?;
        scope
            .record(
                draft::embed(
                    (&mut out).partition([1, tile]),
                    &table,
                    &self.flat(recording, tokens)?,
                )
                .generics(vec![tile.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn conv(&self, recording: &Recording<'_>, conv: DraftConv<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let shape = [conv.rows, conv.hidden];
        let mut out = self.tensor(recording, conv.out, "conv output", &shape)?;
        scope
            .record(
                draft::grouped_conv(
                    (&mut out).partition([1, conv.tile]),
                    &self.tensor(recording, conv.input, "conv input", &shape)?,
                    &self.tensor(
                        recording,
                        conv.coefficients,
                        "conv coefficients",
                        &[conv.rows, conv.coefficients.len() / conv.rows],
                    )?,
                    &self.rows(recording, conv.base, "conv base", conv.hidden)?,
                    conv.side,
                )
                .generics(vec![
                    conv.tile.to_string(),
                    conv.group_size.to_string(),
                    (conv.tile / conv.group_size).to_string(),
                    conv.groups.to_string(),
                    conv.block.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn rope(&self, recording: &Recording<'_>, rope: DraftRope<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let half = rope.dim / 2;
        let mut out = self.rows(recording, rope.out, "rope output", rope.dim)?;
        scope
            .record(
                draft::rope_rows(
                    (&mut out).partition([1, half]),
                    &self.rows(recording, rope.input, "rope input", rope.dim)?,
                    &self.flat(recording, rope.frequencies)?,
                    &self.flat(recording, rope.positions)?,
                )
                .generics(vec![half.to_string(), rope.heads.to_string()]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn ring_store(&self, recording: &Recording<'_>, store: DraftRingStore<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut ring = self.halves(
            recording,
            store.ring,
            "ring",
            &[store.heads, store.capacity, store.dim],
        )?;
        scope
            .record(
                draft::ring_store(
                    (&mut ring).partition([1, store.capacity, store.dim]),
                    &self.tensor(
                        recording,
                        store.input,
                        "ring rows",
                        &[store.rows, store.heads, store.dim],
                    )?,
                    &self.flat(recording, store.rings)?,
                    &self.flat(recording, store.positions)?,
                )
                .generics(vec![
                    store.dim.to_string(),
                    store.capacity.to_string(),
                    store.ring_rows.to_string(),
                    store.rows.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn attend(&self, recording: &Recording<'_>, attention: DraftAttention<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DraftAttention {
            rows,
            kv_heads,
            group,
            dim,
            capacity,
            block,
            ..
        } = attention;
        let width = group * dim;
        let mut out = self.tensor(
            recording,
            attention.out,
            "attention output",
            &[rows, kv_heads, width],
        )?;
        let ring = [kv_heads, capacity, dim];
        scope
            .record(
                draft::block_attention(
                    (&mut out).partition([block, 1, width]),
                    &self.tensor(
                        recording,
                        attention.queries,
                        "queries",
                        &[rows, kv_heads, width],
                    )?,
                    &self.halves(recording, attention.keys, "key ring", &ring)?,
                    &self.halves(recording, attention.values, "value ring", &ring)?,
                    &self.flat(recording, attention.bases)?,
                    &self.flat(recording, attention.rings)?,
                )
                .generics(vec![
                    dim.to_string(),
                    group.to_string(),
                    width.to_string(),
                    block.to_string(),
                    (block * group).to_string(),
                    attention.ring_rows.to_string(),
                    attention.window.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn gated_product(
        &self,
        recording: &Recording<'_>,
        out: Buffer<'_, f32>,
        gate: Buffer<'_, f32>,
        up: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.flat(recording, out)?;
        scope
            .record(draft::silu_mul(
                (&mut out).partition([PRODUCT_TILE]),
                &self.flat(recording, gate)?,
                &self.flat(recording, up)?,
            ))
            .map_err(backend_error)?;
        Ok(())
    }

    fn top_candidates(&self, recording: &Recording<'_>, top: DraftTop<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let k = top.top_k;
        let merged = top.vocabulary.div_ceil(top.top_block) * k;
        let partial = [top.rows, merged];
        let final_shape = [top.rows, k];
        let mut values = self.tensor(recording, top.partial_values, "partial values", &partial)?;
        let mut ids = self.tensor(recording, top.partial_ids, "partial ids", &partial)?;
        scope
            .record(
                draft::top_partial(
                    (&mut values).partition([1, k]),
                    (&mut ids).partition([1, k]),
                    &self.tensor(recording, top.logits, "logits", &[top.rows, top.vocabulary])?,
                )
                .generics(vec![
                    top.vocabulary.to_string(),
                    top.top_block.to_string(),
                    k.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        let mut unary = self.tensor(recording, top.unary, "unary", &final_shape)?;
        let mut candidates = self.tensor(recording, top.candidates, "candidates", &final_shape)?;
        scope
            .record(
                draft::top_finish(
                    (&mut unary).partition([1, k]),
                    (&mut candidates).partition([1, k]),
                    &values,
                    &ids,
                )
                .generics(vec![
                    merged.to_string(),
                    merged.next_power_of_two().to_string(),
                    k.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn walk(&self, recording: &Recording<'_>, walk: DraftWalk<'_>) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DraftWalk {
            rows,
            slots,
            rank,
            top_k,
            block,
            ..
        } = walk;
        let mut tokens = self.tensor(recording, walk.tokens, "tokens", &[slots, block])?;
        let predecessor = retyped_bf16(walk.predecessor);
        let successor = retyped_bf16(walk.successor);
        scope
            .record(
                draft::select_walk(
                    (&mut tokens).partition([1, block]),
                    &self.tensor(recording, walk.candidates, "candidates", &[rows, top_k])?,
                    &self.tensor(recording, walk.unary, "unary", &[rows, top_k])?,
                    &self.tensor(recording, walk.selector, "selector", &[rows, rank])?,
                    &self.rows(recording, predecessor, "predecessor", rank)?,
                    &self.rows(recording, successor, "successor", rank)?,
                    &self.flat(recording, walk.anchors)?,
                )
                .generics(vec![
                    rank.to_string(),
                    top_k.to_string(),
                    block.to_string(),
                ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }
}
