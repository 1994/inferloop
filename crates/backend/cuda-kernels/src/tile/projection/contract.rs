//! cuTile implementation of the projection family.
use super::{
    fp8_gemm, fp8_quant_reduction, linear, linear_batch, nvfp4_gemm, pool_gemv, prefill_gemm,
    prompt_logits, small_gemm,
};
use crate::contract::buffer::{Bf16, Buffer, Fp4Pair, Fp8};
use crate::contract::kernels::{Projection, WeightRef};
use crate::contract::plan::{
    DenseKernel, DenseRowsPlan, DownPlan, EmbeddingBatchPlan, EmbeddingPlan, Fp4QuantKind,
    Fp4QuantPlan, Fp8QuantKind, Fp8QuantPlan, GemmLoop, GemmPlan, LastRowsKind, LastRowsPlan,
    LinearRowsPlan, PoolGemvPlan, PromptPlan, RowsKernel, RowsPlan, SplitGemmPlan, THREE_ROWS,
};
use crate::contract::recording::Recording;
use crate::tile::attention::attention as attention_model;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::{DType, f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};

/// Elements of an embedding row a tile covers.
const EMBEDDING_TILE: usize = 256;

/// Elements one FP4 scale covers.
const FP4_GROUP: usize = 16;
/// Elements in one tile of the copy kernels.
const AUX_TILE: usize = 256;
/// Output tile of the small kernels.
const SMALL_TILE: [usize; 2] = [16, 32];

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
const fn pool_weight(buffer: Buffer<'_, Bf16>) -> Buffer<'_, bf16> {
    // SAFETY: the address and element count are unchanged and carry the caller's guarantee.
    unsafe { Buffer::new(buffer.address(), buffer.len()) }
}

/// The weight's storage under cuTile's element types, with its memory retained by the graph.
enum Weight {
    Dense(Tensor<bf16>),
    Fp8(Tensor<f8e4m3fn>, Tensor<f32>),
    Fp4(Tensor<f4e2m1fnx2>, Tensor<f8e4m3fn>, f32),
}

impl TileProjection {
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

    fn matrix<T: DType>(
        &self,
        recording: &Recording<'_>,
        buffer: Buffer<'_, T>,
        name: &str,
        (rows, columns): (usize, usize),
    ) -> Result<Tensor<T>> {
        recording.retain(buffer);
        borrowed(
            buffer.require(name, rows * columns)?,
            self.ordinal,
            &[rows, columns],
        )
    }

    /// `[outputs, columns]` storage; FP4 packs code pairs and shares scales across a group.
    fn weight(
        &self,
        recording: &Recording<'_>,
        weight: WeightRef<'_>,
        (outputs, columns): (usize, usize),
    ) -> Result<Weight> {
        Ok(match weight {
            WeightRef::Dense(w) => Weight::Dense(self.matrix(
                recording,
                retyped_bf16(w),
                "dense weight",
                (outputs, columns),
            )?),
            WeightRef::Fp8 { weight, scales } => Weight::Fp8(
                self.matrix(
                    recording,
                    retyped_fp8(weight),
                    "FP8 weight",
                    (outputs, columns),
                )?,
                self.flat(recording, scales)?,
            ),
            WeightRef::Fp4 {
                weight,
                scales,
                inverse_global,
            } => Weight::Fp4(
                self.matrix(
                    recording,
                    retyped_fp4(weight),
                    "FP4 weight",
                    (outputs, columns / 2),
                )?,
                self.matrix(
                    recording,
                    retyped_fp8(scales),
                    "FP4 scales",
                    (outputs, columns / FP4_GROUP),
                )?,
                inverse_global,
            ),
        })
    }
}

/// The same memory, typed for cuTile.
#[expect(
    unsafe_code,
    reason = "Each marker is declared for exactly this element's size and layout"
)]
pub(in crate::tile) const fn retyped_bf16(buffer: Buffer<'_, Bf16>) -> Buffer<'_, bf16> {
    // SAFETY: `Bf16` marks `bf16` memory.
    unsafe { buffer.retyped() }
}

/// The same memory, typed for cuTile.
#[expect(
    unsafe_code,
    reason = "Each marker is declared for exactly this element's size and layout"
)]
pub(in crate::tile) const fn retyped_fp8(buffer: Buffer<'_, Fp8>) -> Buffer<'_, f8e4m3fn> {
    // SAFETY: `Fp8` marks `f8e4m3fn` memory.
    unsafe { buffer.retyped() }
}

/// The same memory, typed for cuTile.
#[expect(
    unsafe_code,
    reason = "Each marker is declared for exactly this element's size and layout"
)]
const fn retyped_fp4(buffer: Buffer<'_, Fp4Pair>) -> Buffer<'_, f4e2m1fnx2> {
    // SAFETY: `Fp4Pair` marks `f4e2m1fnx2` memory.
    unsafe { buffer.retyped() }
}

impl Projection for TileProjection {
    fn dense(
        &self,
        recording: &Recording<'_>,
        plan: &DenseRowsPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        w: Buffer<'_, Bf16>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let DenseRowsPlan {
            kernel,
            rows,
            outputs,
            columns,
        } = *plan;
        let mut out = self.matrix(recording, out, "dense output", (rows, outputs))?;
        let x = self.matrix(recording, x, "dense input", (rows, columns))?;
        let w = self.matrix(
            recording,
            retyped_bf16(w),
            "dense weight",
            (outputs, columns),
        )?;
        let element = || -> String { bf16::DTYPE.as_str().into() };
        let tiled = |rows: usize, tile_columns: usize| {
            vec![
                element(),
                rows.to_string(),
                tile_columns.to_string(),
                columns.to_string(),
            ]
        };
        match kernel {
            DenseKernel::Gemv {
                tile_rows,
                tile_columns,
            } => scope
                .record(
                    pool_gemv::dense((&mut out).partition([1, tile_rows]), &x, &w)
                        .generics(tiled(tile_rows, tile_columns)),
                )
                .map(drop),
            DenseKernel::Shared {
                group,
                tile_rows,
                tile_columns,
            } => scope
                .record(
                    linear_batch::dense((&mut out).partition([group, tile_rows]), &x, &w)
                        .generics(tiled(tile_rows, tile_columns)),
                )
                .map(drop),
            DenseKernel::Native => scope
                .record(
                    small_gemm::dense((&mut out).partition(SMALL_TILE), &x, &w)
                        .generics(vec![element(), columns.to_string()]),
                )
                .map(drop),
            DenseKernel::NativeWide {
                rows: tile_rows,
                width,
            } => scope
                .record(
                    small_gemm::dense_wide((&mut out).partition([tile_rows, width]), &x, &w)
                        .generics(vec![element(), width.to_string(), columns.to_string()]),
                )
                .map(drop),
            DenseKernel::PromptTile {
                rows: tile_rows,
                outputs: tile_outputs,
            } => scope
                .record(
                    prefill_gemm::dense_tile(
                        (&mut out).partition([tile_rows, tile_outputs]),
                        &x,
                        &w,
                    )
                    .generics(vec![
                        tile_rows.to_string(),
                        tile_outputs.to_string(),
                        columns.to_string(),
                    ]),
                )
                .map(drop),
            DenseKernel::Prompt { tile } => scope
                .record(
                    prefill_gemm::dense((&mut out).partition(tile), &x, &w)
                        .generics(vec![columns.to_string()]),
                )
                .map(drop),
            DenseKernel::PromptNarrow { tile } => scope
                .record(
                    prefill_gemm::dense_n16((&mut out).partition(tile), &x, &w)
                        .generics(vec![columns.to_string()]),
                )
                .map(drop),
        }
        .map_err(backend_error)
    }

    fn embedding_batch(
        &self,
        recording: &Recording<'_>,
        plan: &EmbeddingBatchPlan,
        out: Buffer<'_, f32>,
        weight: Buffer<'_, Bf16>,
        external: Option<Buffer<'_, f32>>,
        metadata: Buffer<'_, i32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let EmbeddingBatchPlan {
            dimension,
            fields,
            tile,
        } = *plan;
        let rows = out.len() / dimension;
        let mut out = self.tensor(recording, out, "embedding rows", &[rows, dimension])?;
        let weight = retyped_bf16(weight);
        let weight = self.tensor(
            recording,
            weight,
            "embedding weight",
            &[weight.len() / dimension, dimension],
        )?;
        let metadata = self.flat(recording, metadata)?;
        let generics = vec![dimension.to_string(), fields.to_string()];
        let tiles = (&mut out).partition([1, tile]);
        if let Some(external) = external {
            let external = self.tensor(recording, external, "external", &[rows, dimension])?;
            scope
                .record(
                    attention_model::embedding_rows(tiles, &weight, &external, &metadata)
                        .generics(generics),
                )
                .map_err(backend_error)?;
        } else {
            scope
                .record(
                    attention_model::embedding_batch(tiles, &weight, &metadata).generics(generics),
                )
                .map_err(backend_error)?;
        }
        Ok(())
    }

    fn linear_rows(
        &self,
        recording: &Recording<'_>,
        plan: &LinearRowsPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        w: Buffer<'_, Bf16>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let LinearRowsPlan {
            rows,
            row_tile,
            output_tile,
            depth,
            columns,
        } = *plan;
        let mut out = self.tensor(recording, out, "linear output", &[rows, out.len() / rows])?;
        let x = self.tensor(recording, x, "linear input", &[rows, columns])?;
        let w = retyped_bf16(w);
        let w = self.tensor(recording, w, "linear weight", &[w.len() / columns, columns])?;
        scope
            .record(
                linear_batch::dense((&mut out).partition([row_tile, output_tile]), &x, &w)
                    .generics(vec![
                        bf16::DTYPE.as_str().into(),
                        output_tile.to_string(),
                        depth.to_string(),
                        columns.to_string(),
                    ]),
            )
            .map_err(backend_error)?;
        Ok(())
    }

    fn embedding(
        &self,
        recording: &Recording<'_>,
        plan: &EmbeddingPlan,
        out: Buffer<'_, f32>,
        weight: Buffer<'_, Bf16>,
        external: Buffer<'_, f32>,
        metadata: Buffer<'_, i32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let EmbeddingPlan {
            dimension,
            external: allow_external,
        } = *plan;
        let mut out = self.flat(recording, out)?;
        let vocabulary = weight.len() / dimension.max(1);
        let weight = self.matrix(
            recording,
            retyped_bf16(weight),
            "embedding weight",
            (vocabulary, dimension),
        )?;
        let external = self.flat(recording, external)?;
        let metadata = self.flat(recording, metadata)?;
        scope
            .record(
                attention_model::embedding(
                    (&mut out).partition([EMBEDDING_TILE]),
                    &weight,
                    &external,
                    &metadata,
                )
                .generics(vec![
                    dimension.to_string(),
                    u8::from(allow_external).to_string(),
                ]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn dense_pair(
        &self,
        recording: &Recording<'_>,
        plan: &PromptPlan,
        (first, second): (Buffer<'_, f32>, Buffer<'_, f32>),
        x: Buffer<'_, f32>,
        (weight_first, weight_second): (Buffer<'_, Bf16>, Buffer<'_, Bf16>),
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PromptPlan {
            rows,
            outputs,
            columns,
            tile,
        } = *plan;
        let mut first = self.matrix(recording, first, "pair output", (rows, outputs))?;
        let mut second = self.matrix(recording, second, "pair output", (rows, outputs))?;
        let x = self.matrix(recording, x, "pair input", (rows, columns))?;
        let weight_first = self.matrix(
            recording,
            retyped_bf16(weight_first),
            "pair weight",
            (outputs, columns),
        )?;
        let weight_second = self.matrix(
            recording,
            retyped_bf16(weight_second),
            "pair weight",
            (outputs, columns),
        )?;
        scope
            .record(
                prefill_gemm::dense_pair(
                    (&mut first).partition(tile),
                    (&mut second).partition(tile),
                    &x,
                    &weight_first,
                    &weight_second,
                )
                .generics(vec![columns.to_string()]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn prompt_fp8(
        &self,
        recording: &Recording<'_>,
        plan: &PromptPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PromptPlan {
            rows,
            outputs,
            columns,
            tile,
        } = *plan;
        let mut out = self.matrix(recording, out, "prompt output", (rows, outputs))?;
        let x = self.matrix(recording, x, "prompt input", (rows, columns))?;
        let Weight::Fp8(w, s) = self.weight(recording, weight, (outputs, columns))? else {
            return Err(Error::invalid("prompt FP8 GEMM takes an FP8 weight"));
        };
        scope
            .record(
                prefill_gemm::fp8((&mut out).partition(tile), &x, &w, &s)
                    .generics(vec![columns.to_string()]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn prompt_fp4_split(
        &self,
        recording: &Recording<'_>,
        plan: &SplitGemmPlan,
        out: Buffer<'_, f32>,
        partial: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let PromptPlan {
            rows,
            outputs,
            columns,
            tile,
        } = plan.prompt;
        let mut out = self.matrix(recording, out, "prompt output", (rows, outputs))?;
        let mut partial = self.matrix(recording, partial, "prompt partials", (rows, outputs))?;
        let x = self.matrix(recording, x, "prompt input", (rows, columns))?;
        let Weight::Fp4(w, s, inverse_global) =
            self.weight(recording, weight, (outputs, columns))?
        else {
            return Err(Error::invalid("split FP4 GEMM takes an FP4 weight"));
        };
        scope
            .record(
                prefill_gemm::nvfp4_split((&mut partial).partition(tile), &x, &w, &s, plan.k_tiles)
                    .generics(vec![columns.to_string()]),
            )
            .map_err(backend_error)?;
        scope
            .record(prefill_gemm::reduce_split(
                (&mut out).partition(tile),
                &partial,
                inverse_global,
                plan.splits,
            ))
            .map(drop)
            .map_err(backend_error)
    }

    fn quantize_fp4(
        &self,
        recording: &Recording<'_>,
        plan: &Fp4QuantPlan,
        codes: Buffer<'_, Fp4Pair>,
        scales: Buffer<'_, Fp8>,
        input: Buffer<'_, f32>,
        scale: f32,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let Fp4QuantPlan { rows, columns, .. } = *plan;
        let mut codes = self.matrix(
            recording,
            retyped_fp4(codes),
            "FP4 codes",
            (rows, columns / 2),
        )?;
        let mut scales = self.matrix(
            recording,
            retyped_fp8(scales),
            "FP4 scales",
            (rows, columns / FP4_GROUP),
        )?;
        let input = self.matrix(recording, input, "FP4 input", (rows, columns))?;
        let generics = vec![columns.to_string()];
        let (codes, scales) = (
            (&mut codes).partition(plan.codes_tile),
            (&mut scales).partition(plan.scales_tile),
        );
        match plan.kind {
            Fp4QuantKind::Whole => scope
                .record(nvfp4_gemm::quantize(codes, scales, &input, scale).generics(generics))
                .map(drop),
            Fp4QuantKind::RowGroups { .. } => scope
                .record(nvfp4_gemm::quantize_rows(codes, scales, &input, scale).generics(generics))
                .map(drop),
        }
        .map_err(backend_error)
    }

    fn quantize_swiglu(
        &self,
        recording: &Recording<'_>,
        plan: &Fp4QuantPlan,
        codes: Buffer<'_, Fp4Pair>,
        scales: Buffer<'_, Fp8>,
        (gate, up): (Buffer<'_, f32>, Buffer<'_, f32>),
        scale: f32,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let Fp4QuantPlan { rows, columns, .. } = *plan;
        let mut codes = self.matrix(
            recording,
            retyped_fp4(codes),
            "FP4 codes",
            (rows, columns / 2),
        )?;
        let mut scales = self.matrix(
            recording,
            retyped_fp8(scales),
            "FP4 scales",
            (rows, columns / FP4_GROUP),
        )?;
        let gate = self.matrix(recording, gate, "gate", (rows, columns))?;
        let up = self.matrix(recording, up, "up", (rows, columns))?;
        scope
            .record(
                nvfp4_gemm::swiglu_quantize(
                    (&mut codes).partition(plan.codes_tile),
                    (&mut scales).partition(plan.scales_tile),
                    &gate,
                    &up,
                    scale,
                )
                .generics(vec![columns.to_string()]),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn gemm_fp4(
        &self,
        recording: &Recording<'_>,
        plan: &GemmPlan,
        out: Buffer<'_, f32>,
        (codes, scales): (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        weight: WeightRef<'_>,
        inverse_scale: f32,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let GemmPlan {
            rows,
            outputs,
            columns,
            tile,
            ..
        } = *plan;
        let mut out = self.matrix(recording, out, "GEMM output", (rows, outputs))?;
        let codes = self.matrix(
            recording,
            retyped_fp4(codes),
            "FP4 codes",
            (rows, columns / 2),
        )?;
        let scales = self.matrix(
            recording,
            retyped_fp8(scales),
            "FP4 scales",
            (rows, columns / FP4_GROUP),
        )?;
        let Weight::Fp4(w, s, _) = self.weight(recording, weight, (outputs, columns))? else {
            return Err(Error::invalid("FP4 GEMM takes an FP4 weight"));
        };
        let generics = vec![
            columns.to_string(),
            tile[0].to_string(),
            tile[1].to_string(),
        ];
        let out = (&mut out).partition(tile);
        match plan.main_loop {
            GemmLoop::Plain => scope
                .record(
                    nvfp4_gemm::packed(out, &codes, &scales, &w, &s, inverse_scale)
                        .generics(generics),
                )
                .map(drop),
            GemmLoop::Prefetch => scope
                .record(
                    nvfp4_gemm::packed_prefill(out, &codes, &scales, &w, &s, inverse_scale)
                        .generics(generics),
                )
                .map(drop),
        }
        .map_err(backend_error)
    }

    fn gemm_swiglu(
        &self,
        recording: &Recording<'_>,
        plan: &GemmPlan,
        (out_codes, out_scales): (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        (codes, scales): (Buffer<'_, Fp4Pair>, Buffer<'_, Fp8>),
        (gate, up): (WeightRef<'_>, WeightRef<'_>),
        (inverse_gate, inverse_up, down_scale): (f32, f32, f32),
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let GemmPlan {
            rows,
            outputs,
            columns,
            tile: [m, n],
            ..
        } = *plan;
        let mut out_codes = self.matrix(
            recording,
            retyped_fp4(out_codes),
            "SwiGLU codes",
            (rows, outputs / 2),
        )?;
        let mut out_scales = self.matrix(
            recording,
            retyped_fp8(out_scales),
            "SwiGLU scales",
            (rows, outputs / FP4_GROUP),
        )?;
        let codes = self.matrix(
            recording,
            retyped_fp4(codes),
            "FP4 codes",
            (rows, columns / 2),
        )?;
        let scales = self.matrix(
            recording,
            retyped_fp8(scales),
            "FP4 scales",
            (rows, columns / FP4_GROUP),
        )?;
        let (Weight::Fp4(gate, gate_scales, _), Weight::Fp4(up, up_scales, _)) = (
            self.weight(recording, gate, (outputs, columns))?,
            self.weight(recording, up, (outputs, columns))?,
        ) else {
            return Err(Error::invalid("fused SwiGLU takes FP4 weights"));
        };
        scope
            .record(
                nvfp4_gemm::packed_swiglu(
                    (&mut out_codes).partition([m, n / 2]),
                    (&mut out_scales).partition([m, n / FP4_GROUP]),
                    &codes,
                    &scales,
                    &gate,
                    &gate_scales,
                    &up,
                    &up_scales,
                    inverse_gate,
                    inverse_up,
                    down_scale,
                )
                .generics(
                    [columns, m, n, n / 2, n / FP4_GROUP]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )
            .map(drop)
            .map_err(backend_error)
    }

    fn quantize_fp8(
        &self,
        recording: &Recording<'_>,
        plan: &Fp8QuantPlan,
        codes: Buffer<'_, Fp8>,
        scales: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let Fp8QuantPlan {
            kind,
            rows,
            columns,
        } = *plan;
        let mut codes = self.matrix(recording, retyped_fp8(codes), "FP8 codes", (rows, columns))?;
        let input = self.matrix(recording, input, "FP8 input", (rows, columns))?;
        let scale_columns = match kind {
            Fp8QuantKind::Block { block } => columns / block,
            _ => 1,
        };
        let mut scales = self.matrix(recording, scales, "FP8 scales", (rows, scale_columns))?;
        let pad = columns.next_power_of_two();
        match kind {
            Fp8QuantKind::Whole => scope
                .record(
                    fp8_gemm::quantize(
                        (&mut codes).partition([1, pad]),
                        (&mut scales).partition([1, 1]),
                        &input,
                    )
                    .generics(vec![columns.to_string(), pad.to_string()]),
                )
                .map(drop),
            Fp8QuantKind::Block { block } => scope
                .record(
                    fp8_gemm::quantize_block(
                        (&mut codes).partition([1, block]),
                        (&mut scales).partition([1, 1]),
                        &input,
                    )
                    .generics(vec![columns.to_string(), block.to_string()]),
                )
                .map(drop),
            Fp8QuantKind::Parallel {
                conversion_columns,
                maximum_ready,
            } => {
                if !maximum_ready {
                    scope
                        .record(
                            fp8_quant_reduction::maximum((&mut scales).partition([1, 1]), &input)
                                .generics(vec![columns.to_string(), pad.to_string()]),
                        )
                        .map_err(backend_error)?;
                }
                scope
                    .record(
                        fp8_quant_reduction::quantize(
                            (&mut codes).partition([1, conversion_columns]),
                            &scales,
                            &input,
                        )
                        .generics(vec![columns.to_string()]),
                    )
                    .map(drop)
            }
        }
        .map_err(backend_error)
    }

    fn gemm_fp8(
        &self,
        recording: &Recording<'_>,
        plan: &GemmPlan,
        out: Buffer<'_, f32>,
        (codes, scales): (Buffer<'_, Fp8>, Buffer<'_, f32>),
        weight: WeightRef<'_>,
        block: Option<usize>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let GemmPlan {
            rows,
            outputs,
            columns,
            tile,
            ..
        } = *plan;
        let WeightRef::Fp8 {
            weight,
            scales: weight_scales,
        } = weight
        else {
            return Err(Error::invalid("FP8 GEMM takes an FP8 weight"));
        };
        let mut out = self.matrix(recording, out, "GEMM output", (rows, outputs))?;
        let codes = self.matrix(recording, retyped_fp8(codes), "FP8 codes", (rows, columns))?;
        let w = self.matrix(
            recording,
            retyped_fp8(weight),
            "FP8 weight",
            (outputs, columns),
        )?;
        let generics = vec![
            columns.to_string(),
            tile[0].to_string(),
            tile[1].to_string(),
        ];
        let out = (&mut out).partition(tile);
        if let Some(block) = block {
            let groups = columns / block;
            let xs = self.matrix(recording, scales, "FP8 scales", (rows, groups))?;
            let ws = self.matrix(
                recording,
                weight_scales,
                "FP8 weight scales",
                (outputs, groups),
            )?;
            scope
                .record(fp8_gemm::matmul_block(out, &codes, &w, &xs, &ws).generics(generics))
                .map(drop)
        } else {
            let xs = self.matrix(recording, scales, "FP8 scales", (rows, 1))?;
            let ws = self.flat(recording, weight_scales)?;
            scope
                .record(fp8_gemm::matmul(out, &codes, &w, &xs, &ws).generics(generics))
                .map(drop)
        }
        .map_err(backend_error)
    }

    fn rows(
        &self,
        recording: &Recording<'_>,
        plan: &RowsPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let RowsPlan {
            kernel,
            rows,
            outputs,
            columns,
        } = *plan;
        let mut out = self.matrix(recording, out, "projection output", (rows, outputs))?;
        let x = self.matrix(recording, x, "projection input", (rows, columns))?;
        let weight = self.weight(recording, weight, (outputs, columns))?;
        match kernel {
            RowsKernel::Small => {
                let tile = (&mut out).partition(SMALL_TILE);
                match &weight {
                    Weight::Dense(w) => scope
                        .record(
                            small_gemm::dense(tile, &x, w)
                                .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
                        )
                        .map(drop),
                    Weight::Fp8(w, s) => {
                        scope
                            .record(small_gemm::scaled(tile, &x, w, s).generics(vec![
                                f8e4m3fn::DTYPE.as_str().into(),
                                columns.to_string(),
                            ]))
                            .map(drop)
                    }
                    Weight::Fp4(..) => {
                        return Err(Error::unsupported("the small kernel has no FP4 form"));
                    }
                }
            }
            RowsKernel::Tiled {
                output_tile,
                tile_output,
                tile_columns,
                fp4_depths,
            } => {
                let tile = (&mut out).partition(output_tile);
                let mut generics = vec![
                    tile_output.to_string(),
                    tile_columns.to_string(),
                    columns.to_string(),
                ];
                match &weight {
                    Weight::Dense(w) => {
                        generics.insert(0, bf16::DTYPE.as_str().into());
                        scope
                            .record(linear_batch::dense(tile, &x, w).generics(generics))
                            .map(drop)
                    }
                    Weight::Fp8(w, s) => scope
                        .record(linear_batch::fp8(tile, &x, w, s).generics(generics))
                        .map(drop),
                    Weight::Fp4(w, s, inverse) => {
                        generics.extend(fp4_depths.map(|depth| depth.to_string()));
                        scope
                            .record(
                                linear_batch::nvfp4(tile, &x, w, s, *inverse).generics(generics),
                            )
                            .map(drop)
                    }
                }
            }
        }
        .map_err(backend_error)
    }

    fn row(
        &self,
        recording: &Recording<'_>,
        plan: &DownPlan,
        out: Buffer<'_, f32>,
        x: Buffer<'_, f32>,
        weight: WeightRef<'_>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let columns = plan.columns;
        let outputs = out.len();
        let mut out = self.flat(recording, out)?;
        let x = self.flat(recording, x)?;
        let weight = self.weight(recording, weight, (outputs, columns))?;
        let tile = (&mut out).partition([plan.tile_rows]);
        let mut generics = vec![
            plan.tile_rows.to_string(),
            plan.tile_columns.to_string(),
            columns.to_string(),
        ];
        match &weight {
            Weight::Dense(w) => {
                generics.insert(0, bf16::DTYPE.as_str().into());
                scope
                    .record(linear::dense(tile, &x, w).generics(generics))
                    .map(drop)
            }
            Weight::Fp8(w, s) => scope
                .record(linear::fp8(tile, &x, w, s).generics(generics))
                .map(drop),
            Weight::Fp4(w, s, inverse) => {
                generics.extend([
                    (plan.tile_columns / 2).to_string(),
                    (plan.tile_columns / FP4_GROUP).to_string(),
                ]);
                scope
                    .record(linear::nvfp4(tile, &x, w, s, *inverse).generics(generics))
                    .map(drop)
            }
        }
        .map_err(backend_error)
    }

    fn pack_row(
        &self,
        recording: &Recording<'_>,
        shape: (usize, usize),
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
        row: usize,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.matrix(recording, out, "packed rows", shape)?;
        let input = self.flat(recording, input)?;
        let row = i32::try_from(row).map_err(|_| geometry("row"))?;
        scope
            .record(linear_batch::pack_row(
                (&mut out).partition([1, AUX_TILE]),
                &input,
                row,
            ))
            .map_err(backend_error)?;
        Ok(())
    }

    fn unpack_row(
        &self,
        recording: &Recording<'_>,
        shape: (usize, usize),
        out: Buffer<'_, f32>,
        input: Buffer<'_, f32>,
        row: usize,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.flat(recording, out)?;
        let input = self.matrix(recording, input, "packed rows", shape)?;
        let row = i32::try_from(row).map_err(|_| geometry("row"))?;
        scope
            .record(linear_batch::unpack_row(
                (&mut out).partition([AUX_TILE]),
                &input,
                row,
            ))
            .map_err(backend_error)?;
        Ok(())
    }

    fn pack_three(
        &self,
        recording: &Recording<'_>,
        columns: usize,
        out: Buffer<'_, f32>,
        inputs: [Buffer<'_, f32>; THREE_ROWS],
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let mut out = self.matrix(recording, out, "packed rows", (THREE_ROWS, columns))?;
        let [a, b, c] = inputs;
        let (a, b, c) = (
            self.flat(recording, a)?,
            self.flat(recording, b)?,
            self.flat(recording, c)?,
        );
        scope
            .record(linear_batch::pack_inputs(
                (&mut out).partition([1, AUX_TILE]),
                &a,
                &b,
                &c,
            ))
            .map_err(backend_error)?;
        Ok(())
    }

    fn unpack_three(
        &self,
        recording: &Recording<'_>,
        columns: usize,
        outs: [Buffer<'_, f32>; THREE_ROWS],
        input: Buffer<'_, f32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let [a, b, c] = outs;
        let (mut a, mut b, mut c) = (
            self.flat(recording, a)?,
            self.flat(recording, b)?,
            self.flat(recording, c)?,
        );
        let input = self.matrix(recording, input, "packed rows", (THREE_ROWS, columns))?;
        scope
            .record(linear_batch::unpack_outputs(
                (&mut a).partition([AUX_TILE]),
                (&mut b).partition([AUX_TILE]),
                (&mut c).partition([AUX_TILE]),
                &input,
            ))
            .map_err(backend_error)?;
        Ok(())
    }

    fn last_rows(
        &self,
        recording: &Recording<'_>,
        plan: &LastRowsPlan,
        out: Buffer<'_, f32>,
        rows: Buffer<'_, f32>,
        metadata: Buffer<'_, i32>,
    ) -> Result<()> {
        let scope = capture_scope(recording)?;
        let LastRowsPlan {
            kind,
            lanes,
            hidden,
        } = *plan;
        let picked = out
            .len()
            .checked_div(hidden)
            .ok_or_else(|| Error::invalid("last rows hidden width"))?;
        let mut out = self.matrix(recording, out, "last rows", (picked, hidden))?;
        let rows = self.matrix(recording, rows, "prompt rows", (lanes, hidden))?;
        let metadata = self.flat(recording, metadata)?;
        let tile = (&mut out).partition([1, AUX_TILE]);
        match kind {
            LastRowsKind::One => scope
                .record(
                    prompt_logits::last_row(tile, &rows, &metadata)
                        .generics(vec![hidden.to_string()]),
                )
                .map(drop),
            LastRowsKind::Segments { segment_rows } => scope
                .record(
                    prompt_logits::last_rows(tile, &rows, &metadata)
                        .generics(vec![hidden.to_string(), segment_rows.to_string()]),
                )
                .map(drop),
        }
        .map_err(backend_error)
    }

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
        let w = borrowed(pool_weight(w), self.ordinal, &[outputs, columns])?;
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

fn geometry(field: &str) -> Error {
    Error::invalid(format!("projection {field} exceeds i32"))
}
