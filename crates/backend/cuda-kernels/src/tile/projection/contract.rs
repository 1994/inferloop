//! cuTile implementation of the projection family.
use super::{linear, linear_batch, pool_gemv, prompt_logits, small_gemm};
use crate::contract::buffer::{Bf16, Buffer, Fp4Pair, Fp8};
use crate::contract::kernels::{Projection, WeightRef};
use crate::contract::plan::{
    DownPlan, LastRowsKind, LastRowsPlan, PoolGemvPlan, RowsKernel, RowsPlan, THREE_ROWS,
};
use crate::contract::recording::Recording;
use crate::tile::{backend_error, borrowed, capture_scope};
use cuda_core::{DType, f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};

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
const fn retyped_bf16(buffer: Buffer<'_, Bf16>) -> Buffer<'_, bf16> {
    // SAFETY: `Bf16` marks `bf16` memory.
    unsafe { buffer.retyped() }
}

/// The same memory, typed for cuTile.
#[expect(
    unsafe_code,
    reason = "Each marker is declared for exactly this element's size and layout"
)]
const fn retyped_fp8(buffer: Buffer<'_, Fp8>) -> Buffer<'_, f8e4m3fn> {
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
