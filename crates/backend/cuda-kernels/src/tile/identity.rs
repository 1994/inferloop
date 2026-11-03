//! Identities of candidate kernels, which tuning results are keyed by.
//!
//! An identity is cuTile's compiled-kernel cache key over meta tensors, so it changes exactly when
//! the kernel or its compilation does, and no device memory is touched.
use super::backend_error;
use super::projection::{
    fp8_gemm, linear_batch, nvfp4_gemm, pool_gemv, prefill_gemm, small_gemm, vendor_fp8,
};
use super::recurrent::recurrent_prefill;
use cuda_core::{DType, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::Result;
use std::sync::Arc;

/// Elements of a recurrent candidate's metadata.
const METADATA_LEN: usize = 4;
/// Key axes a recurrent identity covers: key heads, value heads, channels and lanes.
const RECURRENT_RANK: usize = 4;
/// Axes of a projection shape: rows, outputs and columns.
const GEMM_RANK: usize = 3;

/// The prompt recurrence and its transpose at each block split.
/// # Errors
/// Reports a failed lowering.
pub fn recurrent_wy(
    stream: &Arc<Stream>,
    shape: [usize; RECURRENT_RANK],
    splits: &[usize],
) -> Result<Vec<String>> {
    let [kh, vh, dim, lanes] = shape;
    let qkv = api::meta::<f32>(&[lanes, 2 * kh + vh, dim])
        .sync_on(stream)
        .map_err(backend_error)?;
    let beta = api::meta::<f32>(&[lanes, vh])
        .sync_on(stream)
        .map_err(backend_error)?;
    let alpha = api::meta::<f32>(&[lanes, vh])
        .sync_on(stream)
        .map_err(backend_error)?;
    let a_log = api::meta::<f32>(&[vh])
        .sync_on(stream)
        .map_err(backend_error)?;
    let bias = api::meta::<f32>(&[vh])
        .sync_on(stream)
        .map_err(backend_error)?;
    let metadata = api::meta::<i32>(&[METADATA_LEN])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut state = api::meta::<f32>(&[vh, dim, dim])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut scratch = api::meta::<f32>(&[vh, lanes, dim])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut output = api::meta::<f32>(&[lanes, vh, dim])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut identities = Vec::new();
    for split in splits.iter().copied() {
        let block = dim / split;
        identities.push(
            recurrent_prefill::delta_wy(
                (&mut state).partition([1, dim, block]),
                (&mut scratch).partition([1, lanes, block]),
                &qkv,
                &beta,
                &alpha,
                &a_log,
                &bias,
                &metadata,
            )
            .generics(vec![
                kh.to_string(),
                vh.to_string(),
                dim.to_string(),
                lanes.to_string(),
                block.to_string(),
            ])
            .l2_cache_key_on(stream)
            .map_err(backend_error)?,
        );
    }
    identities.push(
        recurrent_prefill::transpose((&mut output).partition([1, 1, dim]), &scratch)
            .generics(vec![dim.to_string()])
            .l2_cache_key_on(stream)
            .map_err(backend_error)?,
    );
    if identities.iter().any(String::is_empty) {
        return Err(backend_error("incomplete WY kernel identity"));
    }
    Ok(identities)
}

/// The FP8 quantizer, every candidate tile of its product, and the vendor path's scale.
/// # Errors
/// Reports a failed lowering.
pub fn fp8_candidates(
    stream: &Arc<Stream>,
    shape: [usize; GEMM_RANK],
    row_tiles: &[usize],
    column_tiles: &[usize],
    aux_tile: usize,
) -> Result<Vec<String>> {
    let [rows, outputs, columns] = shape;
    let input = api::meta::<f32>(&[rows, columns])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut codes = api::meta::<f8e4m3fn>(&[rows, columns])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut xs = api::meta::<f32>(&[rows, 1])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight_codes = api::meta::<f8e4m3fn>(&[outputs, columns])
        .sync_on(stream)
        .map_err(backend_error)?;
    let ws = api::meta::<f32>(&[outputs])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut out = api::meta::<f32>(&[rows, outputs])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut ids = vec![
        fp8_gemm::quantize(
            (&mut codes).partition([1, columns.next_power_of_two()]),
            (&mut xs).partition([1, 1]),
            &input,
        )
        .generics(vec![
            columns.to_string(),
            columns.next_power_of_two().to_string(),
        ])
        .l2_cache_key_on(stream)
        .map_err(backend_error)?,
    ];
    for tile_m in row_tiles.iter().copied() {
        for tile_n in column_tiles.iter().copied() {
            ids.push(
                fp8_gemm::matmul(
                    (&mut out).partition([tile_m, tile_n]),
                    &codes,
                    &weight_codes,
                    &xs,
                    &ws,
                )
                .generics(vec![
                    columns.to_string(),
                    tile_m.to_string(),
                    tile_n.to_string(),
                ])
                .l2_cache_key_on(stream)
                .map_err(backend_error)?,
            );
        }
    }
    ids.push(
        vendor_fp8::scale((&mut out).partition([1, aux_tile]), &xs, &ws)
            .l2_cache_key_on(stream)
            .map_err(backend_error)?,
    );
    Ok(ids)
}

/// Tiles an FP4 product identity covers.
#[derive(Clone, Copy, Debug)]
pub struct Fp4Search<'a> {
    pub group: usize,
    pub codes_tile: [usize; 2],
    pub scales_tile: [usize; 2],
    pub tile_m: &'a [usize],
    pub tile_n: &'a [usize],
}

/// The FP4 quantizer and every candidate tile and main loop of its product.
/// # Errors
/// Reports a failed lowering.
pub fn fp4_candidates(
    stream: &Arc<Stream>,
    shape: [usize; GEMM_RANK],
    search: Fp4Search<'_>,
) -> Result<Vec<String>> {
    let [m, n, k] = shape;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut packed = api::meta::<f4e2m1fnx2>(&[m, k / 2])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut scales = api::meta::<f8e4m3fn>(&[m, k / search.group])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight = api::meta::<f4e2m1fnx2>(&[n, k / 2])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight_scales = api::meta::<f8e4m3fn>(&[n, k / search.group])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut output = api::meta::<f32>(&[m, n])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut identities = vec![
        nvfp4_gemm::quantize(
            (&mut packed).partition(search.codes_tile),
            (&mut scales).partition(search.scales_tile),
            &input,
            1.0,
        )
        .generics(vec![k.to_string()])
        .l2_cache_key_on(stream)
        .map_err(backend_error)?,
    ];
    for tile_m in search.tile_m.iter().copied() {
        for tile_n in search.tile_n.iter().copied() {
            for prefill in [false, true] {
                let tile = [tile_m, tile_n];
                let generics = vec![k.to_string(), tile_m.to_string(), tile_n.to_string()];
                let identity = if prefill {
                    nvfp4_gemm::packed_prefill(
                        (&mut output).partition(tile),
                        &packed,
                        &scales,
                        &weight,
                        &weight_scales,
                        1.0,
                    )
                    .generics(generics)
                    .l2_cache_key_on(stream)
                } else {
                    nvfp4_gemm::packed(
                        (&mut output).partition(tile),
                        &packed,
                        &scales,
                        &weight,
                        &weight_scales,
                        1.0,
                    )
                    .generics(generics)
                    .l2_cache_key_on(stream)
                }
                .map_err(backend_error)?;
                identities.push(identity);
            }
        }
    }
    Ok(identities)
}

/// Every prompt-tile candidate of a dense projection.
/// # Errors
/// Reports a failed lowering.
pub fn dense_prompt_candidates(
    stream: &Arc<Stream>,
    shape: [usize; GEMM_RANK],
    tiles: &[[usize; 2]],
) -> Result<Vec<String>> {
    let [m, n, k] = shape;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight = api::meta::<bf16>(&[n, k])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut output = api::meta::<f32>(&[m, n])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut identities = Vec::new();
    for &[rows, outputs] in tiles {
        identities.push(
            prefill_gemm::dense_tile((&mut output).partition([rows, outputs]), &input, &weight)
                .generics(vec![rows.to_string(), outputs.to_string(), k.to_string()])
                .l2_cache_key_on(stream)
                .map_err(backend_error)?,
        );
    }
    Ok(identities)
}

/// Tiles a pooled dense identity covers.
#[derive(Clone, Copy, Debug)]
pub struct DenseSearch<'a> {
    pub tile_rows: usize,
    pub tile_columns: usize,
    pub shared_rows: usize,
    pub native_tile: [usize; 2],
    pub wide_rows: usize,
    /// Wider native tiles; the qualified family lists none, so its identity stays frozen.
    pub wide: &'a [usize],
}

/// The pooled dense candidates of a projection.
/// # Errors
/// Reports a failed lowering.
pub fn dense_candidates(
    stream: &Arc<Stream>,
    shape: [usize; GEMM_RANK],
    search: DenseSearch<'_>,
) -> Result<Vec<String>> {
    let [m, n, k] = shape;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight = api::meta::<bf16>(&[n, k])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut output = api::meta::<f32>(&[m, n])
        .sync_on(stream)
        .map_err(backend_error)?;
    let generics = vec![
        bf16::DTYPE.as_str().into(),
        search.tile_rows.to_string(),
        search.tile_columns.to_string(),
        k.to_string(),
    ];
    let gemv = pool_gemv::dense(
        (&mut output).partition([1, search.tile_rows]),
        &input,
        &weight,
    )
    .generics(generics.clone())
    .l2_cache_key_on(stream)
    .map_err(backend_error)?;
    let shared = linear_batch::dense(
        (&mut output).partition([search.shared_rows, search.tile_rows]),
        &input,
        &weight,
    )
    .generics(generics)
    .l2_cache_key_on(stream)
    .map_err(backend_error)?;
    let native = small_gemm::dense((&mut output).partition(search.native_tile), &input, &weight)
        .generics(vec![bf16::DTYPE.as_str().into(), k.to_string()])
        .l2_cache_key_on(stream)
        .map_err(backend_error)?;
    let mut identities = vec![gemv, shared, native];
    // Wider tiles are candidates only outside the qualified family, whose identity stays frozen.
    {
        for &width in search.wide {
            identities.push(
                small_gemm::dense_wide(
                    (&mut output).partition([search.wide_rows, width]),
                    &input,
                    &weight,
                )
                .generics(vec![
                    bf16::DTYPE.as_str().into(),
                    width.to_string(),
                    k.to_string(),
                ])
                .l2_cache_key_on(stream)
                .map_err(backend_error)?,
            );
        }
    }
    Ok(identities)
}
