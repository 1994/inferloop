//! Identities of candidate kernels, which tuning results are keyed by.
//!
//! An identity is cuTile's compiled-kernel cache key over meta tensors, so it changes exactly when
//! the kernel or its compilation does, and no device memory is touched.
use super::attention::resident_attention_prefill;
use super::backend_error;
use super::projection::{
    BLOCK_BYTES, entries, fp8_gemm, linear_batch, nvfp4_gemm, pool_gemv, prefill_gemm, small_gemm,
    swizzle, vendor_fp8,
};
use super::recurrent::recurrent_prefill;
use super::sampling::greedy;
use super::state::aux;
pub use crate::contract::{AttentionShape, DenseSearch, Fp4Search, SwigluSearch};
use crate::contract::{Fp4ScalePlan, GEMM_RANK, GreedyPlan, RECURRENT_RANK};
use crate::tile::runtime::Device;
use cuda_core::{DType, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::Result;
use std::sync::Arc;

/// Elements of a recurrent candidate's metadata.
const METADATA_LEN: usize = 4;

/// The prompt recurrence and its transpose at each block split.
/// # Errors
/// Reports a failed lowering.
pub fn recurrent_wy(
    device: &Device,
    shape: [usize; RECURRENT_RANK],
    splits: &[usize],
) -> Result<Vec<String>> {
    let stream = device.stream();
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
                f32::DTYPE.as_str().into(),
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
    device: &Device,
    shape: [usize; GEMM_RANK],
    row_tiles: &[usize],
    column_tiles: &[usize],
    aux_tile: usize,
) -> Result<Vec<String>> {
    let stream = device.stream();
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

/// The FP4 quantizer and every candidate tile and main loop of its product.
/// # Errors
/// Reports a failed lowering.
pub fn fp4_candidates(
    device: &Device,
    shape: [usize; GEMM_RANK],
    search: Fp4Search<'_>,
) -> Result<Vec<String>> {
    let stream = device.stream();
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
    device: &Device,
    shape: [usize; GEMM_RANK],
    tiles: &[[usize; 2]],
) -> Result<Vec<String>> {
    let stream = device.stream();
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

/// The pooled dense candidates of a projection.
/// # Errors
/// Reports a failed lowering.
pub fn dense_candidates(
    device: &Device,
    shape: [usize; GEMM_RANK],
    search: DenseSearch<'_>,
) -> Result<Vec<String>> {
    let stream = device.stream();
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

/// The row-grouped quantizer and every candidate tile of the fused `SwiGLU` product.
/// # Errors
/// Reports a failed lowering.
pub fn swiglu_candidates(
    device: &Device,
    shape: [usize; GEMM_RANK],
    search: SwigluSearch<'_>,
) -> Result<Vec<String>> {
    let stream = device.stream();
    let [m, n, k] = shape;
    let group = search.group;
    let mut q = api::meta::<f4e2m1fnx2>(&[m, k / 2])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut qs = api::meta::<f8e4m3fn>(&[m, k / group])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight = api::meta::<f4e2m1fnx2>(&[n, k / 2])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight_scales = api::meta::<f8e4m3fn>(&[n, k / group])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut codes = api::meta::<f4e2m1fnx2>(&[m, n / 2])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut scales = api::meta::<f8e4m3fn>(&[m, n / group])
        .sync_on(stream)
        .map_err(backend_error)?;
    let input = api::meta::<f32>(&[m, k])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut identities = vec![
        nvfp4_gemm::quantize_rows(
            (&mut q).partition(search.codes_tile),
            (&mut qs).partition(search.scales_tile),
            &input,
            1.0,
        )
        .generics(vec![k.to_string()])
        .l2_cache_key_on(stream)
        .map_err(backend_error)?,
    ];
    for &[rows, outputs] in search.tiles {
        identities.push(
            nvfp4_gemm::packed_swiglu(
                (&mut codes).partition([rows, outputs / 2]),
                (&mut scales).partition([rows, outputs / group]),
                &q,
                &qs,
                &weight,
                &weight_scales,
                &weight,
                &weight_scales,
                1.0,
                1.0,
                1.0,
            )
            .generics(
                [k, rows, outputs, outputs / 2, outputs / group]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            )
            .l2_cache_key_on(stream)
            .map_err(backend_error)?,
        );
    }
    Ok(identities)
}

/// The gated norm at each candidate occupancy; `None` is the compiler's default.
/// # Errors
/// Reports a failed lowering.
pub fn gated_norm_candidates(
    device: &Device,
    [rows, dimension]: [usize; 2],
    epsilon: f32,
    occupancies: &[Option<i32>],
) -> Result<Vec<String>> {
    let stream = device.stream();
    let meta = || {
        api::meta::<f32>(&[rows, dimension])
            .sync_on(stream)
            .map_err(backend_error)
    };
    let (x, gate, mut output) = (meta()?, meta()?, meta()?);
    let weight = api::meta::<f32>(&[dimension])
        .sync_on(stream)
        .map_err(backend_error)?;
    let x = x.view(&[rows, dimension]).map_err(backend_error)?;
    let gate = gate.view(&[rows, dimension]).map_err(backend_error)?;
    let mut identities = Vec::new();
    for occupancy in occupancies.iter().copied() {
        identities.push(
            aux::norm(
                (&mut output).partition([1, dimension]),
                &x,
                &weight,
                &gate,
                epsilon,
                0.0_f32,
            )
            .generics(vec![
                dimension.to_string(),
                dimension.to_string(),
                "1".into(),
            ])
            .compile_options(occupancy.map_or_else(CompileOptions::default, |occupancy| {
                CompileOptions::default().occupancy(occupancy)
            }))
            .l2_cache_key_on(stream)
            .map_err(backend_error)?,
        );
    }
    Ok(identities)
}

/// The native prepared recurrence: normalize, gates, precomputed delta and transpose.
/// # Errors
/// Reports a failed lowering or an empty identity.
pub fn exact_recurrence(
    device: &Device,
    shape: [usize; RECURRENT_RANK],
    value_columns: usize,
    metadata_fields: usize,
) -> Result<Vec<String>> {
    let stream = device.stream();
    let [kh, vh, dim, width] = shape;
    let meta = |extent: &[usize]| {
        api::meta::<f32>(extent)
            .sync_on(stream)
            .map_err(backend_error)
    };
    let qkv = meta(&[width, 2 * kh + vh, dim])?;
    let (beta, alpha) = (meta(&[width, vh])?, meta(&[width, vh])?);
    let (a_log, bias) = (meta(&[vh])?, meta(&[vh])?);
    let metadata = api::meta::<i32>(&[metadata_fields])
        .sync_on(stream)
        .map_err(backend_error)?;
    let mut normalized = meta(&[width, 2 * kh, dim])?;
    let (mut betas, mut decays) = (meta(&[width, vh])?, meta(&[width, vh])?);
    let mut state = meta(&[vh, dim, dim])?;
    let mut scratch = meta(&[vh, width, dim])?;
    let mut output = meta(&[width, vh, dim])?;
    let identities = vec![
        recurrent_prefill::normalize_shared((&mut normalized).partition([1, 1, dim]), &qkv)
            .generics(vec![kh.to_string(), dim.to_string()])
            .l2_cache_key_on(stream)
            .map_err(backend_error)?,
        recurrent_prefill::gates_shared(
            (&mut betas).partition([1, 1]),
            (&mut decays).partition([1, 1]),
            &beta,
            &alpha,
            &a_log,
            &bias,
        )
        .l2_cache_key_on(stream)
        .map_err(backend_error)?,
        recurrent_prefill::delta_precomputed(
            (&mut state).partition([1, dim, value_columns]),
            (&mut scratch).partition([1, width, value_columns]),
            &qkv,
            &normalized,
            &betas,
            &decays,
            &metadata,
        )
        .generics(vec![
            f32::DTYPE.as_str().into(),
            kh.to_string(),
            vh.to_string(),
            dim.to_string(),
            width.to_string(),
            value_columns.to_string(),
        ])
        .l2_cache_key_on(stream)
        .map_err(backend_error)?,
        recurrent_prefill::transpose((&mut output).partition([1, 1, dim]), &scratch)
            .generics(vec![dim.to_string()])
            .l2_cache_key_on(stream)
            .map_err(backend_error)?,
    ];
    if identities.iter().any(String::is_empty) {
        return Err(backend_error("incomplete exact recurrence identity"));
    }
    Ok(identities)
}

/// The prompt append and compute of KV element `E`; a query tile selects the tiled compute.
/// # Errors
/// Reports a failed lowering.
pub fn attention_candidates<E: DType>(
    device: &Device,
    shape: &AttentionShape,
    query_tile: Option<usize>,
) -> Result<Vec<String>> {
    let stream = device.stream();
    let AttentionShape {
        heads,
        kv_heads,
        dim,
        lanes,
        capacity,
        block,
        scales: [key_scale, value_scale],
        metadata_fields,
        ..
    } = *shape;
    let meta = |extent: &[usize]| {
        api::meta::<f32>(extent)
            .sync_on(stream)
            .map_err(backend_error)
    };
    let meta_i32 = |extent: &[usize]| {
        api::meta::<i32>(extent)
            .sync_on(stream)
            .map_err(backend_error)
    };
    let meta_kv = || {
        api::meta::<E>(&[kv_heads, capacity, dim])
            .sync_on(stream)
            .map_err(backend_error)
    };
    let query = meta(&[lanes, heads * dim])?;
    let (append_keys, append_values) = (
        meta(&[lanes, kv_heads, dim])?,
        meta(&[lanes, kv_heads, dim])?,
    );
    let metadata = meta_i32(&[metadata_fields])?;
    let table = meta_i32(&[capacity / block])?;
    let (mut keys, mut values) = (meta_kv()?, meta_kv()?);
    let element = || -> String { E::DTYPE.as_str().into() };
    let append = resident_attention_prefill::append(
        (&mut keys).partition([1, capacity, dim]),
        (&mut values).partition([1, capacity, dim]),
        &append_keys,
        &append_values,
        &metadata,
        &table,
        key_scale,
        value_scale,
    )
    .generics(vec![
        element(),
        dim.to_string(),
        capacity.to_string(),
        i32::from(E::DTYPE != f32::DTYPE).to_string(),
        block.to_string(),
        lanes.div_ceil(block).next_power_of_two().to_string(),
    ])
    .l2_cache_key_on(stream)
    .map_err(backend_error)?;
    let compute = attention_compute(
        stream,
        shape,
        query_tile,
        (&query, &keys, &values, &metadata, &table),
    )?;
    Ok(vec![append, compute])
}

/// Query, keys, values, metadata and block table.
type AttentionOperands<'a, E> = (
    &'a Tensor<f32>,
    &'a Tensor<E>,
    &'a Tensor<E>,
    &'a Tensor<i32>,
    &'a Tensor<i32>,
);

fn attention_compute<E: DType>(
    stream: &Arc<Stream>,
    shape: &AttentionShape,
    query_tile: Option<usize>,
    (query, keys, values, metadata, table): AttentionOperands<'_, E>,
) -> Result<String> {
    let AttentionShape {
        heads,
        kv_heads,
        dim,
        lanes,
        block,
        window,
        scales: [key_scale, value_scale],
        ..
    } = *shape;
    let window = i32::try_from(window).map_err(backend_error)?;
    let element = || -> String { E::DTYPE.as_str().into() };
    let meta = |extent: &[usize]| {
        api::meta::<f32>(extent)
            .sync_on(stream)
            .map_err(backend_error)
    };
    let group = heads / kv_heads;
    if let Some(tile) = query_tile {
        let mut output = meta(&[lanes, heads * dim])?;
        let scale = 1.0 / f32::from(u16::try_from(dim).map_err(backend_error)?).sqrt();
        resident_attention_prefill::decode_tiled(
            (&mut output).partition([tile, dim]),
            query,
            keys,
            values,
            metadata,
            table,
            window,
            key_scale,
            value_scale,
            scale,
        )
        .generics(vec![
            element(),
            dim.to_string(),
            group.to_string(),
            tile.to_string(),
            block.to_string(),
            block.to_string(),
            "1".into(),
        ])
        .l2_cache_key_on(stream)
    } else {
        let mut output = meta(&[lanes * heads, dim])?;
        resident_attention_prefill::decode(
            (&mut output).partition([1, dim]),
            &query.view(&[lanes * heads, dim]).map_err(backend_error)?,
            keys,
            values,
            metadata,
            table,
            window,
            key_scale,
            value_scale,
        )
        .generics(vec![
            element(),
            dim.to_string(),
            group.to_string(),
            heads.to_string(),
            block.to_string(),
        ])
        .l2_cache_key_on(stream)
    }
    .map_err(backend_error)
}

/// Both greedy reductions at their exact geometry.
/// # Errors
/// Reports a failed lowering.
pub fn greedy_reductions(device: &Device, plan: &GreedyPlan) -> Result<String> {
    let stream = device.stream();
    let GreedyPlan {
        rows,
        vocabulary,
        tile,
        blocks,
    } = *plan;
    let i32_partials = || {
        api::meta::<i32>(&[rows, blocks])
            .sync_on(stream)
            .map_err(backend_error)
    };
    let mut keys = api::meta::<u32>(&[rows, blocks])
        .sync_on(stream)
        .map_err(backend_error)?;
    let (mut indices, mut invalid) = (i32_partials()?, i32_partials()?);
    let mut tokens = api::meta::<i32>(&[rows, 2])
        .sync_on(stream)
        .map_err(backend_error)?;
    let logits = api::meta::<f32>(&[rows, vocabulary])
        .sync_on(stream)
        .map_err(backend_error)?;
    let first = greedy::partials(
        (&mut keys).partition([1, 1]),
        (&mut indices).partition([1, 1]),
        (&mut invalid).partition([1, 1]),
        &logits,
    )
    .generics(vec![vocabulary.to_string(), tile.to_string()])
    .l2_cache_key_on(stream)
    .map_err(backend_error)?;
    let second = greedy::finish((&mut tokens).partition([1, 2]), &keys, &indices, &invalid)
        .generics(vec![
            blocks.to_string(),
            blocks.next_power_of_two().to_string(),
            "0".into(),
            rows.to_string(),
        ])
        .l2_cache_key_on(stream)
        .map_err(backend_error)?;
    Ok(format!("{first}:{second}"))
}

/// The FP4 scale swizzle a vendor FP4 GEMM consumes, into an `out_len`-byte destination.
/// # Errors
/// Reports a plan that does not tile, or a failed lowering.
pub fn fp4_scale_swizzle(device: &Device, plan: &Fp4ScalePlan, out_len: usize) -> Result<String> {
    let stream = device.stream();
    let launch = swizzle(plan)?;
    let mut out = api::meta::<f8e4m3fn>(&[out_len / launch.row, launch.row])
        .sync_on(stream)
        .map_err(backend_error)?;
    let weight = api::meta::<f8e4m3fn>(&[plan.outputs, launch.groups])
        .sync_on(stream)
        .map_err(backend_error)?;
    let input = api::meta::<f8e4m3fn>(&[plan.rows, launch.groups])
        .sync_on(stream)
        .map_err(backend_error)?;
    entries::pack_pair(
        (&mut out).partition_prefix([1, BLOCK_BYTES]),
        &weight,
        &input,
    )
    .generics(launch.generics)
    .grid(launch.grid)
    .l2_cache_key_on(stream)
    .map_err(backend_error)
}
