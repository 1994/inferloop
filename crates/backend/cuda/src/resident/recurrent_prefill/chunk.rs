//! Chunked gated delta rule on tensor cores: the FLA/FlashInfer prompt formulation.
//!
//! Within a 64-token chunk the recurrence `S_t = d_t S_{t-1} + k_t (b_t (v_t - d_t S_{t-1}^T k_t))^T`
//! has the closed form `U = T (b V) - T (b g K) S_0`, `O = (g Q) S_0 + mask(Q K^T G) U` and
//! `S_64 = g_64 S_0 + (g_64 / g) K^T U`, where `g` is the cumulative decay, `G_ij = g_i / g_j` and
//! `T = (I + strict_lower(b_i k_i k_j G_ij))^-1`.
//!
//! [`kernels::chunk_normalize`] writes L2-normalized BF16 queries and keys per key head;
//! [`kernels::chunk_prepare`] evaluates every chunk independently into BF16 factors (BF16
//! products, TF32 for the triangular inverse);
//! [`kernels::chunk_scan`] then carries the F32 state across chunks with four BF16 products per
//! chunk, as the FLA and `FlashInfer` kernels do.
use crate::device::{CudaDevice, device_error};
use cutile::{half::bf16, prelude::*};
use infer_core::Result;
use std::result::Result as StdResult;

/// Tokens per chunk.
pub(super) const CHUNK: usize = 64;
/// Value columns one scan block carries.
const SCAN_COLUMNS: usize = 32;

#[cutile::module]
pub(super) mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_1, Tile_2, bf16, constant,
        convert_scalar, convert_tile, eq_tile, exp, extract, ftz, ge_tile, get_tile_block_id,
        gt_tile, iota, log, lt_tile, mmaf, reduce_sum, reverse, rsqrt, scan_sum, select, tf32,
        tile_to_scalar,
    };

    /// `left * right + acc` from one TF32 product.
    fn tf32_product<const M: i32, const K: i32, const N: i32>(
        left: Tile<f32, { [M, K] }>,
        right: Tile<f32, { [K, N] }>,
        acc: Tile<f32, { [M, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        let lh: Tile<tf32, { [M, K] }> = convert_tile(left);
        let rh: Tile<tf32, { [K, N] }> = convert_tile(right);
        mmaf(lh, rh, acc)
    }

    /// `left * right + acc` from BF16 operands with F32 accumulation.
    fn bf16_product<const M: i32, const K: i32, const N: i32>(
        left: Tile<f32, { [M, K] }>,
        right: Tile<f32, { [K, N] }>,
        acc: Tile<f32, { [M, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        let lh: Tile<bf16, { [M, K] }> = convert_tile(left);
        let rh: Tile<bf16, { [K, N] }> = convert_tile(right);
        mmaf(lh, rh, acc)
    }

    /// Rows of a chunk that hold a real prompt token at a non-negative state position.
    fn active_rows(metadata: &Tensor<i32, { [-1] }>, chunk: i32) -> Tile<bool, { [64] }> {
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let rows: Tile<i32, { [64] }> = iota(shape![64]) + (chunk * 64i32).broadcast(shape![64]);
        lt_tile(rows, count.broadcast(shape![64]))
            & ge_tile(
                rows + (base + offset).broadcast(shape![64]),
                0i32.broadcast(shape![64]),
            )
    }

    /// `(I + interactions)^-1` for a strictly lower 64x64 matrix, by merging inverted diagonal
    /// blocks: `[[L11, 0], [L21, L22]]^-1` has cross block `-inv(L22) L21 inv(L11)`.
    fn unit_lower_inverse(
        interactions: Tile<f32, { [64, 64] }>,
        rows: Tile<i32, { [64, 64] }>,
        columns: Tile<i32, { [64, 64] }>,
        lower: Tile<bool, { [64, 64] }>,
    ) -> Tile<f32, { [64, 64] }> {
        let zero: Tile<f32, { [64, 64] }> = constant(0.0f32, shape![64, 64]);
        let identity = select(
            eq_tile(rows, columns),
            1.0f32.broadcast(shape![64, 64]),
            zero,
        );
        let two = 2i32.broadcast(shape![64, 64]);
        let pair = eq_tile(rows / two, columns / two) & lower;
        let mut inverse = identity - select(pair, interactions, zero);
        let mut half = 2i32;
        for _level in 0i32..5i32 {
            let width = (2i32 * half).broadcast(shape![64, 64]);
            let row_block = rows / width;
            let column_block = columns / width;
            let middle = row_block * width + half.broadcast(shape![64, 64]);
            let cross =
                eq_tile(row_block, column_block) & ge_tile(rows, middle) & lt_tile(columns, middle);
            let coupling = select(cross, interactions, zero);
            let left: Tile<f32, { [64, 64] }> = tf32_product(inverse, coupling, zero);
            let merged: Tile<f32, { [64, 64] }> = tf32_product(left, inverse, zero);
            inverse = inverse - merged;
            half = half * 2i32;
        }
        inverse
    }

    /// L2-normalized BF16 queries (also scaled by `1/sqrt(D)`) and keys of one key head and
    /// chunk; inactive rows are zero.
    #[cutile::entry()]
    fn chunk_normalize<const KH: i32, const D: i32>(
        queries: &mut Tensor<bf16, { [1, 64, D] }>,
        keys: &mut Tensor<bf16, { [1, 64, D] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let head = pid.0;
        let chunk = pid.1;
        let valid = active_rows(metadata, chunk)
            .reshape(shape![64, 1])
            .broadcast(shape![64, D]);
        let zero: Tile<f32, { [64, D] }> = constant(0.0f32, shape![64, D]);
        let part = qkv.partition(shape![64, 1, D]);
        let q: Tile<f32, { [64, D] }> = part.load([chunk, head, 0i32]).reshape(shape![64, D]);
        let k: Tile<f32, { [64, D] }> = part.load([chunk, KH + head, 0i32]).reshape(shape![64, D]);
        let qs: Tile<f32, { [64] }> = reduce_sum(q * q, 1i32);
        let ks: Tile<f32, { [64] }> = reduce_sum(k * k, 1i32);
        let d: f32 = convert_scalar(D);
        let qscale = rsqrt(qs + 1e-6f32.broadcast(shape![64]), ftz::Disabled)
            * rsqrt(d.broadcast(shape![64]), ftz::Disabled);
        let kscale = rsqrt(ks + 1e-6f32.broadcast(shape![64]), ftz::Disabled);
        let q = select(
            valid,
            q * qscale.reshape(shape![64, 1]).broadcast(shape![64, D]),
            zero,
        );
        let k = select(
            valid,
            k * kscale.reshape(shape![64, 1]).broadcast(shape![64, D]),
            zero,
        );
        let q: Tile<bf16, { [64, D] }> = convert_tile(q);
        let k: Tile<bf16, { [64, D] }> = convert_tile(k);
        queries.store(q.reshape(shape![1, 64, D]));
        keys.store(k.reshape(shape![1, 64, D]));
    }

    /// Per-chunk factors of one value head: `U = T (b V)`, `W = T (b g K)`,
    /// the masked `Q K^T G` and the cumulative log decay. Inactive rows contribute nothing and do
    /// not decay the state.
    #[expect(clippy::too_many_arguments, reason = "explicit CUDA tensor bindings")]
    #[expect(
        clippy::many_single_char_names,
        reason = "chunk factors keep the derivation's symbols"
    )]
    #[cutile::entry()]
    fn chunk_prepare<const KH: i32, const VH: i32, const D: i32>(
        u: &mut Tensor<bf16, { [1, 64, D] }>,
        w: &mut Tensor<bf16, { [1, 64, D] }>,
        p: &mut Tensor<bf16, { [1, 64, 64] }>,
        cumulative: &mut Tensor<f32, { [1, 64] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
        queries: &Tensor<bf16, { [-1, -1, D] }>,
        keys: &Tensor<bf16, { [-1, -1, D] }>,
        betas: &Tensor<f32, { [-1, -1] }>,
        decays: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let head = pid.0;
        let chunk = pid.1;
        let active = active_rows(metadata, chunk);
        let valid = active.reshape(shape![64, 1]).broadcast(shape![64, D]);
        let zero_rows: Tile<f32, { [64, D] }> = constant(0.0f32, shape![64, D]);
        let key = head / (VH / KH);
        let qb: Tile<bf16, { [64, D] }> = queries
            .partition(shape![1, 64, D])
            .load([key, chunk, 0i32])
            .reshape(shape![64, D]);
        let kb: Tile<bf16, { [64, D] }> = keys
            .partition(shape![1, 64, D])
            .load([key, chunk, 0i32])
            .reshape(shape![64, D]);
        let q: Tile<f32, { [64, D] }> = convert_tile(qb);
        let k: Tile<f32, { [64, D] }> = convert_tile(kb);
        let v = select(
            valid,
            qkv.partition(shape![64, 1, D])
                .load([chunk, 2i32 * KH + head, 0i32])
                .reshape(shape![64, D]),
            zero_rows,
        );
        let zero64: Tile<f32, { [64] }> = constant(0.0f32, shape![64]);
        let b = select(
            active,
            betas
                .partition(shape![64, 1])
                .load([chunk, head])
                .reshape(shape![64]),
            zero64,
        );
        let step = select(
            active,
            log(decays
                .partition(shape![64, 1])
                .load([chunk, head])
                .reshape(shape![64])),
            zero64,
        );
        let g = scan_sum(step, 0i32, reverse::Forward, 0.0f32);
        let indices: Tile<i32, { [64] }> = iota(shape![64]);
        let row_index = indices.reshape(shape![64, 1]).broadcast(shape![64, 64]);
        let column_index = indices.reshape(shape![1, 64]).broadcast(shape![64, 64]);
        let lower = gt_tile(row_index, column_index);
        let causal = ge_tile(row_index, column_index);
        let zero_square: Tile<f32, { [64, 64] }> = constant(0.0f32, shape![64, 64]);
        let gap = g.reshape(shape![64, 1]).broadcast(shape![64, 64])
            - g.reshape(shape![1, 64]).broadcast(shape![64, 64]);
        // Mask before exponentiation: upper-triangle gaps can overflow.
        let relative = exp(select(causal, gap, zero_square));
        let kk: Tile<f32, { [64, 64] }> = bf16_product(k, k.transpose(), zero_square);
        let interactions = select(
            lower,
            kk * relative * b.reshape(shape![64, 1]).broadcast(shape![64, 64]),
            zero_square,
        );
        let inverse = unit_lower_inverse(interactions, row_index, column_index, lower);
        let bv = v * b.reshape(shape![64, 1]).broadcast(shape![64, D]);
        let bk = k * (b * exp(g)).reshape(shape![64, 1]).broadcast(shape![64, D]);
        let qk: Tile<f32, { [64, 64] }> = bf16_product(q, k.transpose(), zero_square);
        let updates: Tile<f32, { [64, D] }> = bf16_product(inverse, bv, zero_rows);
        let weights: Tile<f32, { [64, D] }> = bf16_product(inverse, bk, zero_rows);
        let weights: Tile<bf16, { [64, D] }> = convert_tile(weights);
        let scores: Tile<bf16, { [64, 64] }> =
            convert_tile(select(causal, qk * relative, zero_square));
        let updates: Tile<bf16, { [64, D] }> = convert_tile(updates);
        u.store(updates.reshape(shape![1, 64, D]));
        w.store(weights.reshape(shape![1, 64, D]));
        p.store(scores.reshape(shape![1, 64, 64]));
        cumulative.store(g.reshape(shape![1, 64]));
    }

    /// Carry one 32-column block of the state across the active chunks.
    #[expect(clippy::too_many_arguments, reason = "explicit CUDA tensor bindings")]
    #[cutile::entry()]
    fn chunk_scan<const KH: i32, const VH: i32, const LANES: i32>(
        state: &mut Tensor<f32, { [1, 128, 32] }>,
        out: &mut Tensor<f32, { [1, LANES, 32] }>,
        u: &Tensor<bf16, { [-1, -1, 128] }>,
        w: &Tensor<bf16, { [-1, -1, 128] }>,
        queries: &Tensor<bf16, { [-1, -1, 128] }>,
        keys: &Tensor<bf16, { [-1, -1, 128] }>,
        p: &Tensor<bf16, { [-1, -1, 64] }>,
        cumulative: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let head = pid.0;
        let key = head / (VH / KH);
        let count: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([1i32]).reshape(shape![]));
        let mut old = state.load_like(state).reshape(shape![128, 32]);
        let zero_block: Tile<f32, { [64, 32] }> = constant(0.0f32, shape![64, 32]);
        let wp = w.partition(shape![1, 64, 128]);
        let qp = queries.partition(shape![1, 64, 128]);
        let kp = keys.partition(shape![1, 64, 128]);
        let pp = p.partition(shape![1, 64, 64]);
        let gp = cumulative.partition(shape![1, 64]);
        for chunk in 0i32..((count + 63i32) / 64i32) {
            let g: Tile<f32, { [64] }> = gp.load_pipelined::<2>([head, chunk]).reshape(shape![64]);
            let g_last: Tile<f32, { [1] }> = extract(g, [63i32.broadcast(shape![])]);
            let wb: Tile<bf16, { [64, 128] }> = wp
                .load_pipelined::<2>([head, chunk, 0i32])
                .reshape(shape![64, 128]);
            let wc: Tile<f32, { [64, 128] }> = convert_tile(wb);
            let predicted: Tile<f32, { [64, 32] }> = bf16_product(wc, old, zero_block);
            let target: Tile<bf16, { [64, 32] }> = u
                .partition(shape![1, 64, 32])
                .load_pipelined::<2>([head, chunk, pid.2])
                .reshape(shape![64, 32]);
            let target: Tile<f32, { [64, 32] }> = convert_tile(target);
            let updates = target - predicted;
            let qb: Tile<bf16, { [64, 128] }> = qp
                .load_pipelined::<2>([key, chunk, 0i32])
                .reshape(shape![64, 128]);
            let qf: Tile<f32, { [64, 128] }> = convert_tile(qb);
            let qc = qf * exp(g).reshape(shape![64, 1]).broadcast(shape![64, 128]);
            let pb: Tile<bf16, { [64, 64] }> = pp
                .load_pipelined::<2>([head, chunk, 0i32])
                .reshape(shape![64, 64]);
            let pc: Tile<f32, { [64, 64] }> = convert_tile(pb);
            let carried: Tile<f32, { [64, 32] }> = bf16_product(qc, old, zero_block);
            let output: Tile<f32, { [64, 32] }> = bf16_product(pc, updates, carried);
            out.partition_mut(shape![1, 64, 32])
                .store(output.reshape(shape![1, 64, 32]), [0i32, chunk, 0i32]);
            let kb: Tile<bf16, { [64, 128] }> = kp
                .load_pipelined::<2>([key, chunk, 0i32])
                .reshape(shape![64, 128]);
            let kf: Tile<f32, { [64, 128] }> = convert_tile(kb);
            let remaining = exp(g_last.broadcast(shape![64]) - g);
            let kc = kf * remaining.reshape(shape![64, 1]).broadcast(shape![64, 128]);
            let decayed: Tile<f32, { [128, 32] }> =
                old * exp(g_last).reshape(shape![1, 1]).broadcast(shape![128, 32]);
            let transposed: Tile<f32, { [128, 64] }> = kc.transpose();
            old = bf16_product(transposed, updates, decayed);
        }
        state.store(old.reshape(shape![1, 128, 32]));
    }
}

/// Scratch for one prompt width.
pub(super) struct Scratch {
    u: Tensor<bf16>,
    queries: Tensor<bf16>,
    keys: Tensor<bf16>,
    w: Tensor<bf16>,
    p: Tensor<bf16>,
    cumulative: Tensor<f32>,
}

impl Scratch {
    pub(super) fn new(
        device: &CudaDevice,
        (kh, vh, dim): (usize, usize, usize),
        lanes: usize,
    ) -> Result<Self> {
        let halves = |shape: &[usize]| {
            api::zeros::<bf16>(shape)
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        Ok(Self {
            u: halves(&[vh, lanes, dim])?,
            queries: halves(&[kh, lanes, dim])?,
            keys: halves(&[kh, lanes, dim])?,
            w: halves(&[vh, lanes, dim])?,
            p: halves(&[vh, lanes, CHUNK])?,
            cumulative: api::zeros::<f32>(&[vh, lanes])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }

    /// Device bytes of [`Self::new`].
    pub(super) const fn bytes((kh, vh, dim): (usize, usize, usize), lanes: usize) -> usize {
        (2 * kh * lanes * dim + 2 * vh * lanes * dim + vh * lanes * CHUNK) * size_of::<bf16>()
            + vh * lanes * size_of::<f32>()
    }
}

/// Whether the chunked rule covers this geometry.
pub(super) const fn enabled(dim: usize, lanes: usize) -> bool {
    dim == super::ORDERED_DIM && lanes >= CHUNK && lanes.is_multiple_of(CHUNK)
}

/// Record the chunked recurrence; `betas` and `decays` come from the shared gate kernel.
pub(super) fn record(
    scope: &Scope,
    scratch: &mut Scratch,
    qkv: &TensorView<'_, f32>,
    (betas, decays): (&Tensor<f32>, &Tensor<f32>),
    [state, output]: [&mut Tensor<f32>; 2],
    metadata: &Tensor<i32>,
    (kh, vh, dim, lanes): (usize, usize, usize, usize),
) -> StdResult<(), DeviceError> {
    scope.record(
        kernels::chunk_normalize(
            (&mut scratch.queries).partition([1, CHUNK, dim]),
            (&mut scratch.keys).partition([1, CHUNK, dim]),
            qkv,
            metadata,
        )
        .generics(vec![kh.to_string(), dim.to_string()]),
    )?;
    scope.record(
        kernels::chunk_prepare(
            (&mut scratch.u).partition([1, CHUNK, dim]),
            (&mut scratch.w).partition([1, CHUNK, dim]),
            (&mut scratch.p).partition([1, CHUNK, CHUNK]),
            (&mut scratch.cumulative).partition([1, CHUNK]),
            qkv,
            &scratch.queries,
            &scratch.keys,
            betas,
            decays,
            metadata,
        )
        .generics(vec![kh.to_string(), vh.to_string(), dim.to_string()]),
    )?;
    scope.record(
        kernels::chunk_scan(
            state.partition([1, dim, SCAN_COLUMNS]),
            output.partition([1, lanes, SCAN_COLUMNS]),
            &scratch.u,
            &scratch.w,
            &scratch.queries,
            &scratch.keys,
            &scratch.p,
            &scratch.cumulative,
            metadata,
        )
        .generics(vec![kh.to_string(), vh.to_string(), lanes.to_string()]),
    )?;
    Ok(())
}
