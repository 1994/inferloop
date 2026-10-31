//! Model-independent compensated scaled dot-product attention.
#[cutile::module]
pub mod sdpa {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_1,
        Tile_2, bf16, constant, convert_tile, eq_tile, exp, ge_tile, get_tile_block_id, gt_tile,
        iota, le_tile, lt_tile, max_tile, mmaf, reduce_max, reduce_sum, select,
    };
    /// Finite masked score keeps online rescaling defined before the first visible key.
    const MASKED: f32 = -1.0e30;
    /// Compensated BF16 tensor-core product, with F32 inputs and accumulation.
    /// Keep both first-order residual products; the omitted low*low term is O(2^-16).
    fn precise_mma<const M: i32, const K: i32, const N: i32>(
        left: Tile<f32, { [M, K] }>,
        right: Tile<f32, { [K, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        let lh: Tile<bf16, { [M, K] }> = convert_tile(left);
        let rh: Tile<bf16, { [K, N] }> = convert_tile(right);
        let lr: Tile<f32, { [M, K] }> = convert_tile(lh);
        let rr: Tile<f32, { [K, N] }> = convert_tile(rh);
        let ll: Tile<bf16, { [M, K] }> = convert_tile(left - lr);
        let rl: Tile<bf16, { [K, N] }> = convert_tile(right - rr);
        let zero: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        let low = mmaf(ll, rh, zero);
        let low = mmaf(lh, rl, low);
        mmaf(lh, rh, low)
    }

    /// Visibility depends on sequence coordinates, never a model or a device name.
    fn visibility<const Q: i32, const K: i32, const MASK: i32>(
        block: i32,
        query_block: i32,
        queries: i32,
        keys: i32,
        segment: i32,
        query_start: i32,
        left: i32,
        right: i32,
    ) -> Tile<bool, { [Q, K] }> {
        let qi: Tile<i32, { [Q] }> = iota(shape![Q]);
        let ki: Tile<i32, { [K] }> = iota(shape![K]);
        let q = qi + (query_block * Q).broadcast(shape![Q]);
        let k = ki + (block * K).broadcast(shape![K]);
        let q2: Tile<i32, { [Q, K] }> = q.reshape(shape![Q, 1]).broadcast(shape![Q, K]);
        let k2: Tile<i32, { [Q, K] }> = k.reshape(shape![1, K]).broadcast(shape![Q, K]);
        let mut valid: Tile<bool, { [Q, K] }> = lt_tile(q2, queries.broadcast(shape![Q, K]))
            & lt_tile(k2, keys.broadcast(shape![Q, K]));
        if MASK == 1 {
            valid = valid & le_tile(k2, q2 + query_start.broadcast(shape![Q, K]));
        }
        if MASK == 2 {
            let center = q2 + query_start.broadcast(shape![Q, K]);
            valid = valid
                & ge_tile(k2, center - left.broadcast(shape![Q, K]))
                & le_tile(k2, center + right.broadcast(shape![Q, K]));
        }
        if MASK == 3 {
            valid = valid
                & eq_tile(
                    q2 / segment.broadcast(shape![Q, K]),
                    k2 / segment.broadcast(shape![Q, K]),
                );
        }
        valid
    }

    /// Packed non-causal attention over already rotated full heads.
    /// ONLINE=0 first computes the global maximum as an independent streaming reference.
    #[cutile::entry()]
    fn attention<
        const D: i32,
        const ONLINE: i32,
        const Q: i32,
        const K: i32,
        const PIPE: i32,
        const GROUP: i32,
        const DV: i32,
        const MASK: i32,
    >(
        out: &mut Tensor<f32, { [Q, DV] }>,
        query: &Tensor<f32, { [-1, -1] }>,
        key: &Tensor<f32, { [-1, -1] }>,
        value: &Tensor<f32, { [-1, -1] }>,
        tokens: i32,
        scale: f32,
        frame_tokens: i32,
        queries: i32,
        query_start: i32,
        left: i32,
        right: i32,
    ) {
        let pid = get_tile_block_id();
        let q: Tile<f32, { [Q, D] }> = query.partition(shape![Q, D]).load([pid.0, pid.1]);
        let kp = key.partition(shape![K, D]);
        let vp = value.partition(shape![K, DV]);
        let blocks = (tokens + K - 1i32) / K;
        let mut first = 0i32;
        let mut last = blocks;
        if MASK == 3 {
            first = ((pid.0 * Q) / frame_tokens * frame_tokens) / K;
            let end = (((pid.0 + 1i32) * Q - 1i32) / frame_tokens + 1i32) * frame_tokens;
            last = (end + K - 1i32) / K;
            if last > blocks {
                last = blocks;
            }
        }
        let mut accumulator: Tile<f32, { [Q, DV] }> = constant(0.0f32, shape![Q, DV]);
        let mut row_max: Tile<f32, { [Q] }> = constant(MASKED, shape![Q]);
        if ONLINE == 0 {
            for block in first..last {
                let k: Tile<f32, { [K, D] }> = if PIPE == 1 {
                    kp.load_pipelined::<4>([block, pid.1 / GROUP])
                } else {
                    kp.load([block, pid.1 / GROUP])
                };
                let valid: Tile<bool, { [Q, K] }> = visibility::<Q, K, MASK>(
                    block,
                    pid.0,
                    queries,
                    tokens,
                    frame_tokens,
                    query_start,
                    left,
                    right,
                );
                let raw: Tile<f32, { [Q, K] }> =
                    precise_mma(q, k.transpose()) * scale.broadcast(shape![Q, K]);
                let masked: Tile<f32, { [Q, K] }> = constant(MASKED, shape![Q, K]);
                let score: Tile<f32, { [Q, K] }> = select(valid, raw, masked);
                let maxima: Tile<f32, { [Q] }> = reduce_max(score, 1i32);
                row_max = max_tile(row_max, maxima.reshape(shape![Q]));
            }
        }
        let mut row_sum: Tile<f32, { [Q] }> = constant(0.0f32, shape![Q]);
        for block in first..last {
            let k: Tile<f32, { [K, D] }> = if PIPE == 1 {
                kp.load_pipelined::<4>([block, pid.1 / GROUP])
            } else {
                kp.load([block, pid.1 / GROUP])
            };
            let valid: Tile<bool, { [Q, K] }> = visibility::<Q, K, MASK>(
                block,
                pid.0,
                queries,
                tokens,
                frame_tokens,
                query_start,
                left,
                right,
            );
            let raw: Tile<f32, { [Q, K] }> =
                precise_mma(q, k.transpose()) * scale.broadcast(shape![Q, K]);
            let masked: Tile<f32, { [Q, K] }> = constant(MASKED, shape![Q, K]);
            let score: Tile<f32, { [Q, K] }> = select(valid, raw, masked);
            let mut shift: Tile<f32, { [Q] }> = row_max;
            let mut correction: Tile<f32, { [Q] }> = 1.0f32.broadcast(shape![Q]);
            if ONLINE == 1 {
                let maxima: Tile<f32, { [Q] }> = reduce_max(score, 1i32);
                let next_max: Tile<f32, { [Q] }> = max_tile(row_max, maxima.reshape(shape![Q]));
                correction = exp(row_max - next_max);
                shift = next_max;
                row_max = next_max;
            }
            let probabilities = select(
                valid,
                exp(score - shift.reshape(shape![Q, 1]).broadcast(shape![Q, K])),
                0.0f32.broadcast(shape![Q, K]),
            );
            let sums: Tile<f32, { [Q] }> = reduce_sum(probabilities, 1i32);
            row_sum = row_sum * correction + sums.reshape(shape![Q]);
            let values: Tile<f32, { [K, DV] }> = if PIPE == 1 {
                vp.load_pipelined::<4>([block, pid.1 / GROUP])
            } else {
                vp.load([block, pid.1 / GROUP])
            };
            let rescale = correction.reshape(shape![Q, 1]).broadcast(shape![Q, DV]);
            let product: Tile<f32, { [Q, DV] }> = precise_mma(probabilities, values);
            accumulator = accumulator * rescale + product.reshape(shape![Q, DV]);
        }
        let positive = gt_tile(row_sum, 0.0f32.broadcast(shape![Q]));
        let denominator = select(positive, row_sum, 1.0f32.broadcast(shape![Q]));
        out.store(accumulator / denominator.reshape(shape![Q, 1]).broadcast(shape![Q, DV]));
    }
}
