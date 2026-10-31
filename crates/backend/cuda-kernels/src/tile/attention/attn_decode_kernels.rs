//! Single-query SDPA avoids computing padded query rows; same semantic contract as prefill.
#[cutile::module]
pub mod kernel {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_0,
        Tile_1, Tile_2, constant, exp, ge_tile, get_tile_block_id, gt_tile, iota, le_tile, lt_tile,
        max_tile, reduce_max, reduce_sum, select,
    };
    const MASKED: f32 = -1.0e30;

    #[cutile::entry()]
    fn decode<const D: i32, const DV: i32, const GROUP: i32, const MASK: i32>(
        out: &mut Tensor<f32, { [1, DV] }>,
        query: &Tensor<f32, { [-1, -1] }>,
        key: &Tensor<f32, { [-1, -1] }>,
        value: &Tensor<f32, { [-1, -1] }>,
        tokens: i32,
        scale: f32,
        query_start: i32,
        left: i32,
        right: i32,
    ) {
        let pid = get_tile_block_id();
        let q: Tile<f32, { [32, D] }> = query
            .partition(shape![1, D])
            .load([0i32, pid.1])
            .broadcast(shape![32, D]);
        let kp = key.partition(shape![32, D]);
        let vp = value.partition(shape![32, DV]);
        let mut numerator: Tile<f32, { [DV] }> = constant(0.0f32, shape![DV]);
        let mut total: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
        let mut maximum: Tile<f32, { [1] }> = constant(MASKED, shape![1]);
        for block in 0i32..((tokens + 31i32) / 32i32) {
            let k: Tile<f32, { [32, D] }> = kp.load([block, pid.1 / GROUP]);
            let v: Tile<f32, { [32, DV] }> = vp.load([block, pid.1 / GROUP]);
            let indices: Tile<i32, { [32] }> = iota(shape![32]);
            let positions = indices + (block * 32i32).broadcast(shape![32]);
            let mut valid = lt_tile(positions, tokens.broadcast(shape![32]));
            if MASK == 1 {
                valid = valid & le_tile(positions, query_start.broadcast(shape![32]));
            }
            if MASK == 2 {
                valid = valid
                    & ge_tile(positions, (query_start - left).broadcast(shape![32]))
                    & le_tile(positions, (query_start + right).broadcast(shape![32]));
            }
            let dot: Tile<f32, { [32] }> = reduce_sum(q * k, 1i32);
            let raw: Tile<f32, { [32] }> = dot * scale.broadcast(shape![32]);
            let masked: Tile<f32, { [32] }> = constant(MASKED, shape![32]);
            let scores: Tile<f32, { [32] }> = select(valid, raw, masked);
            let block_max: Tile<f32, { [] }> = reduce_max(scores, 0i32);
            let next: Tile<f32, { [1] }> = max_tile(maximum, block_max.reshape(shape![1]));
            let correction: Tile<f32, { [1] }> = exp(maximum - next);
            let weights: Tile<f32, { [32] }> = select(
                valid,
                exp(scores - next.broadcast(shape![32])),
                0.0f32.broadcast(shape![32]),
            );
            let block_sum: Tile<f32, { [] }> = reduce_sum(weights, 0i32);
            let sum: Tile<f32, { [1] }> = block_sum.reshape(shape![1]);
            let part: Tile<f32, { [DV] }> = reduce_sum(
                v * weights.reshape(shape![32, 1]).broadcast(shape![32, DV]),
                0i32,
            );
            numerator = numerator * correction.broadcast(shape![DV]) + part.reshape(shape![DV]);
            total = total * correction + sum;
            maximum = next;
        }
        let positive = gt_tile(total, 0.0f32.broadcast(shape![1]));
        let denominator = select(positive, total, 1.0f32.broadcast(shape![1]));
        out.store((numerator / denominator.broadcast(shape![DV])).reshape(shape![1, DV]));
    }
}
