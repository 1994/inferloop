//! Device-resident routing and selected-expert projections.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3,
        absf, bf16, convert_scalar, convert_tile, eq_tile, exp, f8e4m3fn, get_tile_block_id, iota,
        lt_tile, max_tile, min_tile, reduce_max, reduce_sum, select, tile_to_scalar,
    };

    #[cutile::entry()]
    fn route<const EXPERTS: i32, const B: i32, const A: i32, const NORMALIZE: i32>(
        out: &mut Tensor<f32, { [1] }>,
        scores: &Tensor<f32, { [-1] }>,
        excluded: f32,
    ) {
        let pid = get_tile_block_id();
        let ids: Tile<i32, { [B] }> = iota(shape![B]);
        let indices: Tile<f32, { [B] }> = convert_tile(ids);
        let raw = scores.partition(shape![B]).load([0i32]);
        let mut remaining = select(
            lt_tile(ids, EXPERTS.broadcast(shape![B])),
            raw,
            excluded.broadcast(shape![B]),
        );
        let maximum: Tile<f32, { [] }> = reduce_max(remaining, 0i32);
        let probabilities = exp(remaining - maximum.reshape(shape![1]).broadcast(shape![B]));
        let total: Tile<f32, { [] }> = reduce_sum(probabilities, 0i32);
        let mut selected = 0.0f32;
        let mut probability = 0.0f32;
        let mut selected_total = 0.0f32;
        for rank in 0i32..A {
            let best: Tile<f32, { [] }> = reduce_max(remaining, 0i32);
            let candidates = select(
                eq_tile(remaining, best.reshape(shape![1]).broadcast(shape![B])),
                0.0f32.broadcast(shape![B]) - indices,
                excluded.broadcast(shape![B]),
            );
            let negative_id: Tile<f32, { [] }> = reduce_max(candidates, 0i32);
            let negative: f32 = tile_to_scalar(negative_id);
            let expert = 0.0f32 - negative;
            let p: f32 = tile_to_scalar(exp(best - maximum));
            if rank == pid.0 % A {
                selected = expert;
                probability = p;
            }
            selected_total = selected_total + p;
            remaining = select(
                eq_tile(indices, expert.broadcast(shape![B])),
                excluded.broadcast(shape![B]),
                remaining,
            );
        }
        if pid.0 < A {
            out.store(selected.broadcast(shape![1]));
        } else if NORMALIZE == 1 {
            out.store((probability / selected_total).broadcast(shape![1]));
        } else {
            let sum: f32 = tile_to_scalar(total);
            out.store((probability / sum).broadcast(shape![1]));
        }
    }

    #[cutile::entry()]
    fn dense<const O: i32, const I: i32, const ROUTED: i32>(
        out: &mut Tensor<f32, { [1, 4] }>,
        x: &Tensor<f32, { [-1, I] }>,
        routes: &Tensor<f32, { [-1] }>,
        w: &Tensor<bf16, { [-1, O, I] }>,
    ) {
        let pid = get_tile_block_id();
        let id = routes.partition(shape![1]).load([pid.0]).reshape(shape![]);
        let scalar: f32 = tile_to_scalar(id);
        let expert: i32 = convert_scalar(scalar);
        let row = if ROUTED == 1 { pid.0 } else { 0i32 };
        let mut sum: Tile<f32, { [4, 32] }> = 0.0f32.broadcast(shape![4, 32]);
        for k in 0i32..((I + 31) / 32) {
            let a = x.partition(shape![1, 32]).load([row, k]);
            let b: Tile<f32, { [1, 4, 32] }> =
                convert_tile(w.partition(shape![1, 4, 32]).load([expert, pid.1, k]));
            sum = sum + b.reshape(shape![4, 32]) * a.broadcast(shape![4, 32]);
        }
        let reduced: Tile<f32, { [4] }> = reduce_sum(sum, 1i32);
        out.store(reduced.reshape(shape![1, 4]));
    }

    #[cutile::entry()]
    fn fp8<
        const O: i32,
        const I: i32,
        const ROUTED: i32,
        const CHANNEL: i32,
        const QUANT: i32,
        const PAD: i32,
    >(
        out: &mut Tensor<f32, { [1, 4] }>,
        x: &Tensor<f32, { [-1, I] }>,
        routes: &Tensor<f32, { [-1] }>,
        w: &Tensor<f8e4m3fn, { [-1, O, I] }>,
        scales: &Tensor<f32, { [-1, O, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let id = routes.partition(shape![1]).load([pid.0]).reshape(shape![]);
        let scalar: f32 = tile_to_scalar(id);
        let expert: i32 = convert_scalar(scalar);
        let row = if ROUTED == 1 { pid.0 } else { 0i32 };
        let token_scale: Tile<f32, { [1] }> = if QUANT == 1 && CHANNEL == 1 {
            let whole = x.partition(shape![1, PAD]).load([row, 0i32]);
            let maximum: Tile<f32, { [1] }> = reduce_max(absf(whole), 1i32);
            max_tile(
                maximum / 448.0f32.broadcast(shape![1]),
                1.0e-12f32.broadcast(shape![1]),
            )
        } else {
            1.0f32.broadcast(shape![1])
        };
        let mut sum: Tile<f32, { [4] }> = 0.0f32.broadcast(shape![4]);
        for k in 0i32..((I + 127) / 128) {
            let mut a = x.partition(shape![1, 128]).load([row, k]);
            let mut input_scale = token_scale;
            if QUANT == 1 {
                if CHANNEL == 0 {
                    let maximum: Tile<f32, { [1] }> = reduce_max(absf(a), 1i32);
                    input_scale = max_tile(
                        maximum / 448.0f32.broadcast(shape![1]),
                        1.0e-12f32.broadcast(shape![1]),
                    );
                }
                let scale = input_scale.reshape(shape![1, 1]).broadcast(shape![1, 128]);
                let normalized = min_tile(
                    max_tile(a / scale, (-448.0f32).broadcast(shape![1, 128])),
                    448.0f32.broadcast(shape![1, 128]),
                );
                let packed: Tile<f8e4m3fn, { [1, 128] }> = convert_tile(normalized);
                a = convert_tile(packed);
            }
            let b: Tile<f32, { [1, 4, 128] }> =
                convert_tile(w.partition(shape![1, 4, 128]).load([expert, pid.1, k]));
            let value: Tile<f32, { [4] }> = reduce_sum(
                b.reshape(shape![4, 128]) * a.broadcast(shape![4, 128]),
                1i32,
            );
            let scale = scales
                .partition(shape![1, 4, 1])
                .load([expert, pid.1, if CHANNEL == 1 { 0i32 } else { k }])
                .reshape(shape![4]);
            sum = sum + value * scale * input_scale.broadcast(shape![4]);
        }
        out.store(sum.reshape(shape![1, 4]));
    }

    #[cutile::entry()]
    fn combine<const A: i32, const H: i32>(
        out: &mut Tensor<f32, { [256] }>,
        x: &Tensor<f32, { [A, H] }>,
        routes: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut sum: Tile<f32, { [256] }> = 0.0f32.broadcast(shape![256]);
        for rank in 0i32..A {
            let p = routes
                .partition(shape![1])
                .load([A + rank])
                .reshape(shape![]);
            let row = x
                .partition(shape![1, 256])
                .load([rank, pid.0])
                .reshape(shape![256]);
            sum = sum + row * p.reshape(shape![1]).broadcast(shape![256]);
        }
        out.store(sum);
    }

    #[cutile::entry()]
    fn scale(
        out: &mut Tensor<f32, { [256] }>,
        x: &Tensor<f32, { [-1] }>,
        scalar: &Tensor<f32, { [-1] }>,
    ) {
        let value = scalar.partition(shape![1]).load([0i32]).reshape(shape![]);
        out.store(x.load_like(out) * value.reshape(shape![1]).broadcast(shape![256]));
    }
}
