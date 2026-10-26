//! Checkpoint-declared dynamic per-token E4M3 activation quantization.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tensor_3, Tile_1, Tile_2, absf, constant, convert_tile, f8e4m3fn,
        get_tile_block_id, max_tile, min_tile, mmaf, reduce_max,
    };

    #[cutile::entry()]
    fn quantize<const K: i32, const PAD: i32>(
        output: &mut Tensor<f8e4m3fn, { [1, PAD] }>,
        scale: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![1, PAD]).load([pid.0, 0i32]);
        let maximum: Tile<f32, { [1] }> = reduce_max(absf(x), 1i32);
        let maximum = maximum.reshape(shape![1, 1]);
        let s = max_tile(
            maximum / 448.0f32.broadcast(shape![1, 1]),
            1.0e-12f32.broadcast(shape![1, 1]),
        );
        let x = x / s.broadcast(shape![1, PAD]);
        let x = min_tile(
            max_tile(x, (-448.0f32).broadcast(shape![1, PAD])),
            448.0f32.broadcast(shape![1, PAD]),
        );
        let packed: Tile<f8e4m3fn, { [1, PAD] }> = convert_tile(x);
        output.store(packed);
        scale.store(s);
    }

    /// Block-scaled twin of `quantize`: one scale per `PAD`-sized input block instead of one per
    /// row, matching a checkpoint that declares `weight_block_size` along the input dimension.
    #[cutile::entry()]
    fn quantize_block<const K: i32, const PAD: i32>(
        output: &mut Tensor<f8e4m3fn, { [1, PAD] }>,
        scale: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![1, PAD]).load([pid.0, pid.1]);
        let maximum: Tile<f32, { [1] }> = reduce_max(absf(x), 1i32);
        let maximum = maximum.reshape(shape![1, 1]);
        let s = max_tile(
            maximum / 448.0f32.broadcast(shape![1, 1]),
            1.0e-12f32.broadcast(shape![1, 1]),
        );
        let x = x / s.broadcast(shape![1, PAD]);
        let x = min_tile(
            max_tile(x, (-448.0f32).broadcast(shape![1, PAD])),
            448.0f32.broadcast(shape![1, PAD]),
        );
        let packed: Tile<f8e4m3fn, { [1, PAD] }> = convert_tile(x);
        output.store(packed);
        scale.store(s);
    }

    /// Block-scaled FP8 GEMM: both scales are per 128-element input block, so each K step's
    /// partial product is scaled before it joins the accumulator. Scaling the operands instead
    /// would round them through FP8 a second time.
    #[cutile::entry()]
    fn matmul_block<const K: i32, const M: i32, const N: i32>(
        output: &mut Tensor<f32, { [M, N] }>,
        input: &Tensor<f8e4m3fn, { [-1, K] }>,
        weight: &Tensor<f8e4m3fn, { [-1, K] }>,
        input_scale: &Tensor<f32, { [-1, -1] }>,
        weight_scale: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let wp = weight.partition(shape![N, 128]);
        let xs = input_scale.partition(shape![M, 1]);
        let ws = weight_scale.partition(shape![N, 1]);
        let zero: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        let mut acc: Tile<f32, { [M, N] }> = zero;
        for k in 0i32..((K + 127) / 128) {
            let part = mmaf(xp.load([pid.0, k]), wp.load([pid.1, k]).transpose(), zero);
            let scale = xs.load([pid.0, k]).broadcast(shape![M, N])
                * ws.load([pid.1, k]).transpose().broadcast(shape![M, N]);
            acc = acc + part * scale;
        }
        output.store(acc);
    }

    #[cutile::entry()]
    fn matmul<const K: i32, const M: i32, const N: i32>(
        output: &mut Tensor<f32, { [M, N] }>,
        input: &Tensor<f8e4m3fn, { [-1, K] }>,
        weight: &Tensor<f8e4m3fn, { [-1, K] }>,
        input_scale: &Tensor<f32, { [-1, 1] }>,
        weight_scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let wp = weight.partition(shape![N, 128]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for k in 0i32..((K + 127) / 128) {
            acc = mmaf(xp.load([pid.0, k]), wp.load([pid.1, k]).transpose(), acc);
        }
        let xs = input_scale
            .partition(shape![M, 1])
            .load([pid.0, 0i32])
            .broadcast(shape![M, N]);
        let ws = weight_scale
            .partition(shape![N])
            .load([pid.1])
            .reshape(shape![1, N])
            .broadcast(shape![M, N]);
        output.store(acc * xs * ws);
    }

    /// Split-K twin of `matmul`: `pid.0` selects the K window, so a narrow-M replay stops
    /// paying one deep dependent K chain per output tile. The row and column scales are
    /// uniform across K, so the raw partials are summed and scaled once by `reduce_split`.
    #[cutile::entry()]
    fn matmul_split<const K: i32>(
        output: &mut Tensor<f32, { [1, 16, 64] }>,
        input: &Tensor<f8e4m3fn, { [-1, K] }>,
        weight: &Tensor<f8e4m3fn, { [-1, K] }>,
        k_tiles: i32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![64, 128]);
        let total = (K + 127) / 128;
        let k0 = pid.0 * k_tiles;
        let mut k1 = k0 + k_tiles;
        if k1 > total {
            k1 = total;
        }
        let mut acc: Tile<f32, { [16, 64] }> = constant(0.0f32, shape![16, 64]);
        for k in k0..k1 {
            acc = mmaf(xp.load([0i32, k]), wp.load([pid.1, k]).transpose(), acc);
        }
        output.store(acc.reshape(shape![1, 16, 64]));
    }

    /// Sums `[splits, 16, N]` partial windows in split order and applies both scales.
    #[cutile::entry()]
    fn reduce_split(
        output: &mut Tensor<f32, { [16, 64] }>,
        partials: &Tensor<f32, { [-1, -1, -1] }>,
        input_scale: &Tensor<f32, { [-1, 1] }>,
        weight_scale: &Tensor<f32, { [-1] }>,
        splits: i32,
    ) {
        let pid = get_tile_block_id();
        let pp = partials.partition(shape![1, 16, 64]);
        let mut acc: Tile<f32, { [16, 64] }> = constant(0.0f32, shape![16, 64]);
        for s in 0i32..splits {
            acc = acc + pp.load([s, 0i32, pid.1]).reshape(shape![16, 64]);
        }
        let xs = input_scale
            .partition(shape![16, 1])
            .load([0i32, 0i32])
            .broadcast(shape![16, 64]);
        let ws = weight_scale
            .partition(shape![64])
            .load([pid.1])
            .reshape(shape![1, 64])
            .broadcast(shape![16, 64]);
        output.store(acc * xs * ws);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_fp8_gemm.rs"]
mod tests;
