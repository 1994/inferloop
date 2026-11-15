//! `Q4_K` GEMV over the 16-row tile-major layout produced by `infer-gguf`.
//!
//! One super-block (256 values) is decoded at a time. The four 64-value groups become a
//! `[4, 32, 16]` nibble tile; low and high halves are concatenated into `[4, 2, 32, 16]` so one
//! reduction produces both dot products. Per super-block scales, minima and `d`/`dmin` are loaded
//! once, not once per group.
#[cutile::module]
pub mod q4_k {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, Shape_4, StoreTileAtCurrentBlock,
        Tensor_1, Tile_1, Tile_3, Tile_4, andi, cat, constant, convert_tile, get_tile_block_id,
        reduce_sum, shri,
    };

    /// One output tile of sixteen rows. `K` is the reduction length.
    #[cutile::entry()]
    fn gemv<const K: i32, const BLOCKS: i32>(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
        biases: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![256]);
        let cp = codes.partition(shape![2048]);
        let fp = factors.partition(shape![128]);
        let bp = biases.partition(shape![128]);
        let mut sum: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let packed: Tile<u8, { [4, 32, 16] }> = cp.load([index]).reshape(shape![4, 32, 16]);
            let low: Tile<f32, { [4, 32, 16] }> =
                convert_tile(andi(packed, (0x0fu8).broadcast(shape![4, 32, 16])));
            let high: Tile<f32, { [4, 32, 16] }> =
                convert_tile(shri(packed, (4u8).broadcast(shape![4, 32, 16])));
            let pair: Tile<f32, { [4, 2, 32, 16] }> = cat(
                low.reshape(shape![4, 1, 32, 16]),
                high.reshape(shape![4, 1, 32, 16]),
                1i32,
            );
            let values: Tile<f32, { [4, 2, 32, 16] }> = xp
                .load([block])
                .reshape(shape![4, 2, 32])
                .reshape(shape![4, 2, 32, 1])
                .broadcast(shape![4, 2, 32, 16]);
            let code_dot: Tile<f32, { [4, 2, 16] }> = reduce_sum(pair * values, 2i32);
            let x_sum: Tile<f32, { [4, 2, 16] }> = reduce_sum(values, 2i32);
            let factor: Tile<f32, { [4, 2, 16] }> = fp.load([index]).reshape(shape![4, 2, 16]);
            let bias: Tile<f32, { [4, 2, 16] }> = bp.load([index]).reshape(shape![4, 2, 16]);
            let partial: Tile<f32, { [4, 2, 16] }> = factor * code_dot - bias * x_sum;
            sum = sum + reduce_sum(partial.reshape(shape![8, 16]), 0i32);
        }
        out.store(sum);
    }
}
