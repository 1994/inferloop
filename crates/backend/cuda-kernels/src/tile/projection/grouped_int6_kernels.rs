//! `Q6_K` GEMV over the tile-major layout produced by the loader.
//!
//! Codes are repacked to one byte per value and factors are `d * scale` per 16-value group, so
//! the kernel subtracts the 32 zero point, multiplies the activation, and reduces over K.
#[cutile::module]
pub mod q6_k {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock, Tensor_1,
        Tile_1, Tile_2, Tile_3, constant, convert_tile, get_tile_block_id, reduce_sum,
    };

    /// One output tile of sixteen rows. `K` is the reduction length.
    #[cutile::entry()]
    fn gemv<const K: i32, const BLOCKS: i32>(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let cp = codes.partition(shape![4096]);
        let fp = factors.partition(shape![256]);
        let xp = x.partition(shape![256]);
        let mut sum: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let raw: Tile<u8, { [256, 16] }> = cp.load([index]).reshape(shape![256, 16]);
            let code: Tile<f32, { [256, 16] }> =
                convert_tile(raw) - (32.0f32).broadcast(shape![256, 16]);
            let activation: Tile<f32, { [256, 16] }> = xp
                .load([block])
                .reshape(shape![256, 1])
                .broadcast(shape![256, 16]);
            let product: Tile<f32, { [16, 16, 16] }> =
                (code * activation).reshape(shape![16, 16, 16]);
            let factor: Tile<f32, { [16, 16] }> = fp.load([index]).reshape(shape![16, 16]);
            let scaled: Tile<f32, { [16, 16, 16] }> = product
                * factor
                    .reshape(shape![16, 1, 16])
                    .broadcast(shape![16, 16, 16]);
            let by_group: Tile<f32, { [16, 16] }> = reduce_sum(scaled, 0i32);
            sum = sum + reduce_sum(by_group, 0i32);
        }
        out.store(sum);
    }
}
