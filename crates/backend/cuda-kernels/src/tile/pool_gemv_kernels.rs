//! Row-independent F32 GEMV with the original private accumulation and reduction tree.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        ElementType, Reshape, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_1, Tile_2, constant,
        convert_tile, get_tile_block_id, reduce_sum,
    };
    #[cutile::entry()]
    fn dense<E: ElementType, const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [1, BN] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![1, BK]);
        let wp = weight.partition(shape![BN, BK]);
        let mut sum: Tile<f32, { [BN, BK] }> = constant(0.0f32, shape![BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let a = xp.load([pid.0, k]);
            let b: Tile<f32, { [BN, BK] }> = convert_tile(wp.load([pid.1, k]));
            sum = sum + b * a.broadcast(shape![BN, BK]);
        }
        let reduced: Tile<f32, { [BN] }> = reduce_sum(sum, 1i32);
        out.store(reduced.reshape(shape![1, BN]));
    }
}
