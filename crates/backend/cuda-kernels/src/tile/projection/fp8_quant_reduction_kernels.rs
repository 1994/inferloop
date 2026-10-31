//! Exact token FP8 quantization with parallel conversion for large aligned rows.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_1, Tile_2, absf,
        convert_tile, f8e4m3fn, get_tile_block_id, max_tile, min_tile, reduce_max,
    };
    #[cutile::entry()]
    fn maximum<const K: i32, const PAD: i32>(
        scale: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![1, PAD]).load([pid.0, 0i32]);
        let maximum: Tile<f32, { [1] }> = reduce_max(absf(x), 1i32);
        let s = max_tile(
            maximum.reshape(shape![1, 1]) / 448.0f32.broadcast(shape![1, 1]),
            1.0e-12f32.broadcast(shape![1, 1]),
        );
        scale.store(s);
    }
    #[cutile::entry()]
    fn quantize<const K: i32>(
        codes: &mut Tensor<f8e4m3fn, { [1, 512] }>,
        scale: &Tensor<f32, { [-1, 1] }>,
        input: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let s = scale.partition(shape![1, 1]).load([pid.0, 0i32]);
        let x = input.partition(shape![1, 512]).load([pid.0, pid.1]);
        let x = x / s.broadcast(shape![1, 512]);
        let clipped = min_tile(
            max_tile(x, (-448.0f32).broadcast(shape![1, 512])),
            448.0f32.broadcast(shape![1, 512]),
        );
        let packed: Tile<f8e4m3fn, { [1, 512] }> = convert_tile(clipped);
        codes.store(packed);
    }
}
