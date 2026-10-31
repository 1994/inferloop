//! Small batched projections using tensor cores with a two-term BF16 input.
//! Retaining the input residual avoids reducing F32 activations to one BF16 value.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        ElementType, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1, Tensor_2,
        Tile_2, bf16, constant, convert_tile, get_tile_block_id, mmaf,
    };
    #[cutile::entry()]
    fn dense<E: ElementType, const K: i32>(
        out: &mut Tensor<f32, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![32, 128]);
        let mut acc: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        for k in 0i32..((K + 127) / 128) {
            let input = xp.load([pid.0, k]);
            let high: Tile<bf16, { [16, 128] }> = convert_tile(input);
            let rounded: Tile<f32, { [16, 128] }> = convert_tile(high);
            let low: Tile<bf16, { [16, 128] }> = convert_tile(input - rounded);
            let weight: Tile<bf16, { [32, 128] }> = convert_tile(wp.load([pid.1, k]));
            acc = mmaf(low, weight.transpose(), acc);
            acc = mmaf(high, weight.transpose(), acc);
        }
        out.store(acc);
    }
    #[cutile::entry()]
    fn scaled<E: ElementType, const K: i32>(
        out: &mut Tensor<f32, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![32, 128]);
        let mut acc: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        for k in 0i32..((K + 127) / 128) {
            let input = xp.load([pid.0, k]);
            let high: Tile<bf16, { [16, 128] }> = convert_tile(input);
            let rounded: Tile<f32, { [16, 128] }> = convert_tile(high);
            let low: Tile<bf16, { [16, 128] }> = convert_tile(input - rounded);
            let weight: Tile<bf16, { [32, 128] }> = convert_tile(wp.load([pid.1, k]));
            acc = mmaf(low, weight.transpose(), acc);
            acc = mmaf(high, weight.transpose(), acc);
        }
        let scale = scale
            .partition(shape![32])
            .load([pid.1])
            .reshape(shape![1, 32])
            .broadcast(shape![16, 32]);
        out.store(acc * scale);
    }
}
