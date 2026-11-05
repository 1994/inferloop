//! Small batched projection splitting each F32 input into three BF16 terms with compensated
//! accumulation. It is an arithmetic diagnostic: no production path selects it.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        ElementType, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_2, bf16, constant,
        convert_tile, get_tile_block_id, mmaf,
    };
    /// The third BF16 component reduces the residual input quantization error.
    #[cutile::entry()]
    fn dense<E: ElementType, const K: i32>(
        out: &mut Tensor<f32, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![32, 128]);
        let zero: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        let mut acc: Tile<f32, { [16, 32] }> = zero;
        let mut correction: Tile<f32, { [16, 32] }> = zero;
        for k in 0i32..((K + 127) / 128) {
            let input = xp.load([pid.0, k]);
            let high: Tile<bf16, { [16, 128] }> = convert_tile(input);
            let rounded: Tile<f32, { [16, 128] }> = convert_tile(high);
            let residual = input - rounded;
            let low: Tile<bf16, { [16, 128] }> = convert_tile(residual);
            let rounded_low: Tile<f32, { [16, 128] }> = convert_tile(low);
            let tail: Tile<bf16, { [16, 128] }> = convert_tile(residual - rounded_low);
            let weight: Tile<bf16, { [32, 128] }> = convert_tile(wp.load([pid.1, k]));
            // Sum each component without injecting it into a large previous accumulator.
            // Keep block summation compensated so cancellation across K stays within the F64 gate.
            let high_product = mmaf(high, weight.transpose(), zero);
            let low_product = mmaf(low, weight.transpose(), zero);
            let tail_product = mmaf(tail, weight.transpose(), zero);
            let block = (high_product + low_product) + tail_product;
            let corrected = block - correction;
            let next = acc + corrected;
            correction = (next - acc) - corrected;
            acc = next;
        }
        out.store(acc);
    }
}
