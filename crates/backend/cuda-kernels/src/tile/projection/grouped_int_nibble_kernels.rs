//! Nibble-packed grouped-int GEMV over the 16-row tile-major layout.
//!
//! Each iteration decodes four 256-value spans into a `[1024, 16]` tile, applies the per-group
//! terms, multiplies the broadcast activation and reduces over K. Lane `l` of each 64-value group
//! holds value `l` in its low nibble and value `l + 32` in its high nibble. `gemv` subtracts a
//! per-group bias; `gemv_zero` subtracts the format's constant zero point instead.
#[cutile::module]
pub mod nibble {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, Shape_4, StoreTileAtCurrentBlock,
        Tensor_1, Tile_1, Tile_2, Tile_3, Tile_4, andi, cat, constant, convert_scalar,
        convert_tile, get_tile_block_id, reduce_sum, shri,
    };

    /// One output tile of sixteen rows: `factor * code - bias`. `K` is the reduction length.
    #[cutile::entry()]
    fn gemv<const K: i32, const BLOCKS: i32>(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
        biases: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1024]);
        let cp = codes.partition(shape![8192]);
        let fp = factors.partition(shape![512]);
        let bp = biases.partition(shape![512]);
        let mut sum: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let packed: Tile<u8, { [16, 32, 16] }> = cp.load([index]).reshape(shape![16, 32, 16]);
            let low: Tile<f32, { [16, 32, 16] }> =
                convert_tile(andi(packed, (0x0fu8).broadcast(shape![16, 32, 16])));
            let high: Tile<f32, { [16, 32, 16] }> =
                convert_tile(shri(packed, (4u8).broadcast(shape![16, 32, 16])));
            let pair: Tile<f32, { [16, 2, 32, 16] }> = cat(
                low.reshape(shape![16, 1, 32, 16]),
                high.reshape(shape![16, 1, 32, 16]),
                1i32,
            );
            let values: Tile<f32, { [1024, 16] }> = pair.reshape(shape![1024, 16]);
            let factor: Tile<f32, { [1024, 16] }> = fp
                .load([index])
                .reshape(shape![16, 2, 16])
                .reshape(shape![16, 2, 1, 16])
                .broadcast(shape![16, 2, 32, 16])
                .reshape(shape![1024, 16]);
            let bias: Tile<f32, { [1024, 16] }> = bp
                .load([index])
                .reshape(shape![16, 2, 16])
                .reshape(shape![16, 2, 1, 16])
                .broadcast(shape![16, 2, 32, 16])
                .reshape(shape![1024, 16]);
            let activation: Tile<f32, { [1024, 16] }> = xp
                .load([block])
                .reshape(shape![1024, 1])
                .broadcast(shape![1024, 16]);
            let dequant = factor * values - bias;
            sum = sum + reduce_sum(dequant * activation, 0i32);
        }
        out.store(sum);
    }

    /// One output tile of sixteen rows: `factor * (code - ZERO)`.
    #[cutile::entry()]
    fn gemv_zero<const K: i32, const BLOCKS: i32, const ZERO: i32>(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1024]);
        let cp = codes.partition(shape![8192]);
        let fp = factors.partition(shape![512]);
        let zero: f32 = convert_scalar(ZERO);
        let mut sum: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let packed: Tile<u8, { [16, 32, 16] }> = cp.load([index]).reshape(shape![16, 32, 16]);
            let low: Tile<f32, { [16, 32, 16] }> =
                convert_tile(andi(packed, (0x0fu8).broadcast(shape![16, 32, 16])));
            let high: Tile<f32, { [16, 32, 16] }> =
                convert_tile(shri(packed, (4u8).broadcast(shape![16, 32, 16])));
            let pair: Tile<f32, { [16, 2, 32, 16] }> = cat(
                low.reshape(shape![16, 1, 32, 16]),
                high.reshape(shape![16, 1, 32, 16]),
                1i32,
            );
            let values: Tile<f32, { [1024, 16] }> = pair.reshape(shape![1024, 16]);
            let values = values - zero.broadcast(shape![1024, 16]);
            let factor: Tile<f32, { [1024, 16] }> = fp
                .load([index])
                .reshape(shape![16, 2, 16])
                .reshape(shape![16, 2, 1, 16])
                .broadcast(shape![16, 2, 32, 16])
                .reshape(shape![1024, 16]);
            let activation: Tile<f32, { [1024, 16] }> = xp
                .load([block])
                .reshape(shape![1024, 1])
                .broadcast(shape![1024, 16]);
            sum = sum + reduce_sum(factor * values * activation, 0i32);
        }
        out.store(sum);
    }
}
