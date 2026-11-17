//! Byte-per-code grouped-int GEMV over the 16-row tile-major layout.
//!
//! Each iteration reads one 256-value span as a `[256, 16]` code tile and its `[GROUPS, 16]`
//! per-group terms (`TERMS = GROUPS * 16`, `GROUPS * GROUP = 256`), multiplies the broadcast
//! activation and reduces over K. `gemv_zero` subtracts the format's constant zero point;
//! `gemv` subtracts a per-group bias.
//!
//! `gemm` / `gemm_zero` cover `ROWS` activation rows per CTA: each span is decoded once into a
//! `[256, 16]` BF16 weight tile and multiplied by the `[32, 256]` BF16 activation tile with F32
//! accumulation, the prompt GEMM numeric contract of the other weight formats.
#[cutile::module]
pub mod byte {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tile_1, Tile_2, Tile_3, bf16, constant, convert_scalar, convert_tile,
        get_tile_block_id, mmaf, reduce_sum,
    };

    /// One output tile of sixteen rows: `factor * (code - ZERO)`. `K` is the reduction length.
    #[cutile::entry()]
    fn gemv_zero<
        const K: i32,
        const BLOCKS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const TERMS: i32,
        const ZERO: i32,
    >(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let cp = codes.partition(shape![4096]);
        let fp = factors.partition(shape![TERMS]);
        let xp = x.partition(shape![256]);
        let zero: f32 = convert_scalar(ZERO);
        let mut sum: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let raw: Tile<u8, { [256, 16] }> = cp.load([index]).reshape(shape![256, 16]);
            let code: Tile<f32, { [256, 16] }> =
                convert_tile(raw) - zero.broadcast(shape![256, 16]);
            let activation: Tile<f32, { [256, 16] }> = xp
                .load([block])
                .reshape(shape![256, 1])
                .broadcast(shape![256, 16]);
            let product: Tile<f32, { [GROUPS, GROUP, 16] }> =
                (code * activation).reshape(shape![GROUPS, GROUP, 16]);
            let factor: Tile<f32, { [GROUPS, 16] }> = fp.load([index]).reshape(shape![GROUPS, 16]);
            let scaled: Tile<f32, { [GROUPS, GROUP, 16] }> = product
                * factor
                    .reshape(shape![GROUPS, 1, 16])
                    .broadcast(shape![GROUPS, GROUP, 16]);
            let by_group: Tile<f32, { [GROUP, 16] }> = reduce_sum(scaled, 0i32);
            sum = sum + reduce_sum(by_group, 0i32);
        }
        out.store(sum);
    }

    /// One output tile of sixteen rows: `factor * code - bias`.
    #[cutile::entry()]
    fn gemv<
        const K: i32,
        const BLOCKS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const TERMS: i32,
    >(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
        biases: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let cp = codes.partition(shape![4096]);
        let fp = factors.partition(shape![TERMS]);
        let bp = biases.partition(shape![TERMS]);
        let xp = x.partition(shape![256]);
        let mut sum: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let raw: Tile<u8, { [256, 16] }> = cp.load([index]).reshape(shape![256, 16]);
            let code: Tile<f32, { [256, 16] }> = convert_tile(raw);
            let code: Tile<f32, { [GROUPS, GROUP, 16] }> = code.reshape(shape![GROUPS, GROUP, 16]);
            let activation: Tile<f32, { [GROUPS, GROUP, 16] }> = xp
                .load([block])
                .reshape(shape![256, 1])
                .broadcast(shape![256, 16])
                .reshape(shape![GROUPS, GROUP, 16]);
            let factor: Tile<f32, { [GROUPS, GROUP, 16] }> = fp
                .load([index])
                .reshape(shape![GROUPS, 1, 16])
                .broadcast(shape![GROUPS, GROUP, 16]);
            let bias: Tile<f32, { [GROUPS, GROUP, 16] }> = bp
                .load([index])
                .reshape(shape![GROUPS, 1, 16])
                .broadcast(shape![GROUPS, GROUP, 16]);
            let dequant = factor * code - bias;
            let by_group: Tile<f32, { [GROUP, 16] }> = reduce_sum(dequant * activation, 0i32);
            sum = sum + reduce_sum(by_group, 0i32);
        }
        out.store(sum);
    }

    /// `ROWS` activation rows of one sixteen-output tile: `factor * (code - ZERO)`. `SPANS = K / 256`.
    #[cutile::entry()]
    fn gemm_zero<
        const ROWS: i32,
        const K: i32,
        const SPANS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const TERMS: i32,
        const ZERO: i32,
    >(
        out: &mut Tensor<f32, { [ROWS, 16] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![ROWS, 256]);
        let cp = codes.partition(shape![4096]);
        let fp = factors.partition(shape![TERMS]);
        let zero: f32 = convert_scalar(ZERO);
        let mut acc: Tile<f32, { [ROWS, 16] }> = constant(0.0f32, shape![ROWS, 16]);
        for span in 0i32..SPANS {
            let index = pid.1 * SPANS + span;
            let raw: Tile<u8, { [256, 16] }> = cp.load([index]).reshape(shape![256, 16]);
            let code: Tile<f32, { [256, 16] }> =
                convert_tile(raw) - zero.broadcast(shape![256, 16]);
            let factor: Tile<f32, { [256, 16] }> = fp
                .load([index])
                .reshape(shape![GROUPS, 1, 16])
                .broadcast(shape![GROUPS, GROUP, 16])
                .reshape(shape![256, 16]);
            let weight: Tile<bf16, { [256, 16] }> = convert_tile(factor * code);
            let input: Tile<bf16, { [ROWS, 256] }> = convert_tile(xp.load([pid.0, span]));
            acc = mmaf(input, weight, acc);
        }
        out.store(acc);
    }

    /// `ROWS` activation rows of one sixteen-output tile: `factor * code - bias`.
    #[cutile::entry()]
    fn gemm<
        const ROWS: i32,
        const K: i32,
        const SPANS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const TERMS: i32,
    >(
        out: &mut Tensor<f32, { [ROWS, 16] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
        biases: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![ROWS, 256]);
        let cp = codes.partition(shape![4096]);
        let fp = factors.partition(shape![TERMS]);
        let bp = biases.partition(shape![TERMS]);
        let mut acc: Tile<f32, { [ROWS, 16] }> = constant(0.0f32, shape![ROWS, 16]);
        for span in 0i32..SPANS {
            let index = pid.1 * SPANS + span;
            let raw: Tile<u8, { [256, 16] }> = cp.load([index]).reshape(shape![256, 16]);
            let code: Tile<f32, { [256, 16] }> = convert_tile(raw);
            let factor: Tile<f32, { [256, 16] }> = fp
                .load([index])
                .reshape(shape![GROUPS, 1, 16])
                .broadcast(shape![GROUPS, GROUP, 16])
                .reshape(shape![256, 16]);
            let bias: Tile<f32, { [256, 16] }> = bp
                .load([index])
                .reshape(shape![GROUPS, 1, 16])
                .broadcast(shape![GROUPS, GROUP, 16])
                .reshape(shape![256, 16]);
            let weight: Tile<bf16, { [256, 16] }> = convert_tile(factor * code - bias);
            let input: Tile<bf16, { [ROWS, 256] }> = convert_tile(xp.load([pid.0, span]));
            acc = mmaf(input, weight, acc);
        }
        out.store(acc);
    }
}
