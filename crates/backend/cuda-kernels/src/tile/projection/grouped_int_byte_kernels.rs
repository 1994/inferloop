//! Byte-per-code grouped-int GEMV over the 16-row tile-major layout.
//!
//! A 256-value span stores `[256, 16]` codes, sixteen rows per value, with `[GROUPS, 16]`
//! per-group terms (`GROUPS * GROUP` values per step). The GEMV reads the codes as `u32` words:
//! word `w` of a value carries rows `4w..4w + 4`, one per byte, decoded along the last
//! tile axis into one `[VALUES, W, 4]` accumulator per activation row. A CTA covers `W` of the four words, so a tile can spread over several
//! CTAs without a cross-CTA reduction. A byte becomes a float exactly by OR-ing it into the mantissa of `2^23`
//! and subtracting `2^23`, which keeps integer conversions out of the loop. Each step covers
//! `VALUES` values. `gemv_zero` subtracts the format's constant zero point; `gemv` subtracts a
//! per-group bias. Both carry `R` activation rows per CTA and decode each word once for all of them.
//!
//! `gemm` / `gemm_zero` cover `ROWS` activation rows per CTA: each span is decoded once into a
//! `[256, 16]` BF16 weight tile and multiplied by the `[32, 256]` BF16 activation tile with F32
//! accumulation, the prompt GEMM numeric contract of the other weight formats.
#[cutile::module]
pub mod byte {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, Shape_4, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tile_1, Tile_2, Tile_3, Tile_4, andi, bf16, bitcast, constant,
        convert_scalar, convert_tile, get_tile_block_id, iota, mmaf, ori, reduce_sum, shri,
    };

    /// Bits of the float `2^23`, whose mantissa holds a small integer exactly.
    const MAGIC_BITS: u32 = 0x4b00_0000;
    /// `2^23`, subtracted to recover the integer from [`MAGIC_BITS`].
    const MAGIC: f32 = 8_388_608.0;

    /// `OUT = 4 * W` rows of a sixteen-row tile for `R` activation rows, `factor * (code - ZERO)`:
    /// CTA `(r, p)` covers row block `r` and part `p % PARTS` of tile `p / PARTS`. Each of the
    /// `BLOCKS` steps reads `W` words of `VALUES` values and the terms of `GROUPS` groups of `GROUP`
    /// values.
    #[cutile::entry()]
    fn gemv_zero<
        const R: i32,
        const K: i32,
        const BLOCKS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const ZERO: i32,
        const VALUES: i32,
        const W: i32,
        const PARTS: i32,
        const OUT: i32,
    >(
        out: &mut Tensor<f32, { [R, OUT] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let cp = codes.partition(shape![VALUES, W]);
        let fp = factors.partition(shape![GROUPS, OUT]);
        let tile = pid.1 / PARTS;
        let part = pid.1 % PARTS;
        let xp = x.partition(shape![R, VALUES]);
        let zero: f32 = convert_scalar(ZERO);
        let shift: f32 = MAGIC + zero;
        let offset = shift.broadcast(shape![VALUES, W, 4]);
        let mask = (0xffu32).broadcast(shape![VALUES, W, 4]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W, 4]);
        // The code of output `4w + b` is byte `b` of word `w`.
        let byte: Tile<i32, { [4] }> = iota(shape![4]) * 8i32.broadcast(shape![4]);
        let shift: Tile<u32, { [VALUES, W, 4] }> = bitcast(
            byte.reshape(shape![1, 1, 4])
                .broadcast(shape![VALUES, W, 4]),
        );
        let mut acc: Tile<f32, { [R, VALUES, W, 4] }> = constant(0.0, shape![R, VALUES, W, 4]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [VALUES, W, 4] }> = cp
                .load([index, part])
                .reshape(shape![VALUES, W, 1])
                .broadcast(shape![VALUES, W, 4]);
            let code: Tile<f32, { [VALUES, W, 4] }> =
                bitcast(ori(andi(shri(words, shift), mask), magic));
            let factor: Tile<f32, { [VALUES, W, 4] }> = fp
                .load([index, part])
                .reshape(shape![GROUPS, 1, W, 4])
                .broadcast(shape![GROUPS, GROUP, W, 4])
                .reshape(shape![VALUES, W, 4]);
            // The weight once, multiplied by every row in the single-row order.
            let weight: Tile<f32, { [VALUES, W, 4] }> = (code - offset) * factor;
            let weight: Tile<f32, { [R, VALUES, W, 4] }> = weight
                .reshape(shape![1, VALUES, W, 4])
                .broadcast(shape![R, VALUES, W, 4]);
            let activation: Tile<f32, { [R, VALUES, W, 4] }> = xp
                .load([pid.0, block])
                .reshape(shape![R, VALUES, 1, 1])
                .broadcast(shape![R, VALUES, W, 4]);
            acc = acc + weight * activation;
        }
        let sums: Tile<f32, { [R, W, 4] }> = reduce_sum(acc, 1i32);
        out.store(sums.reshape(shape![R, OUT]));
    }

    /// `OUT = 4 * W` rows of a sixteen-row tile for `R` activation rows, `factor * code - bias`,
    /// partitioned as in [`gemv_zero`].
    #[cutile::entry()]
    fn gemv<
        const R: i32,
        const K: i32,
        const BLOCKS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const VALUES: i32,
        const W: i32,
        const PARTS: i32,
        const OUT: i32,
    >(
        out: &mut Tensor<f32, { [R, OUT] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
        biases: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let cp = codes.partition(shape![VALUES, W]);
        let fp = factors.partition(shape![GROUPS, OUT]);
        let bp = biases.partition(shape![GROUPS, OUT]);
        let tile = pid.1 / PARTS;
        let part = pid.1 % PARTS;
        let xp = x.partition(shape![R, VALUES]);
        let offset = MAGIC.broadcast(shape![VALUES, W, 4]);
        let mask = (0xffu32).broadcast(shape![VALUES, W, 4]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W, 4]);
        // The code of output `4w + b` is byte `b` of word `w`.
        let byte: Tile<i32, { [4] }> = iota(shape![4]) * 8i32.broadcast(shape![4]);
        let shift: Tile<u32, { [VALUES, W, 4] }> = bitcast(
            byte.reshape(shape![1, 1, 4])
                .broadcast(shape![VALUES, W, 4]),
        );
        let mut acc: Tile<f32, { [R, VALUES, W, 4] }> = constant(0.0, shape![R, VALUES, W, 4]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [VALUES, W, 4] }> = cp
                .load([index, part])
                .reshape(shape![VALUES, W, 1])
                .broadcast(shape![VALUES, W, 4]);
            let code: Tile<f32, { [VALUES, W, 4] }> =
                bitcast(ori(andi(shri(words, shift), mask), magic));
            let factor: Tile<f32, { [VALUES, W, 4] }> = fp
                .load([index, part])
                .reshape(shape![GROUPS, 1, W, 4])
                .broadcast(shape![GROUPS, GROUP, W, 4])
                .reshape(shape![VALUES, W, 4]);
            // The weight once, multiplied by every row in the single-row order.
            let weight: Tile<f32, { [VALUES, W, 4] }> = (code - offset) * factor
                - bp.load([index, part])
                    .reshape(shape![GROUPS, 1, W, 4])
                    .broadcast(shape![GROUPS, GROUP, W, 4])
                    .reshape(shape![VALUES, W, 4]);
            let weight: Tile<f32, { [R, VALUES, W, 4] }> = weight
                .reshape(shape![1, VALUES, W, 4])
                .broadcast(shape![R, VALUES, W, 4]);
            let activation: Tile<f32, { [R, VALUES, W, 4] }> = xp
                .load([pid.0, block])
                .reshape(shape![R, VALUES, 1, 1])
                .broadcast(shape![R, VALUES, W, 4]);
            acc = acc + weight * activation;
        }
        let sums: Tile<f32, { [R, W, 4] }> = reduce_sum(acc, 1i32);
        out.store(sums.reshape(shape![R, OUT]));
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
