//! Byte-per-code grouped-int GEMV over the 16-row tile-major layout.
//!
//! A 256-value span stores `[256, 16]` codes, sixteen rows per value, with `[GROUPS, 16]`
//! per-group terms (`GROUPS * GROUP` values per step). The GEMV reads the codes as `u32` words:
//! word `w` of a value carries rows `4w..4w + 4`, one per byte, and byte `b` of every word feeds
//! its own accumulator. A CTA covers `W` of the four words, so a tile can spread over several
//! CTAs without a cross-CTA reduction. A byte becomes a float exactly by OR-ing it into the mantissa of `2^23`
//! and subtracting `2^23`, which keeps integer conversions out of the loop. Each step covers
//! `VALUES` values. `gemv_zero` subtracts the format's constant zero point; `gemv` subtracts a
//! per-group bias.
//!
//! `gemm` / `gemm_zero` cover `ROWS` activation rows per CTA: each span is decoded once into a
//! `[256, 16]` BF16 weight tile and multiplied by the `[32, 256]` BF16 activation tile with F32
//! accumulation, the prompt GEMM numeric contract of the other weight formats.
#[cutile::module]
pub mod byte {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tile_1, Tile_2, Tile_3, andi, bf16, bitcast, cat, constant,
        convert_scalar, convert_tile, extract, get_tile_block_id, mmaf, ori, reduce_sum, shri,
    };

    /// Bits of the float `2^23`, whose mantissa holds a small integer exactly.
    const MAGIC_BITS: u32 = 0x4b00_0000;
    /// `2^23`, subtracted to recover the integer from [`MAGIC_BITS`].
    const MAGIC: f32 = 8_388_608.0;

    /// Store the `OUT = 4W` row sums: row `4w + b` is word `w` of accumulator `b`.
    fn store_rows<const VALUES: i32, const W: i32, const OUT: i32>(
        out: &mut Tensor<f32, { [OUT] }>,
        accumulators: [Tile<f32, { [VALUES, W] }>; 4],
    ) {
        let [first, second, third, fourth] = accumulators;
        let a: Tile<f32, { [W] }> = reduce_sum(first, 0i32);
        let b: Tile<f32, { [W] }> = reduce_sum(second, 0i32);
        let c: Tile<f32, { [W] }> = reduce_sum(third, 0i32);
        let d: Tile<f32, { [W] }> = reduce_sum(fourth, 0i32);
        let low: Tile<f32, { [W, 2] }> =
            cat(a.reshape(shape![W, 1]), b.reshape(shape![W, 1]), 1i32);
        let high: Tile<f32, { [W, 2] }> =
            cat(c.reshape(shape![W, 1]), d.reshape(shape![W, 1]), 1i32);
        let rows: Tile<f32, { [W, 4] }> = cat(low, high, 1i32);
        let total: Tile<f32, { [OUT] }> = rows.reshape(shape![OUT]);
        out.store(total);
    }

    /// `OUT = 4 * W` rows of a sixteen-row tile, `factor * (code - ZERO)`: CTA `p` covers part
    /// `p % PARTS` of tile `p / PARTS`. Each of the `BLOCKS` steps reads `W` words of `VALUES`
    /// values and the terms of `GROUPS` groups of `GROUP` values.
    #[cutile::entry()]
    fn gemv_zero<
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
        out: &mut Tensor<f32, { [OUT] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let cp = codes.partition(shape![VALUES, W]);
        let fp = factors.partition(shape![GROUPS, OUT]);
        let tile = pid.0 / PARTS;
        let part = pid.0 % PARTS;
        let xp = x.partition(shape![VALUES]);
        let zero: f32 = convert_scalar(ZERO);
        let shift: f32 = MAGIC + zero;
        let offset = shift.broadcast(shape![VALUES, W]);
        let mask = (0xffu32).broadcast(shape![VALUES, W]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W]);
        // Bit offsets of the codes after the first, in word order.
        let shifts = [
            (8u32).broadcast(shape![VALUES, W]),
            (16u32).broadcast(shape![VALUES, W]),
            (24u32).broadcast(shape![VALUES, W]),
        ];
        let first = 0i32.broadcast(shape![]);
        let mut acc0: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        let mut acc1: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        let mut acc2: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        let mut acc3: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [VALUES, W] }> =
                cp.load([index, part]).reshape(shape![VALUES, W]);
            let activation: Tile<f32, { [VALUES, W] }> = xp
                .load([block])
                .reshape(shape![VALUES, 1])
                .broadcast(shape![VALUES, W]);
            let factor: Tile<f32, { [GROUPS, W, 4] }> =
                fp.load([index, part]).reshape(shape![GROUPS, W, 4]);
            // Byte 0 of each word: rows `4w + 0`.
            let code: Tile<f32, { [VALUES, W] }> = bitcast(ori(andi(words, mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> = extract(factor, [first, first, first]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc0 = acc0 + (code - offset) * factor_at * activation;
            // Byte 1 of each word: rows `4w + 1`.
            let code: Tile<f32, { [VALUES, W] }> =
                bitcast(ori(andi(shri(words, shifts[0]), mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(factor, [first, first, 1i32.broadcast(shape![])]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc1 = acc1 + (code - offset) * factor_at * activation;
            // Byte 2 of each word: rows `4w + 2`.
            let code: Tile<f32, { [VALUES, W] }> =
                bitcast(ori(andi(shri(words, shifts[1]), mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(factor, [first, first, 2i32.broadcast(shape![])]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc2 = acc2 + (code - offset) * factor_at * activation;
            // Byte 3 of each word: rows `4w + 3`.
            let code: Tile<f32, { [VALUES, W] }> =
                bitcast(ori(andi(shri(words, shifts[2]), mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(factor, [first, first, 3i32.broadcast(shape![])]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc3 = acc3 + (code - offset) * factor_at * activation;
        }
        store_rows::<VALUES, W, OUT>(out, [acc0, acc1, acc2, acc3]);
    }

    /// `OUT = 4 * W` rows of a sixteen-row tile, `factor * code - bias`, partitioned as in
    /// [`gemv_zero`].
    #[cutile::entry()]
    fn gemv<
        const K: i32,
        const BLOCKS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const VALUES: i32,
        const W: i32,
        const PARTS: i32,
        const OUT: i32,
    >(
        out: &mut Tensor<f32, { [OUT] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
        biases: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let cp = codes.partition(shape![VALUES, W]);
        let fp = factors.partition(shape![GROUPS, OUT]);
        let tile = pid.0 / PARTS;
        let part = pid.0 % PARTS;
        let xp = x.partition(shape![VALUES]);
        let bp = biases.partition(shape![GROUPS, OUT]);
        let offset = MAGIC.broadcast(shape![VALUES, W]);
        let mask = (0xffu32).broadcast(shape![VALUES, W]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W]);
        // Bit offsets of the codes after the first, in word order.
        let shifts = [
            (8u32).broadcast(shape![VALUES, W]),
            (16u32).broadcast(shape![VALUES, W]),
            (24u32).broadcast(shape![VALUES, W]),
        ];
        let first = 0i32.broadcast(shape![]);
        let mut acc0: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        let mut acc1: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        let mut acc2: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        let mut acc3: Tile<f32, { [VALUES, W] }> = constant(0.0, shape![VALUES, W]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [VALUES, W] }> =
                cp.load([index, part]).reshape(shape![VALUES, W]);
            let activation: Tile<f32, { [VALUES, W] }> = xp
                .load([block])
                .reshape(shape![VALUES, 1])
                .broadcast(shape![VALUES, W]);
            let factor: Tile<f32, { [GROUPS, W, 4] }> =
                fp.load([index, part]).reshape(shape![GROUPS, W, 4]);
            let bias: Tile<f32, { [GROUPS, W, 4] }> =
                bp.load([index, part]).reshape(shape![GROUPS, W, 4]);
            // Byte 0 of each word: rows `4w + 0`.
            let code: Tile<f32, { [VALUES, W] }> = bitcast(ori(andi(words, mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> = extract(factor, [first, first, first]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            let bias_byte: Tile<f32, { [GROUPS, W, 1] }> = extract(bias, [first, first, first]);
            let bias_at: Tile<f32, { [VALUES, W] }> = bias_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc0 = acc0 + ((code - offset) * factor_at - bias_at) * activation;
            // Byte 1 of each word: rows `4w + 1`.
            let code: Tile<f32, { [VALUES, W] }> =
                bitcast(ori(andi(shri(words, shifts[0]), mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(factor, [first, first, 1i32.broadcast(shape![])]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            let bias_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(bias, [first, first, 1i32.broadcast(shape![])]);
            let bias_at: Tile<f32, { [VALUES, W] }> = bias_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc1 = acc1 + ((code - offset) * factor_at - bias_at) * activation;
            // Byte 2 of each word: rows `4w + 2`.
            let code: Tile<f32, { [VALUES, W] }> =
                bitcast(ori(andi(shri(words, shifts[1]), mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(factor, [first, first, 2i32.broadcast(shape![])]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            let bias_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(bias, [first, first, 2i32.broadcast(shape![])]);
            let bias_at: Tile<f32, { [VALUES, W] }> = bias_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc2 = acc2 + ((code - offset) * factor_at - bias_at) * activation;
            // Byte 3 of each word: rows `4w + 3`.
            let code: Tile<f32, { [VALUES, W] }> =
                bitcast(ori(andi(shri(words, shifts[2]), mask), magic));
            let at_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(factor, [first, first, 3i32.broadcast(shape![])]);
            let factor_at: Tile<f32, { [VALUES, W] }> = at_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            let bias_byte: Tile<f32, { [GROUPS, W, 1] }> =
                extract(bias, [first, first, 3i32.broadcast(shape![])]);
            let bias_at: Tile<f32, { [VALUES, W] }> = bias_byte
                .reshape(shape![GROUPS, 1, W])
                .broadcast(shape![GROUPS, GROUP, W])
                .reshape(shape![VALUES, W]);
            acc3 = acc3 + ((code - offset) * factor_at - bias_at) * activation;
        }
        store_rows::<VALUES, W, OUT>(out, [acc0, acc1, acc2, acc3]);
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
