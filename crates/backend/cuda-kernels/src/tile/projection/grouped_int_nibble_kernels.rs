//! Nibble-packed grouped-int GEMV over the 16-row tile-major layout.
//!
//! Lane `l` of each 64-value group holds value `l` in its low nibble and value `l + 32` in its
//! high nibble, sixteen rows per byte run. The GEMV reads the codes as `u32` words: word `w` of a
//! lane carries rows `4w..4w + 4`, one per byte. Each step covers `G` 64-value groups; byte `b` of
//! every word feeds its own accumulator, so rows `4w + b` never need gathering until the final
//! sixteen sums. A nibble becomes a float exactly by OR-ing it into the mantissa of `2^23` and
//! subtracting `2^23`, which keeps integer conversions out of the loop. `gemv` subtracts a
//! per-group bias; `gemv_zero` subtracts the format's constant zero point instead.
//!
//! `gemm` / `gemm_zero` cover `ROWS` activation rows per CTA: each span is decoded once into a
//! `[256, 16]` BF16 weight tile and multiplied by the `[32, 256]` BF16 activation tile with F32
//! accumulation, the prompt GEMM numeric contract of the other weight formats.
#[cutile::module]
pub mod nibble {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, Shape_3, Shape_4,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tile_1, Tile_2, Tile_3, Tile_4, andi, bf16,
        bitcast, cat, constant, convert_scalar, convert_tile, extract, get_tile_block_id, mmaf,
        ori, reduce_sum, shri,
    };

    /// Bits of the float `2^23`, whose mantissa holds a small integer exactly.
    const MAGIC_BITS: u32 = 0x4b00_0000;
    /// `2^23`, subtracted to recover the integer from [`MAGIC_BITS`].
    const MAGIC: f32 = 8_388_608.0;

    /// Sixteen row sums from the four byte-position accumulators: row `4w + b` is word `w` of
    /// accumulator `b`.
    fn row_sums<const G: i32>(
        first: Tile<f32, { [G, 32, 4] }>,
        second: Tile<f32, { [G, 32, 4] }>,
        third: Tile<f32, { [G, 32, 4] }>,
        fourth: Tile<f32, { [G, 32, 4] }>,
    ) -> Tile<f32, { [4, 4] }> {
        let a: Tile<f32, { [32, 4] }> = reduce_sum(first, 0i32);
        let a: Tile<f32, { [4] }> = reduce_sum(a, 0i32);
        let b: Tile<f32, { [32, 4] }> = reduce_sum(second, 0i32);
        let b: Tile<f32, { [4] }> = reduce_sum(b, 0i32);
        let c: Tile<f32, { [32, 4] }> = reduce_sum(third, 0i32);
        let c: Tile<f32, { [4] }> = reduce_sum(c, 0i32);
        let d: Tile<f32, { [32, 4] }> = reduce_sum(fourth, 0i32);
        let d: Tile<f32, { [4] }> = reduce_sum(d, 0i32);
        let low: Tile<f32, { [4, 2] }> =
            cat(a.reshape(shape![4, 1]), b.reshape(shape![4, 1]), 1i32);
        let high: Tile<f32, { [4, 2] }> =
            cat(c.reshape(shape![4, 1]), d.reshape(shape![4, 1]), 1i32);
        cat(low, high, 1i32)
    }

    /// Every value of a group shares its bias, so a step subtracts `bias * sum(x)` per group:
    /// the step's `[4, 4]` (word, byte) row offsets.
    fn group_offsets<const G: i32>(
        bias: Tile<f32, { [G, 2, 4, 4] }>,
        inputs: Tile<f32, { [G, 2, 32] }>,
    ) -> Tile<f32, { [4, 4] }> {
        let totals: Tile<f32, { [G, 2] }> = reduce_sum(inputs, 2i32);
        let weighted = bias
            * totals
                .reshape(shape![G, 2, 1, 1])
                .broadcast(shape![G, 2, 4, 4]);
        let by_half: Tile<f32, { [2, 4, 4] }> = reduce_sum(weighted, 0i32);
        reduce_sum(by_half, 0i32)
    }

    /// One output tile of sixteen rows: `factor * code - bias`. `K` is the reduction length; each of the
    /// `BLOCKS` steps reads `WORDS` code words of `G` groups, `VALUES` activations and `TERMS`
    /// terms.
    #[cutile::entry()]
    fn gemv<
        const K: i32,
        const BLOCKS: i32,
        const G: i32,
        const WORDS: i32,
        const VALUES: i32,
        const TERMS: i32,
    >(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u32, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
        biases: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![VALUES]);
        let cp = codes.partition(shape![WORDS]);
        let fp = factors.partition(shape![TERMS]);
        let bp = biases.partition(shape![TERMS]);
        let offset = MAGIC.broadcast(shape![G, 32, 4]);
        let nibble = (0x0fu32).broadcast(shape![G, 32, 4]);
        let magic = MAGIC_BITS.broadcast(shape![G, 32, 4]);
        let first = 0i32.broadcast(shape![]);
        let second = 1i32.broadcast(shape![]);
        let mut offsets: Tile<f32, { [4, 4] }> = constant(0.0, shape![4, 4]);
        let mut acc0: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        let mut acc1: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        let mut acc2: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        let mut acc3: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let words: Tile<u32, { [G, 32, 4] }> = cp.load([index]).reshape(shape![G, 32, 4]);
            let inputs: Tile<f32, { [G, 2, 32] }> = xp.load([block]).reshape(shape![G, 2, 32]);
            let x_low: Tile<f32, { [G, 1, 32] }> = extract(inputs, [first, first, first]);
            let x_low: Tile<f32, { [G, 32, 4] }> =
                x_low.reshape(shape![G, 32, 1]).broadcast(shape![G, 32, 4]);
            let x_high: Tile<f32, { [G, 1, 32] }> = extract(inputs, [first, second, first]);
            let x_high: Tile<f32, { [G, 32, 4] }> =
                x_high.reshape(shape![G, 32, 1]).broadcast(shape![G, 32, 4]);
            let factor: Tile<f32, { [G, 2, 4, 4] }> = fp.load([index]).reshape(shape![G, 2, 4, 4]);
            offsets = offsets + group_offsets(bp.load([index]).reshape(shape![G, 2, 4, 4]), inputs);
            // Byte 0 of each word: rows `4w + 0`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (0u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (4u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, first]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc0 = acc0
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
            // Byte 1 of each word: rows `4w + 1`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (8u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (12u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, 1i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc1 = acc1
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
            // Byte 2 of each word: rows `4w + 2`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (16u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (20u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, 2i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc2 = acc2
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
            // Byte 3 of each word: rows `4w + 3`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (24u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (28u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, 3i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc3 = acc3
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
        }
        let rows: Tile<f32, { [4, 4] }> = row_sums(acc0, acc1, acc2, acc3);
        let total: Tile<f32, { [16] }> = (rows - offsets).reshape(shape![16]);
        out.store(total);
    }

    /// One output tile of sixteen rows: `factor * (code - ZERO)`. `K` is the reduction length; each of the
    /// `BLOCKS` steps reads `WORDS` code words of `G` groups, `VALUES` activations and `TERMS`
    /// terms.
    #[cutile::entry()]
    fn gemv_zero<
        const K: i32,
        const BLOCKS: i32,
        const ZERO: i32,
        const G: i32,
        const WORDS: i32,
        const VALUES: i32,
        const TERMS: i32,
    >(
        out: &mut Tensor<f32, { [16] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u32, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![VALUES]);
        let cp = codes.partition(shape![WORDS]);
        let fp = factors.partition(shape![TERMS]);
        let zero: f32 = convert_scalar(ZERO);
        let shift: f32 = MAGIC + zero;
        let offset = shift.broadcast(shape![G, 32, 4]);
        let nibble = (0x0fu32).broadcast(shape![G, 32, 4]);
        let magic = MAGIC_BITS.broadcast(shape![G, 32, 4]);
        let first = 0i32.broadcast(shape![]);
        let second = 1i32.broadcast(shape![]);
        let mut acc0: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        let mut acc1: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        let mut acc2: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        let mut acc3: Tile<f32, { [G, 32, 4] }> = constant(0.0, shape![G, 32, 4]);
        for block in 0i32..BLOCKS {
            let index = pid.0 * BLOCKS + block;
            let words: Tile<u32, { [G, 32, 4] }> = cp.load([index]).reshape(shape![G, 32, 4]);
            let inputs: Tile<f32, { [G, 2, 32] }> = xp.load([block]).reshape(shape![G, 2, 32]);
            let x_low: Tile<f32, { [G, 1, 32] }> = extract(inputs, [first, first, first]);
            let x_low: Tile<f32, { [G, 32, 4] }> =
                x_low.reshape(shape![G, 32, 1]).broadcast(shape![G, 32, 4]);
            let x_high: Tile<f32, { [G, 1, 32] }> = extract(inputs, [first, second, first]);
            let x_high: Tile<f32, { [G, 32, 4] }> =
                x_high.reshape(shape![G, 32, 1]).broadcast(shape![G, 32, 4]);
            let factor: Tile<f32, { [G, 2, 4, 4] }> = fp.load([index]).reshape(shape![G, 2, 4, 4]);
            // Byte 0 of each word: rows `4w + 0`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (0u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (4u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, first]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc0 = acc0
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
            // Byte 1 of each word: rows `4w + 1`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (8u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (12u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, 1i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc1 = acc1
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
            // Byte 2 of each word: rows `4w + 2`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (16u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (20u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, 2i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc2 = acc2
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
            // Byte 3 of each word: rows `4w + 3`.
            let low: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (24u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let high: Tile<f32, { [G, 32, 4] }> = bitcast(ori(
                andi(shri(words, (28u32).broadcast(shape![G, 32, 4])), nibble),
                magic,
            ));
            let at_byte: Tile<f32, { [G, 2, 4, 1] }> =
                extract(factor, [first, first, first, 3i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, 4] }> = at_byte.reshape(shape![G, 2, 4]);
            let low_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, 4] }> = extract(at_byte, [first, second, first]);
            acc3 = acc3
                + (low - offset) * low_factor.broadcast(shape![G, 32, 4]) * x_low
                + (high - offset) * high_factor.broadcast(shape![G, 32, 4]) * x_high;
        }
        let rows: Tile<f32, { [4, 4] }> = row_sums(acc0, acc1, acc2, acc3);
        let total: Tile<f32, { [16] }> = rows.reshape(shape![16]);
        out.store(total);
    }

    /// `ROWS` activation rows of one sixteen-output tile: `factor * code - bias`. `SPANS = K / 256`.
    #[cutile::entry()]
    fn gemm<const ROWS: i32, const K: i32, const SPANS: i32>(
        out: &mut Tensor<f32, { [ROWS, 16] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
        biases: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![ROWS, 256]);
        let cp = codes.partition(shape![2048]);
        let fp = factors.partition(shape![128]);
        let bp = biases.partition(shape![128]);
        let mut acc: Tile<f32, { [ROWS, 16] }> = constant(0.0f32, shape![ROWS, 16]);
        for span in 0i32..SPANS {
            let index = pid.1 * SPANS + span;
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
            let values: Tile<f32, { [256, 16] }> = pair.reshape(shape![256, 16]);
            let factor: Tile<f32, { [256, 16] }> = fp
                .load([index])
                .reshape(shape![4, 2, 16])
                .reshape(shape![4, 2, 1, 16])
                .broadcast(shape![4, 2, 32, 16])
                .reshape(shape![256, 16]);
            let bias: Tile<f32, { [256, 16] }> = bp
                .load([index])
                .reshape(shape![4, 2, 16])
                .reshape(shape![4, 2, 1, 16])
                .broadcast(shape![4, 2, 32, 16])
                .reshape(shape![256, 16]);
            let weight: Tile<bf16, { [256, 16] }> = convert_tile(factor * values - bias);
            let input: Tile<bf16, { [ROWS, 256] }> = convert_tile(xp.load([pid.0, span]));
            acc = mmaf(input, weight, acc);
        }
        out.store(acc);
    }

    /// `ROWS` activation rows of one sixteen-output tile: `factor * (code - ZERO)`.
    #[cutile::entry()]
    fn gemm_zero<const ROWS: i32, const K: i32, const SPANS: i32, const ZERO: i32>(
        out: &mut Tensor<f32, { [ROWS, 16] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u8, { [-1] }>,
        factors: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![ROWS, 256]);
        let cp = codes.partition(shape![2048]);
        let fp = factors.partition(shape![128]);
        let zero: f32 = convert_scalar(ZERO);
        let mut acc: Tile<f32, { [ROWS, 16] }> = constant(0.0f32, shape![ROWS, 16]);
        for span in 0i32..SPANS {
            let index = pid.1 * SPANS + span;
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
            let values: Tile<f32, { [256, 16] }> = pair.reshape(shape![256, 16]);
            let values = values - zero.broadcast(shape![256, 16]);
            let factor: Tile<f32, { [256, 16] }> = fp
                .load([index])
                .reshape(shape![4, 2, 16])
                .reshape(shape![4, 2, 1, 16])
                .broadcast(shape![4, 2, 32, 16])
                .reshape(shape![256, 16]);
            let weight: Tile<bf16, { [256, 16] }> = convert_tile(factor * values);
            let input: Tile<bf16, { [ROWS, 256] }> = convert_tile(xp.load([pid.0, span]));
            acc = mmaf(input, weight, acc);
        }
        out.store(acc);
    }
}
