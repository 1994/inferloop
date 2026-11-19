//! Grouped-int kernels for 5- and 6-bit codes split into a nibble plane and a high plane over the
//! 16-row tile-major layout.
//!
//! The nibble plane is laid out as in the nibble kernels; the `HIGH` bits above each nibble sit
//! in a plane after the matrix's nibbles, `[WIDE = 32 * HIGH, 4]` words per span (see
//! `CodePacking::NibbleHigh`). The GEMV reads both planes as `u32` words and decodes all four
//! rows of a word along the last tile axis, so a CTA covering `W` of a tile's four words keeps one
//! `[VALUES, W, 4]` accumulator. A code becomes a float exactly by OR-ing it into the mantissa of
//! `2^23` and subtracting `2^23`. Each step covers `SPANS` spans. `gemv_zero` subtracts the
//! format's constant zero point; `gemv` subtracts a per-group bias.
//!
//! `gemm` / `gemm_zero` cover `ROWS` activation rows per CTA: each span is decoded once into a
//! `[256, 16]` BF16 weight tile and multiplied by the `[32, 256]` BF16 activation tile with F32
//! accumulation, the prompt GEMM numeric contract of the other weight formats.
#[cutile::module]
pub mod high {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, Shape_4, Shape_5,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tile_1, Tile_2, Tile_3, Tile_4, Tile_5, andi,
        bf16, bitcast, cat, constant, convert_scalar, convert_tile, get_tile_block_id, iota, mmaf,
        ori, overflow, reduce_sum, shli, shri,
    };

    /// Bits of the float `2^23`, whose mantissa holds a small integer exactly.
    const MAGIC_BITS: u32 = 0x4b00_0000;
    /// `2^23`, subtracted to recover the integer from [`MAGIC_BITS`].
    const MAGIC: f32 = 8_388_608.0;

    /// `factor * (code - ZERO)` over `OUT = 4 * W` rows of a sixteen-row tile: CTA `p` covers
    /// part `p % PARTS` of tile `p / PARTS`. Each of the `BLOCKS` steps reads `SPANS` spans:
    /// `LANES = 128 * SPANS` nibble rows of `NIBBLES = 4 * SPANS` 64-value groups, `PLANE =
    /// WIDE * SPANS` high rows of `SLOTS = 256 / WIDE` slots, and the terms of `GROUPS` groups of
    /// `GROUP` values. `MASK` keeps the `HIGH` bits of one code.
    #[cutile::entry()]
    fn gemv_zero<
        const K: i32,
        const BLOCKS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const ZERO: i32,
        const VALUES: i32,
        const SPANS: i32,
        const NIBBLES: i32,
        const LANES: i32,
        const HIGH: i32,
        const MASK: i32,
        const SLOTS: i32,
        const WIDE: i32,
        const PLANE: i32,
        const W: i32,
        const PARTS: i32,
        const OUT: i32,
    >(
        out: &mut Tensor<f32, { [OUT] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        high: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![VALUES]);
        let cp = codes.partition(shape![LANES, W]);
        let hp = high.partition(shape![PLANE, W]);
        let fp = factors.partition(shape![GROUPS, OUT]);
        let tile = pid.0 / PARTS;
        let part = pid.0 % PARTS;
        let zero: f32 = convert_scalar(ZERO);
        let shift: f32 = MAGIC + zero;
        let offset = shift.broadcast(shape![VALUES, W, 4]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W, 4]);
        let nibble = (0x0fu32).broadcast(shape![NIBBLES, 2, 32, W, 4]);
        let mask: Tile<u32, { [SPANS, SLOTS, WIDE, W, 4] }> =
            bitcast(MASK.broadcast(shape![SPANS, SLOTS, WIDE, W, 4]));
        let four = (4u32).broadcast(shape![SPANS, SLOTS, WIDE, W, 4]);
        // Nibble `h` of row `4w + b` sits at bit `8b + 4h` of word `w`.
        let byte: Tile<i32, { [4] }> = iota(shape![4]) * 8i32.broadcast(shape![4]);
        let half: Tile<i32, { [2] }> = iota(shape![2]) * 4i32.broadcast(shape![2]);
        let nibble_shift: Tile<u32, { [NIBBLES, 2, 32, W, 4] }> = bitcast(
            byte.reshape(shape![1, 1, 1, 1, 4])
                .broadcast(shape![NIBBLES, 2, 32, W, 4])
                + half
                    .reshape(shape![1, 2, 1, 1, 1])
                    .broadcast(shape![NIBBLES, 2, 32, W, 4]),
        );
        // Slot `k` of row `4w + b` sits at bit `4 * HIGH * k + HIGH * b` of high word `w`.
        let slot: Tile<i32, { [SLOTS] }> =
            iota(shape![SLOTS]) * (4 * HIGH).broadcast(shape![SLOTS]);
        let row: Tile<i32, { [4] }> = iota(shape![4]) * HIGH.broadcast(shape![4]);
        let high_shift: Tile<u32, { [SPANS, SLOTS, WIDE, W, 4] }> = bitcast(
            slot.reshape(shape![1, SLOTS, 1, 1, 1])
                .broadcast(shape![SPANS, SLOTS, WIDE, W, 4])
                + row
                    .reshape(shape![1, 1, 1, 1, 4])
                    .broadcast(shape![SPANS, SLOTS, WIDE, W, 4]),
        );
        let mut acc: Tile<f32, { [VALUES, W, 4] }> = constant(0.0, shape![VALUES, W, 4]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [NIBBLES, 2, 32, W, 4] }> = cp
                .load([index, part])
                .reshape(shape![NIBBLES, 1, 32, W, 1])
                .broadcast(shape![NIBBLES, 2, 32, W, 4]);
            let low: Tile<u32, { [VALUES, W, 4] }> =
                andi(shri(words, nibble_shift), nibble).reshape(shape![VALUES, W, 4]);
            let upper: Tile<u32, { [SPANS, SLOTS, WIDE, W, 4] }> = hp
                .load([index, part])
                .reshape(shape![SPANS, 1, WIDE, W, 1])
                .broadcast(shape![SPANS, SLOTS, WIDE, W, 4]);
            let top: Tile<u32, { [VALUES, W, 4] }> =
                shli(andi(shri(upper, high_shift), mask), four, overflow::NoWrap)
                    .reshape(shape![VALUES, W, 4]);
            let code: Tile<f32, { [VALUES, W, 4] }> = bitcast(ori(ori(low, top), magic));
            let factor: Tile<f32, { [VALUES, W, 4] }> = fp
                .load([index, part])
                .reshape(shape![GROUPS, 1, W, 4])
                .broadcast(shape![GROUPS, GROUP, W, 4])
                .reshape(shape![VALUES, W, 4]);
            let activation: Tile<f32, { [VALUES, W, 4] }> = xp
                .load([block])
                .reshape(shape![VALUES, 1, 1])
                .broadcast(shape![VALUES, W, 4]);
            acc = acc + (code - offset) * factor * activation;
        }
        let sums: Tile<f32, { [W, 4] }> = reduce_sum(acc, 0i32);
        out.store(sums.reshape(shape![OUT]));
    }

    /// `factor * code - bias` over `OUT = 4 * W` rows of a sixteen-row tile, partitioned as in
    /// [`gemv_zero`].
    #[cutile::entry()]
    fn gemv<
        const K: i32,
        const BLOCKS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const VALUES: i32,
        const SPANS: i32,
        const NIBBLES: i32,
        const LANES: i32,
        const HIGH: i32,
        const MASK: i32,
        const SLOTS: i32,
        const WIDE: i32,
        const PLANE: i32,
        const W: i32,
        const PARTS: i32,
        const OUT: i32,
    >(
        out: &mut Tensor<f32, { [OUT] }>,
        x: &Tensor<f32, { [K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        high: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
        biases: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![VALUES]);
        let cp = codes.partition(shape![LANES, W]);
        let hp = high.partition(shape![PLANE, W]);
        let fp = factors.partition(shape![GROUPS, OUT]);
        let bp = biases.partition(shape![GROUPS, OUT]);
        let tile = pid.0 / PARTS;
        let part = pid.0 % PARTS;
        let offset = MAGIC.broadcast(shape![VALUES, W, 4]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W, 4]);
        let nibble = (0x0fu32).broadcast(shape![NIBBLES, 2, 32, W, 4]);
        let mask: Tile<u32, { [SPANS, SLOTS, WIDE, W, 4] }> =
            bitcast(MASK.broadcast(shape![SPANS, SLOTS, WIDE, W, 4]));
        let four = (4u32).broadcast(shape![SPANS, SLOTS, WIDE, W, 4]);
        // Nibble `h` of row `4w + b` sits at bit `8b + 4h` of word `w`.
        let byte: Tile<i32, { [4] }> = iota(shape![4]) * 8i32.broadcast(shape![4]);
        let half: Tile<i32, { [2] }> = iota(shape![2]) * 4i32.broadcast(shape![2]);
        let nibble_shift: Tile<u32, { [NIBBLES, 2, 32, W, 4] }> = bitcast(
            byte.reshape(shape![1, 1, 1, 1, 4])
                .broadcast(shape![NIBBLES, 2, 32, W, 4])
                + half
                    .reshape(shape![1, 2, 1, 1, 1])
                    .broadcast(shape![NIBBLES, 2, 32, W, 4]),
        );
        // Slot `k` of row `4w + b` sits at bit `4 * HIGH * k + HIGH * b` of high word `w`.
        let slot: Tile<i32, { [SLOTS] }> =
            iota(shape![SLOTS]) * (4 * HIGH).broadcast(shape![SLOTS]);
        let row: Tile<i32, { [4] }> = iota(shape![4]) * HIGH.broadcast(shape![4]);
        let high_shift: Tile<u32, { [SPANS, SLOTS, WIDE, W, 4] }> = bitcast(
            slot.reshape(shape![1, SLOTS, 1, 1, 1])
                .broadcast(shape![SPANS, SLOTS, WIDE, W, 4])
                + row
                    .reshape(shape![1, 1, 1, 1, 4])
                    .broadcast(shape![SPANS, SLOTS, WIDE, W, 4]),
        );
        let mut acc: Tile<f32, { [VALUES, W, 4] }> = constant(0.0, shape![VALUES, W, 4]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [NIBBLES, 2, 32, W, 4] }> = cp
                .load([index, part])
                .reshape(shape![NIBBLES, 1, 32, W, 1])
                .broadcast(shape![NIBBLES, 2, 32, W, 4]);
            let low: Tile<u32, { [VALUES, W, 4] }> =
                andi(shri(words, nibble_shift), nibble).reshape(shape![VALUES, W, 4]);
            let upper: Tile<u32, { [SPANS, SLOTS, WIDE, W, 4] }> = hp
                .load([index, part])
                .reshape(shape![SPANS, 1, WIDE, W, 1])
                .broadcast(shape![SPANS, SLOTS, WIDE, W, 4]);
            let top: Tile<u32, { [VALUES, W, 4] }> =
                shli(andi(shri(upper, high_shift), mask), four, overflow::NoWrap)
                    .reshape(shape![VALUES, W, 4]);
            let code: Tile<f32, { [VALUES, W, 4] }> = bitcast(ori(ori(low, top), magic));
            let factor: Tile<f32, { [VALUES, W, 4] }> = fp
                .load([index, part])
                .reshape(shape![GROUPS, 1, W, 4])
                .broadcast(shape![GROUPS, GROUP, W, 4])
                .reshape(shape![VALUES, W, 4]);
            let bias: Tile<f32, { [VALUES, W, 4] }> = bp
                .load([index, part])
                .reshape(shape![GROUPS, 1, W, 4])
                .broadcast(shape![GROUPS, GROUP, W, 4])
                .reshape(shape![VALUES, W, 4]);
            let activation: Tile<f32, { [VALUES, W, 4] }> = xp
                .load([block])
                .reshape(shape![VALUES, 1, 1])
                .broadcast(shape![VALUES, W, 4]);
            acc = acc + ((code - offset) * factor - bias) * activation;
        }
        let sums: Tile<f32, { [W, 4] }> = reduce_sum(acc, 0i32);
        out.store(sums.reshape(shape![OUT]));
    }

    /// `ROWS` activation rows of one sixteen-output tile: `factor * code - bias`. `SPANS = K / 256`.
    #[cutile::entry()]
    fn gemm<
        const ROWS: i32,
        const K: i32,
        const SPANS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const TERMS: i32,
        const HIGH: i32,
        const MASK: i32,
        const SLOTS: i32,
        const WIDE: i32,
    >(
        out: &mut Tensor<f32, { [ROWS, 16] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u8, { [-1] }>,
        high: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1] }>,
        biases: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![ROWS, 256]);
        let cp = codes.partition(shape![2048]);
        let hp = high.partition(shape![WIDE, 4]);
        let fp = factors.partition(shape![TERMS]);
        let bp = biases.partition(shape![TERMS]);
        let mask: Tile<u32, { [SLOTS, WIDE, 4, 4] }> =
            bitcast(MASK.broadcast(shape![SLOTS, WIDE, 4, 4]));
        let magic = MAGIC_BITS.broadcast(shape![256, 16]);
        // Sixteen per high bit, once the `2^23` the mantissa trick adds is gone.
        let magic_value = MAGIC.broadcast(shape![256, 16]);
        let lift = (16.0f32).broadcast(shape![256, 16]);
        let offset = constant(0.0f32, shape![256, 16]);
        let slot: Tile<i32, { [SLOTS] }> =
            iota(shape![SLOTS]) * (4 * HIGH).broadcast(shape![SLOTS]);
        let row: Tile<i32, { [4] }> = iota(shape![4]) * HIGH.broadcast(shape![4]);
        let high_shift: Tile<u32, { [SLOTS, WIDE, 4, 4] }> = bitcast(
            slot.reshape(shape![SLOTS, 1, 1, 1])
                .broadcast(shape![SLOTS, WIDE, 4, 4])
                + row
                    .reshape(shape![1, 1, 1, 4])
                    .broadcast(shape![SLOTS, WIDE, 4, 4]),
        );
        let mut acc: Tile<f32, { [ROWS, 16] }> = constant(0.0f32, shape![ROWS, 16]);
        for span in 0i32..SPANS {
            let index = pid.1 * SPANS + span;
            let packed: Tile<u8, { [4, 32, 16] }> = cp.load([index]).reshape(shape![4, 32, 16]);
            let low: Tile<f32, { [4, 32, 16] }> =
                convert_tile(andi(packed, (0x0fu8).broadcast(shape![4, 32, 16])));
            let high_nibble: Tile<f32, { [4, 32, 16] }> =
                convert_tile(shri(packed, (4u8).broadcast(shape![4, 32, 16])));
            let pair: Tile<f32, { [4, 2, 32, 16] }> = cat(
                low.reshape(shape![4, 1, 32, 16]),
                high_nibble.reshape(shape![4, 1, 32, 16]),
                1i32,
            );
            let upper: Tile<u32, { [SLOTS, WIDE, 4, 4] }> = hp
                .load([index, 0i32])
                .reshape(shape![1, WIDE, 4, 1])
                .broadcast(shape![SLOTS, WIDE, 4, 4]);
            let top: Tile<u32, { [256, 16] }> =
                andi(shri(upper, high_shift), mask).reshape(shape![256, 16]);
            let top: Tile<f32, { [256, 16] }> = bitcast(ori(top, magic));
            let values: Tile<f32, { [256, 16] }> =
                pair.reshape(shape![256, 16]) + (top - magic_value) * lift - offset;
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
            let weight: Tile<bf16, { [256, 16] }> = convert_tile(factor * values - bias);
            let input: Tile<bf16, { [ROWS, 256] }> = convert_tile(xp.load([pid.0, span]));
            acc = mmaf(input, weight, acc);
        }
        out.store(acc);
    }

    /// `ROWS` activation rows of one sixteen-output tile: `factor * (code - ZERO)`.
    #[cutile::entry()]
    fn gemm_zero<
        const ROWS: i32,
        const K: i32,
        const SPANS: i32,
        const GROUPS: i32,
        const GROUP: i32,
        const TERMS: i32,
        const HIGH: i32,
        const MASK: i32,
        const SLOTS: i32,
        const WIDE: i32,
        const ZERO: i32,
    >(
        out: &mut Tensor<f32, { [ROWS, 16] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u8, { [-1] }>,
        high: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![ROWS, 256]);
        let cp = codes.partition(shape![2048]);
        let hp = high.partition(shape![WIDE, 4]);
        let fp = factors.partition(shape![TERMS]);
        let zero: f32 = convert_scalar(ZERO);
        let mask: Tile<u32, { [SLOTS, WIDE, 4, 4] }> =
            bitcast(MASK.broadcast(shape![SLOTS, WIDE, 4, 4]));
        let magic = MAGIC_BITS.broadcast(shape![256, 16]);
        // Sixteen per high bit, once the `2^23` the mantissa trick adds is gone.
        let magic_value = MAGIC.broadcast(shape![256, 16]);
        let lift = (16.0f32).broadcast(shape![256, 16]);
        let offset = zero.broadcast(shape![256, 16]);
        let slot: Tile<i32, { [SLOTS] }> =
            iota(shape![SLOTS]) * (4 * HIGH).broadcast(shape![SLOTS]);
        let row: Tile<i32, { [4] }> = iota(shape![4]) * HIGH.broadcast(shape![4]);
        let high_shift: Tile<u32, { [SLOTS, WIDE, 4, 4] }> = bitcast(
            slot.reshape(shape![SLOTS, 1, 1, 1])
                .broadcast(shape![SLOTS, WIDE, 4, 4])
                + row
                    .reshape(shape![1, 1, 1, 4])
                    .broadcast(shape![SLOTS, WIDE, 4, 4]),
        );
        let mut acc: Tile<f32, { [ROWS, 16] }> = constant(0.0f32, shape![ROWS, 16]);
        for span in 0i32..SPANS {
            let index = pid.1 * SPANS + span;
            let packed: Tile<u8, { [4, 32, 16] }> = cp.load([index]).reshape(shape![4, 32, 16]);
            let low: Tile<f32, { [4, 32, 16] }> =
                convert_tile(andi(packed, (0x0fu8).broadcast(shape![4, 32, 16])));
            let high_nibble: Tile<f32, { [4, 32, 16] }> =
                convert_tile(shri(packed, (4u8).broadcast(shape![4, 32, 16])));
            let pair: Tile<f32, { [4, 2, 32, 16] }> = cat(
                low.reshape(shape![4, 1, 32, 16]),
                high_nibble.reshape(shape![4, 1, 32, 16]),
                1i32,
            );
            let upper: Tile<u32, { [SLOTS, WIDE, 4, 4] }> = hp
                .load([index, 0i32])
                .reshape(shape![1, WIDE, 4, 1])
                .broadcast(shape![SLOTS, WIDE, 4, 4]);
            let top: Tile<u32, { [256, 16] }> =
                andi(shri(upper, high_shift), mask).reshape(shape![256, 16]);
            let top: Tile<f32, { [256, 16] }> = bitcast(ori(top, magic));
            let values: Tile<f32, { [256, 16] }> =
                pair.reshape(shape![256, 16]) + (top - magic_value) * lift - offset;
            let factor: Tile<f32, { [256, 16] }> = fp
                .load([index])
                .reshape(shape![GROUPS, 1, 16])
                .broadcast(shape![GROUPS, GROUP, 16])
                .reshape(shape![256, 16]);
            let weight: Tile<bf16, { [256, 16] }> = convert_tile(factor * values);
            let input: Tile<bf16, { [ROWS, 256] }> = convert_tile(xp.load([pid.0, span]));
            acc = mmaf(input, weight, acc);
        }
        out.store(acc);
    }
}
