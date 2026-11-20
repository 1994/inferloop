//! Nibble-packed grouped-int GEMV over the 16-row tile-major layout.
//!
//! Lane `l` of each 64-value group holds value `l` in its low nibble and value `l + 32` in its
//! high nibble, sixteen rows per byte run. The GEMV reads the codes as `u32` words: word `w` of a
//! lane carries rows `4w..4w + 4`, one per byte. A CTA covers `W` of the four words, so a tile
//! can spread over several CTAs without a cross-CTA reduction. Each step covers `G` 64-value
//! groups; byte `b` of every word feeds its own accumulator, so rows `4w + b` never need
//! gathering until the final sums. A nibble becomes a float exactly by OR-ing it into the mantissa of `2^23` and
//! subtracting `2^23`, which keeps integer conversions out of the loop. `gemv` subtracts a
//! per-group bias; `gemv_zero` subtracts the format's constant zero point instead. Both carry `R`
//! activation rows per CTA and decode each word once for all of them.
//!
//! `gemm` / `gemm_zero` cover `ROWS` activation rows per CTA: each span is decoded once into a
//! `[256, 16]` BF16 weight tile and multiplied by the `[32, 256]` BF16 activation tile with F32
//! accumulation, the prompt GEMM numeric contract of the other weight formats.
#[cutile::module]
pub mod nibble {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, Shape_3, Shape_4, Shape_5,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tile_2, Tile_3, Tile_4, Tile_5, andi, bf16,
        bitcast, cat, constant, convert_scalar, convert_tile, extract, get_tile_block_id, mmaf,
        ori, reduce_sum, shri,
    };

    /// Bits of the float `2^23`, whose mantissa holds a small integer exactly.
    const MAGIC_BITS: u32 = 0x4b00_0000;
    /// `2^23`, subtracted to recover the integer from [`MAGIC_BITS`].
    const MAGIC: f32 = 8_388_608.0;

    /// Store `OUT = 4W` sums of each of the `R` rows less `offsets`: output `4w + b` is word `w`
    /// of accumulator `b`.
    fn store_rows<const R: i32, const G: i32, const W: i32, const OUT: i32>(
        out: &mut Tensor<f32, { [R, OUT] }>,
        accumulators: [Tile<f32, { [R, G, 32, W] }>; 4],
        offsets: Tile<f32, { [R, W, 4] }>,
    ) {
        let [first, second, third, fourth] = accumulators;
        let a: Tile<f32, { [R, 32, W] }> = reduce_sum(first, 1i32);
        let b: Tile<f32, { [R, 32, W] }> = reduce_sum(second, 1i32);
        let c: Tile<f32, { [R, 32, W] }> = reduce_sum(third, 1i32);
        let d: Tile<f32, { [R, 32, W] }> = reduce_sum(fourth, 1i32);
        let a: Tile<f32, { [R, W] }> = reduce_sum(a, 1i32);
        let b: Tile<f32, { [R, W] }> = reduce_sum(b, 1i32);
        let c: Tile<f32, { [R, W] }> = reduce_sum(c, 1i32);
        let d: Tile<f32, { [R, W] }> = reduce_sum(d, 1i32);
        let low: Tile<f32, { [R, W, 2] }> =
            cat(a.reshape(shape![R, W, 1]), b.reshape(shape![R, W, 1]), 2i32);
        let high: Tile<f32, { [R, W, 2] }> =
            cat(c.reshape(shape![R, W, 1]), d.reshape(shape![R, W, 1]), 2i32);
        let sums: Tile<f32, { [R, W, 4] }> = cat(low, high, 2i32);
        out.store((sums - offsets).reshape(shape![R, OUT]));
    }

    /// `OUT = 4 * W` outputs of a sixteen-row tile for `R` activation rows, `factor * code - bias`:
    /// CTA `(r, p)` covers row block `r` and part `p % PARTS` of tile `p / PARTS`. Each of the
    /// `BLOCKS` steps reads `W` words of `LANES = 32G` lanes, `VALUES = 64G` activations per row and
    /// `HALVES = 2G` half-group term rows; the codes are decoded once for every row.
    #[cutile::entry()]
    fn gemv<
        const R: i32,
        const K: i32,
        const BLOCKS: i32,
        const G: i32,
        const VALUES: i32,
        const W: i32,
        const PARTS: i32,
        const OUT: i32,
        const LANES: i32,
        const HALVES: i32,
    >(
        out: &mut Tensor<f32, { [R, OUT] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
        biases: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![R, VALUES]);
        let cp = codes.partition(shape![LANES, W]);
        let fp = factors.partition(shape![HALVES, OUT]);
        let bp = biases.partition(shape![HALVES, OUT]);
        let tile = pid.1 / PARTS;
        let part = pid.1 % PARTS;
        let offset = MAGIC.broadcast(shape![G, 32, W]);
        let nibble = (0x0fu32).broadcast(shape![G, 32, W]);
        let magic = MAGIC_BITS.broadcast(shape![G, 32, W]);
        // Bit offsets of the codes after the first, in word order.
        let shifts = [
            (4u32).broadcast(shape![G, 32, W]),
            (8u32).broadcast(shape![G, 32, W]),
            (12u32).broadcast(shape![G, 32, W]),
            (16u32).broadcast(shape![G, 32, W]),
            (20u32).broadcast(shape![G, 32, W]),
            (24u32).broadcast(shape![G, 32, W]),
            (28u32).broadcast(shape![G, 32, W]),
        ];
        let first = 0i32.broadcast(shape![]);
        let second = 1i32.broadcast(shape![]);
        let mut offsets: Tile<f32, { [R, G, 2, W, 4] }> = constant(0.0, shape![R, G, 2, W, 4]);
        let mut acc0: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        let mut acc1: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        let mut acc2: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        let mut acc3: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [G, 32, W] }> = cp.load([index, part]).reshape(shape![G, 32, W]);
            let inputs: Tile<f32, { [R, G, 2, 32] }> =
                xp.load([pid.0, block]).reshape(shape![R, G, 2, 32]);
            let x_low: Tile<f32, { [R, G, 1, 32] }> = extract(inputs, [first, first, first, first]);
            let x_low: Tile<f32, { [R, G, 32, W] }> = x_low
                .reshape(shape![R, G, 32, 1])
                .broadcast(shape![R, G, 32, W]);
            let x_high: Tile<f32, { [R, G, 1, 32] }> =
                extract(inputs, [first, first, second, first]);
            let x_high: Tile<f32, { [R, G, 32, W] }> = x_high
                .reshape(shape![R, G, 32, 1])
                .broadcast(shape![R, G, 32, W]);
            let factor: Tile<f32, { [G, 2, W, 4] }> =
                fp.load([index, part]).reshape(shape![G, 2, W, 4]);
            // Every value of a group shares its bias: subtract `bias * sum(x)` per group and row.
            let bias: Tile<f32, { [G, 2, W, 4] }> =
                bp.load([index, part]).reshape(shape![G, 2, W, 4]);
            let totals: Tile<f32, { [R, G, 2] }> = reduce_sum(inputs, 3i32);
            let bias: Tile<f32, { [R, G, 2, W, 4] }> = bias
                .reshape(shape![1, G, 2, W, 4])
                .broadcast(shape![R, G, 2, W, 4]);
            let totals: Tile<f32, { [R, G, 2, W, 4] }> = totals
                .reshape(shape![R, G, 2, 1, 1])
                .broadcast(shape![R, G, 2, W, 4]);
            offsets = offsets + bias * totals;
            // Byte 0 of each word: outputs `4w + 0`.
            let low: Tile<f32, { [G, 32, W] }> = bitcast(ori(andi(words, nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[0]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 0i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc0 = acc0 + low * x_low + high * x_high;
            // Byte 1 of each word: outputs `4w + 1`.
            let low: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[1]), nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[2]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 1i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc1 = acc1 + low * x_low + high * x_high;
            // Byte 2 of each word: outputs `4w + 2`.
            let low: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[3]), nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[4]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 2i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc2 = acc2 + low * x_low + high * x_high;
            // Byte 3 of each word: outputs `4w + 3`.
            let low: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[5]), nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[6]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 3i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc3 = acc3 + low * x_low + high * x_high;
        }
        let by_half: Tile<f32, { [R, 2, W, 4] }> = reduce_sum(offsets, 1i32);
        let by_row: Tile<f32, { [R, W, 4] }> = reduce_sum(by_half, 1i32);
        store_rows::<R, G, W, OUT>(out, [acc0, acc1, acc2, acc3], by_row);
    }

    /// `OUT = 4 * W` outputs of a sixteen-row tile for `R` activation rows, `factor * (code - ZERO)`,
    /// partitioned as in [`gemv`].
    #[cutile::entry()]
    fn gemv_zero<
        const R: i32,
        const K: i32,
        const BLOCKS: i32,
        const ZERO: i32,
        const G: i32,
        const VALUES: i32,
        const W: i32,
        const PARTS: i32,
        const OUT: i32,
        const LANES: i32,
        const HALVES: i32,
    >(
        out: &mut Tensor<f32, { [R, OUT] }>,
        x: &Tensor<f32, { [-1, K] }>,
        codes: &Tensor<u32, { [-1, 4] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![R, VALUES]);
        let cp = codes.partition(shape![LANES, W]);
        let fp = factors.partition(shape![HALVES, OUT]);
        let tile = pid.1 / PARTS;
        let part = pid.1 % PARTS;
        let zero: f32 = convert_scalar(ZERO);
        let shift: f32 = MAGIC + zero;
        let offset = shift.broadcast(shape![G, 32, W]);
        let nibble = (0x0fu32).broadcast(shape![G, 32, W]);
        let magic = MAGIC_BITS.broadcast(shape![G, 32, W]);
        // Bit offsets of the codes after the first, in word order.
        let shifts = [
            (4u32).broadcast(shape![G, 32, W]),
            (8u32).broadcast(shape![G, 32, W]),
            (12u32).broadcast(shape![G, 32, W]),
            (16u32).broadcast(shape![G, 32, W]),
            (20u32).broadcast(shape![G, 32, W]),
            (24u32).broadcast(shape![G, 32, W]),
            (28u32).broadcast(shape![G, 32, W]),
        ];
        let first = 0i32.broadcast(shape![]);
        let second = 1i32.broadcast(shape![]);
        let mut acc0: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        let mut acc1: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        let mut acc2: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        let mut acc3: Tile<f32, { [R, G, 32, W] }> = constant(0.0, shape![R, G, 32, W]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [G, 32, W] }> = cp.load([index, part]).reshape(shape![G, 32, W]);
            let inputs: Tile<f32, { [R, G, 2, 32] }> =
                xp.load([pid.0, block]).reshape(shape![R, G, 2, 32]);
            let x_low: Tile<f32, { [R, G, 1, 32] }> = extract(inputs, [first, first, first, first]);
            let x_low: Tile<f32, { [R, G, 32, W] }> = x_low
                .reshape(shape![R, G, 32, 1])
                .broadcast(shape![R, G, 32, W]);
            let x_high: Tile<f32, { [R, G, 1, 32] }> =
                extract(inputs, [first, first, second, first]);
            let x_high: Tile<f32, { [R, G, 32, W] }> = x_high
                .reshape(shape![R, G, 32, 1])
                .broadcast(shape![R, G, 32, W]);
            let factor: Tile<f32, { [G, 2, W, 4] }> =
                fp.load([index, part]).reshape(shape![G, 2, W, 4]);
            // Byte 0 of each word: outputs `4w + 0`.
            let low: Tile<f32, { [G, 32, W] }> = bitcast(ori(andi(words, nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[0]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 0i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc0 = acc0 + low * x_low + high * x_high;
            // Byte 1 of each word: outputs `4w + 1`.
            let low: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[1]), nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[2]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 1i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc1 = acc1 + low * x_low + high * x_high;
            // Byte 2 of each word: outputs `4w + 2`.
            let low: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[3]), nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[4]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 2i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc2 = acc2 + low * x_low + high * x_high;
            // Byte 3 of each word: outputs `4w + 3`.
            let low: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[5]), nibble), magic));
            let high: Tile<f32, { [G, 32, W] }> =
                bitcast(ori(andi(shri(words, shifts[6]), nibble), magic));
            let at_byte: Tile<f32, { [G, 2, W, 1] }> =
                extract(factor, [first, first, first, 3i32.broadcast(shape![])]);
            let at_byte: Tile<f32, { [G, 2, W] }> = at_byte.reshape(shape![G, 2, W]);
            let low_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, first, first]);
            let high_factor: Tile<f32, { [G, 1, W] }> = extract(at_byte, [first, second, first]);
            // `(code - offset) * factor` once, multiplied by every row in the single-row order.
            let low: Tile<f32, { [G, 32, W] }> =
                (low - offset) * low_factor.broadcast(shape![G, 32, W]);
            let high: Tile<f32, { [G, 32, W] }> =
                (high - offset) * high_factor.broadcast(shape![G, 32, W]);
            let low: Tile<f32, { [R, G, 32, W] }> = low
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            let high: Tile<f32, { [R, G, 32, W] }> = high
                .reshape(shape![1, G, 32, W])
                .broadcast(shape![R, G, 32, W]);
            acc3 = acc3 + low * x_low + high * x_high;
        }
        let none: Tile<f32, { [R, W, 4] }> = constant(0.0, shape![R, W, 4]);
        store_rows::<R, G, W, OUT>(out, [acc0, acc1, acc2, acc3], none);
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
