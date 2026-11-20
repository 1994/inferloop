//! Nibble-packed grouped-int GEMV over the 16-row tile-major layout.
//!
//! Lane `l` of each 64-value group holds value `l` in its low nibble and value `l + 32` in its
//! high nibble, sixteen rows per byte run. The GEMV reads the codes as `u32` words: word `w` of a
//! lane carries rows `4w..4w + 4`, one per byte. A CTA covers `W` of the four words, so a tile
//! can spread over several CTAs without a cross-CTA reduction. Each step covers `G` 64-value
//! groups and decodes all four rows of a word along the last tile axis, so one `[VALUES, W, 4]`
//! accumulator per activation row holds every output. A nibble becomes a float exactly by OR-ing it into the mantissa of `2^23` and
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
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, Shape_4, Shape_5,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tile_1, Tile_2, Tile_3, Tile_4, Tile_5, andi,
        bf16, bitcast, cat, constant, convert_scalar, convert_tile, get_tile_block_id, iota, mmaf,
        ori, reduce_sum, shri,
    };

    /// Bits of the float `2^23`, whose mantissa holds a small integer exactly.
    const MAGIC_BITS: u32 = 0x4b00_0000;
    /// `2^23`, subtracted to recover the integer from [`MAGIC_BITS`].
    const MAGIC: f32 = 8_388_608.0;

    /// `OUT = 4 * W` outputs of a sixteen-row tile for `R` activation rows, `factor * code - bias`:
    /// CTA `(r, p)` covers row block `r` and part `p % PARTS` of tile `p / PARTS`. Each of the
    /// `BLOCKS` steps reads `W` words of `LANES = 32G` lanes, `VALUES = 64G` activations per row and
    /// `HALVES = 2G` half-group term rows; every bias multiplies its half-group's activation sum.
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
        let offset = MAGIC.broadcast(shape![VALUES, W, 4]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W, 4]);
        let nibble = (0x0fu32).broadcast(shape![G, 2, 32, W, 4]);
        // Nibble `h` of output `4w + b` sits at bit `8b + 4h` of word `w`.
        let byte: Tile<i32, { [4] }> = iota(shape![4]) * 8i32.broadcast(shape![4]);
        let half: Tile<i32, { [2] }> = iota(shape![2]) * 4i32.broadcast(shape![2]);
        let nibble_shift: Tile<u32, { [G, 2, 32, W, 4] }> = bitcast(
            byte.reshape(shape![1, 1, 1, 1, 4])
                .broadcast(shape![G, 2, 32, W, 4])
                + half
                    .reshape(shape![1, 2, 1, 1, 1])
                    .broadcast(shape![G, 2, 32, W, 4]),
        );
        let mut offsets: Tile<f32, { [R, HALVES, W, 4] }> = constant(0.0, shape![R, HALVES, W, 4]);
        let mut acc: Tile<f32, { [R, VALUES, W, 4] }> = constant(0.0, shape![R, VALUES, W, 4]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [G, 2, 32, W, 4] }> = cp
                .load([index, part])
                .reshape(shape![G, 1, 32, W, 1])
                .broadcast(shape![G, 2, 32, W, 4]);
            let low: Tile<u32, { [VALUES, W, 4] }> =
                andi(shri(words, nibble_shift), nibble).reshape(shape![VALUES, W, 4]);
            let code: Tile<f32, { [VALUES, W, 4] }> = bitcast(ori(low, magic));
            let factor: Tile<f32, { [VALUES, W, 4] }> = fp
                .load([index, part])
                .reshape(shape![HALVES, 1, W, 4])
                .broadcast(shape![HALVES, 32, W, 4])
                .reshape(shape![VALUES, W, 4]);
            // The weight once, multiplied by every row in the single-row order.
            let weight: Tile<f32, { [VALUES, W, 4] }> = (code - offset) * factor;
            let weight: Tile<f32, { [R, VALUES, W, 4] }> = weight
                .reshape(shape![1, VALUES, W, 4])
                .broadcast(shape![R, VALUES, W, 4]);
            let inputs: Tile<f32, { [R, VALUES] }> = xp.load([pid.0, block]);
            let activation: Tile<f32, { [R, VALUES, W, 4] }> = inputs
                .reshape(shape![R, VALUES, 1, 1])
                .broadcast(shape![R, VALUES, W, 4]);
            acc = acc + weight * activation;
            let bias: Tile<f32, { [R, HALVES, W, 4] }> = bp
                .load([index, part])
                .reshape(shape![1, HALVES, W, 4])
                .broadcast(shape![R, HALVES, W, 4]);
            let totals: Tile<f32, { [R, HALVES] }> =
                reduce_sum(inputs.reshape(shape![R, HALVES, 32]), 2i32);
            offsets = offsets
                + bias
                    * totals
                        .reshape(shape![R, HALVES, 1, 1])
                        .broadcast(shape![R, HALVES, W, 4]);
        }
        let sums: Tile<f32, { [R, W, 4] }> = reduce_sum(acc, 1i32);
        let by_row: Tile<f32, { [R, W, 4] }> = reduce_sum(offsets, 1i32);
        out.store((sums - by_row).reshape(shape![R, OUT]));
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
        let offset = shift.broadcast(shape![VALUES, W, 4]);
        let magic = MAGIC_BITS.broadcast(shape![VALUES, W, 4]);
        let nibble = (0x0fu32).broadcast(shape![G, 2, 32, W, 4]);
        // Nibble `h` of output `4w + b` sits at bit `8b + 4h` of word `w`.
        let byte: Tile<i32, { [4] }> = iota(shape![4]) * 8i32.broadcast(shape![4]);
        let half: Tile<i32, { [2] }> = iota(shape![2]) * 4i32.broadcast(shape![2]);
        let nibble_shift: Tile<u32, { [G, 2, 32, W, 4] }> = bitcast(
            byte.reshape(shape![1, 1, 1, 1, 4])
                .broadcast(shape![G, 2, 32, W, 4])
                + half
                    .reshape(shape![1, 2, 1, 1, 1])
                    .broadcast(shape![G, 2, 32, W, 4]),
        );
        let mut acc: Tile<f32, { [R, VALUES, W, 4] }> = constant(0.0, shape![R, VALUES, W, 4]);
        for block in 0i32..BLOCKS {
            let index = tile * BLOCKS + block;
            let words: Tile<u32, { [G, 2, 32, W, 4] }> = cp
                .load([index, part])
                .reshape(shape![G, 1, 32, W, 1])
                .broadcast(shape![G, 2, 32, W, 4]);
            let low: Tile<u32, { [VALUES, W, 4] }> =
                andi(shri(words, nibble_shift), nibble).reshape(shape![VALUES, W, 4]);
            let code: Tile<f32, { [VALUES, W, 4] }> = bitcast(ori(low, magic));
            let factor: Tile<f32, { [VALUES, W, 4] }> = fp
                .load([index, part])
                .reshape(shape![HALVES, 1, W, 4])
                .broadcast(shape![HALVES, 32, W, 4])
                .reshape(shape![VALUES, W, 4]);
            // The weight once, multiplied by every row in the single-row order.
            let weight: Tile<f32, { [VALUES, W, 4] }> = (code - offset) * factor;
            let weight: Tile<f32, { [R, VALUES, W, 4] }> = weight
                .reshape(shape![1, VALUES, W, 4])
                .broadcast(shape![R, VALUES, W, 4]);
            let inputs: Tile<f32, { [R, VALUES] }> = xp.load([pid.0, block]);
            let activation: Tile<f32, { [R, VALUES, W, 4] }> = inputs
                .reshape(shape![R, VALUES, 1, 1])
                .broadcast(shape![R, VALUES, W, 4]);
            acc = acc + weight * activation;
        }
        let sums: Tile<f32, { [R, W, 4] }> = reduce_sum(acc, 1i32);
        out.store(sums.reshape(shape![R, OUT]));
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
