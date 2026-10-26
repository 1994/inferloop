//! cuTile kernels for the vision encoder.
//!
//! The projections reuse the 32×64 tile shape of the prompt GEMM: BF16 weights, F32 activation
//! tiles split into a BF16 high part plus residual, with F32 accumulation. Vision token counts are padded to a multiple of the 32-row tile, and
//! the padding rows are discarded after the readback.
#[cutile::module]
pub(crate) mod encoder {
    /// One half, the leading factor of both GELU forms.
    const HALF: f32 = 0.5;
    /// `gelu_pytorch_tanh` cubic coefficient.
    const GELU_CUBIC: f32 = 0.044_715;
    /// `gelu_pytorch_tanh` scale, `sqrt(2/pi)`.
    const GELU_SCALE: f32 = 0.797_884_6;
    /// `1/sqrt(2)`, the argument scaling of the exact `erf` GELU.
    const FRAC_1_SQRT_2: f32 = 0.707_106_8;
    /// `erf` rational approximation constants; Abramowitz & Stegun 7.1.26, `|eps| <= 1.5e-7`.
    const ERF_P: f32 = 0.327_591_1;
    const ERF_A1: f32 = 0.254_829_6;
    const ERF_A2: f32 = -0.284_496_7;
    const ERF_A3: f32 = 1.421_413_7;
    const ERF_A4: f32 = -1.453_152;
    const ERF_A5: f32 = 1.061_405_4;
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tile_0, Tile_1, Tile_2, absf, bf16, cmp_ordering, cmpf, constant, convert_scalar,
        convert_tile, exp, ftz, get_tile_block_id, iota, lt_tile, mmaf, predicate, reduce_sum,
        rsqrt, select, tanh,
    };

    /// Element-wise sum of two activation tensors of the same shape.
    #[cutile::entry()]
    fn add<const N: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        left: &Tensor<f32, { [-1, N] }>,
        right: &Tensor<f32, { [-1, N] }>,
    ) {
        let pid = get_tile_block_id();
        let a: Tile<f32, { [32, 64] }> = left.partition(shape![32, 64]).load([pid.0, pid.1]);
        let b: Tile<f32, { [32, 64] }> = right.partition(shape![32, 64]).load([pid.0, pid.1]);
        out.store(a + b);
    }

    /// `LayerNorm` with bias over one padded row: `D` real values inside a `P`-wide tile.
    ///
    /// Padding a row to a power of two is what makes a single-tile reduction legal; the mask keeps
    /// the zeros out of the variance, and the padded weight/bias entries are zero.
    #[cutile::entry()]
    fn layernorm<const D: i32, const P: i32>(
        out: &mut Tensor<f32, { [P] }>,
        x: &Tensor<f32, { [-1, P] }>,
        weight: &Tensor<bf16, { [-1] }>,
        bias: &Tensor<bf16, { [-1] }>,
        epsilon: f32,
    ) {
        let pid = get_tile_block_id();
        let tile: Tile<f32, { [P] }> = x
            .partition(shape![1, P])
            .load([pid.0, 0i32])
            .reshape(shape![P]);
        let indices: Tile<i32, { [P] }> = iota(shape![P]);
        let valid = lt_tile(indices, D.broadcast(shape![P]));
        let mask: Tile<f32, { [P] }> = select(
            valid,
            1.0f32.broadcast(shape![P]),
            0.0f32.broadcast(shape![P]),
        );
        let dim: f32 = convert_scalar(D);
        let total: Tile<f32, { [] }> = reduce_sum(tile, 0i32);
        let mean: Tile<f32, { [] }> = total / dim.broadcast(shape![]);
        let centered: Tile<f32, { [P] }> =
            (tile - mean.reshape(shape![1]).broadcast(shape![P])) * mask;
        let square: Tile<f32, { [] }> = reduce_sum(centered * centered, 0i32);
        let variance: Tile<f32, { [] }> = square / dim.broadcast(shape![]);
        let inverse = rsqrt(variance + epsilon.broadcast(shape![]), ftz::Disabled);
        let w: Tile<f32, { [P] }> = convert_tile(weight.partition(shape![P]).load([0i32]));
        let b: Tile<f32, { [P] }> = convert_tile(bias.partition(shape![P]).load([0i32]));
        out.store(centered * inverse.reshape(shape![1]).broadcast(shape![P]) * w + b);
    }

    /// Apply `RoPE` once per Q/K element, before the quadratic attention loop.
    #[cutile::entry()]
    fn rope<const HD: i32>(
        out: &mut Tensor<f32, { [32, HD] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        cos: &Tensor<f32, { [-1, -1] }>,
        sin: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let half = pid.1 % 2i32;
        let sibling = pid.1 - half + (1i32 - half);
        let values = input.partition(shape![32, HD]);
        let own: Tile<f32, { [32, HD] }> = values.load([pid.0, pid.1]);
        let other: Tile<f32, { [32, HD] }> = values.load([pid.0, sibling]);
        let cosine: Tile<f32, { [32, HD] }> = cos.partition(shape![32, HD]).load([pid.0, half]);
        let sine: Tile<f32, { [32, HD] }> = sin.partition(shape![32, HD]).load([pid.0, half]);
        let rotated = if half == 0i32 {
            own * cosine - other * sine
        } else {
            own * cosine + other * sine
        };
        out.store(rotated);
    }

    /// `RoPE` over a compact physical half-head with a power-of-two register tile.
    ///
    /// `HALF` is the physical (padded) half stride and need not be a power of two; each tile is a
    /// 16-lane segment wholly inside one half. Head, half, sibling and cosine lane are derived
    /// from the physical stride, and lanes beyond `REAL` are zeroed after the rotation.
    #[cutile::entry()]
    fn rope_indexed<const HALF: i32, const REAL: i32>(
        out: &mut Tensor<f32, { [32, 16] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        cos: &Tensor<f32, { [-1, -1] }>,
        sin: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let segments = HALF / 16i32;
        let head_segments = segments * 2i32;
        let head = pid.1 / head_segments;
        let segment = pid.1 - head * head_segments;
        let half = segment / segments;
        let lane_segment = segment - half * segments;
        let sibling = head * head_segments + (1i32 - half) * segments + lane_segment;
        let values = input.partition(shape![32, 16]);
        let own: Tile<f32, { [32, 16] }> = values.load([pid.0, pid.1]);
        let other: Tile<f32, { [32, 16] }> = values.load([pid.0, sibling]);
        let table = cos.partition(shape![32, 16]);
        let table_sin = sin.partition(shape![32, 16]);
        let table_column = half * segments + lane_segment;
        let cosine: Tile<f32, { [32, 16] }> = table.load([pid.0, table_column]);
        let sine: Tile<f32, { [32, 16] }> = table_sin.load([pid.0, table_column]);
        let rotated = if half == 0i32 {
            own * cosine - other * sine
        } else {
            own * cosine + other * sine
        };
        let lane: Tile<i32, { [16] }> = iota(shape![16]);
        let position = lane + (lane_segment * 16i32).broadcast(shape![16]);
        let valid = lt_tile(position, REAL.broadcast(shape![16]));
        let zero: Tile<f32, { [32, 16] }> = 0.0f32.broadcast(shape![32, 16]);
        out.store(select(
            valid.reshape(shape![1, 16]).broadcast(shape![32, 16]),
            rotated,
            zero,
        ));
    }

    /// Pack physical 96-wide compact heads into the qualified 128-wide attention layout.
    ///
    /// Each tile is a power-of-two 16-lane segment. Compact half0 segments 0..2 map to wide
    /// segments 0..2, and compact half1 segments 3..5 map to wide segments 4..6; wide segments 3
    /// and 7 (the extra padding half lanes) are zero.
    #[cutile::entry()]
    fn pack_wide(out: &mut Tensor<f32, { [32, 16] }>, input: &Tensor<f32, { [-1, -1] }>) {
        let pid = get_tile_block_id();
        let local = pid.1 % 8i32;
        let head = pid.1 / 8i32;
        let source_local = if local < 3i32 {
            local
        } else if local < 7i32 {
            local - 1i32
        } else {
            0i32
        };
        let source = input
            .partition(shape![32, 16])
            .load([pid.0, head * 6i32 + source_local]);
        let lane: Tile<i32, { [16] }> = iota(shape![16]);
        let real_limit = if local == 2i32 || local == 6i32 {
            4i32
        } else {
            16i32
        };
        let lane_valid = lt_tile(lane, real_limit.broadcast(shape![16]));
        let valid = (local != 3i32 && local != 7i32).broadcast(shape![32, 16])
            & lane_valid.reshape(shape![1, 16]).broadcast(shape![32, 16]);
        let zero: Tile<f32, { [32, 16] }> = 0.0f32.broadcast(shape![32, 16]);
        out.store(select(valid, source, zero));
    }

    /// Pack the qualified 128-wide attention result back into physical 96-wide compact heads.
    #[cutile::entry()]
    fn pack_compact(out: &mut Tensor<f32, { [32, 16] }>, input: &Tensor<f32, { [-1, -1] }>) {
        let pid = get_tile_block_id();
        let local = pid.1 % 6i32;
        let head = pid.1 / 6i32;
        let source_local = if local < 3i32 { local } else { local + 1i32 };
        let source = input
            .partition(shape![32, 16])
            .load([pid.0, head * 8i32 + source_local]);
        let lane: Tile<i32, { [16] }> = iota(shape![16]);
        let real_limit = if local == 2i32 || local == 5i32 {
            4i32
        } else {
            16i32
        };
        let lane_valid = lt_tile(lane, real_limit.broadcast(shape![16]));
        let zero: Tile<f32, { [32, 16] }> = 0.0f32.broadcast(shape![32, 16]);
        out.store(select(
            lane_valid.reshape(shape![1, 16]).broadcast(shape![32, 16]),
            source,
            zero,
        ));
    }

    /// `gelu_pytorch_tanh`, the activation of every vision block.
    #[cutile::entry()]
    fn gelu_tanh<const N: i32>(out: &mut Tensor<f32, { [32, 64] }>, x: &Tensor<f32, { [-1, N] }>) {
        let pid = get_tile_block_id();
        let value: Tile<f32, { [32, 64] }> = x.partition(shape![32, 64]).load([pid.0, pid.1]);
        let cubic = value * value * value * GELU_CUBIC.broadcast(shape![32, 64]);
        let inner = (value + cubic) * GELU_SCALE.broadcast(shape![32, 64]);
        out.store(
            HALF.broadcast(shape![32, 64])
                * value
                * (1.0f32.broadcast(shape![32, 64]) + tanh(inner)),
        );
    }

    /// Exact `erf` GELU, the activation of the patch merger.
    #[cutile::entry()]
    fn gelu_erf<const N: i32>(out: &mut Tensor<f32, { [32, 64] }>, x: &Tensor<f32, { [-1, N] }>) {
        let pid = get_tile_block_id();
        let value: Tile<f32, { [32, 64] }> = x.partition(shape![32, 64]).load([pid.0, pid.1]);
        // Exact GELU is `0.5x(1 + erf(x/sqrt(2)))`: the argument is scaled before the erf.
        let argument = value * FRAC_1_SQRT_2.broadcast(shape![32, 64]);
        let magnitude = absf(argument);
        let ones: Tile<f32, { [32, 64] }> = 1.0f32.broadcast(shape![32, 64]);
        let t = ones / (ones + ERF_P.broadcast(shape![32, 64]) * magnitude);
        let polynomial = t
            * (ERF_A1.broadcast(shape![32, 64])
                + t * (ERF_A2.broadcast(shape![32, 64])
                    + t * (ERF_A3.broadcast(shape![32, 64])
                        + t * (ERF_A4.broadcast(shape![32, 64])
                            + t * ERF_A5.broadcast(shape![32, 64])))));
        let zeros: Tile<f32, { [32, 64] }> = 0.0f32.broadcast(shape![32, 64]);
        let tail = ones - polynomial * exp(zeros - magnitude * magnitude);
        let negative = cmpf(argument, zeros, predicate::LessThan, cmp_ordering::Ordered);
        let erf = select(negative, zeros - tail, tail);
        out.store(HALF.broadcast(shape![32, 64]) * value * (ones + erf));
    }

    /// `out[m, n] = sum_k input[m, k] * weight[n, k] + bias[n]`, the patch-embed projection.
    #[cutile::entry()]
    fn dense_bias<const K: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<bf16, { [-1, K] }>,
        bias: &Tensor<bf16, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![32, 64]);
        let wp = weight.partition(shape![64, 64]);
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for k in 0i32..((K + 63) / 64) {
            let input: Tile<f32, { [32, 64] }> = xp.load([pid.0, k]);
            let x: Tile<bf16, { [32, 64] }> = convert_tile(input);
            let rounded: Tile<f32, { [32, 64] }> = convert_tile(x);
            let residual: Tile<bf16, { [32, 64] }> = convert_tile(input - rounded);
            let weight = wp.load([pid.1, k]).transpose();
            acc = mmaf(residual, weight, acc);
            acc = mmaf(x, weight, acc);
        }
        let b: Tile<f32, { [64] }> = convert_tile(bias.partition(shape![64]).load([pid.1]));
        out.store(acc + b.reshape(shape![1, 64]).broadcast(shape![32, 64]));
    }
}
