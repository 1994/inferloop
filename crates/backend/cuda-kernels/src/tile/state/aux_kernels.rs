#[cutile::module]
pub mod aux {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tile_0, Tile_1, Tile_2, convert_scalar, convert_tile, cos, eq_tile,
        exp, ftz, get_tile_block_id, reduce_sum, rsqrt, select, sin, tile_to_scalar,
    };

    #[cutile::entry()]
    fn zero(out: &mut Tensor<f32, { [256] }>) {
        out.store(0.0f32.broadcast(shape![256]));
    }

    #[cutile::entry()]
    fn unary<const OP: i32>(out: &mut Tensor<f32, { [256] }>, input: &Tensor<f32, { [-1] }>) {
        let x = input.load_like(out);
        let negative: Tile<f32, { [256] }> = 0.0f32.broadcast(shape![256]) - x;
        let denominator = 1.0f32.broadcast(shape![256]) + exp(negative);
        if OP == 0 {
            out.store(x / denominator);
        } else {
            out.store(1.0f32.broadcast(shape![256]) / denominator);
        }
    }

    #[cutile::entry()]
    fn binary<const OP: i32>(
        out: &mut Tensor<f32, { [256] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        if OP == 0 {
            out.store(a.load_like(out) + b.load_like(out));
        } else {
            out.store(a.load_like(out) * b.load_like(out));
        }
    }

    #[cutile::entry()]
    fn norm<const D: i32, const B: i32, const GATED: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        x: &Tensor<f32, { [-1, D] }>,
        weight: &Tensor<f32, { [-1] }>,
        gate: &Tensor<f32, { [-1, D] }>,
        epsilon: f32,
        offset: f32,
    ) {
        let x = x.load_like(out);
        let square: Tile<f32, { [1] }> = reduce_sum(x * x, 1i32);
        let d: f32 = convert_scalar(D);
        let variance: Tile<f32, { [1] }> =
            square / d.broadcast(shape![1]) + epsilon.broadcast(shape![1]);
        let inverse = rsqrt(variance, ftz::Disabled)
            .reshape(shape![1, 1])
            .broadcast(shape![1, B]);
        let weights = weight
            .partition(shape![B])
            .load([0i32])
            .reshape(shape![1, B]);
        let mut result = x * inverse * (weights + offset.broadcast(shape![1, B]));
        if GATED == 1 {
            let z = gate.load_like(out);
            let negative: Tile<f32, { [1, B] }> = 0.0f32.broadcast(shape![1, B]) - z;
            let denominator = 1.0f32.broadcast(shape![1, B]) + exp(negative);
            result = result * z / denominator;
        }
        out.store(result);
    }

    /// [`norm`] over 16 rows per block for a row that fills one power-of-two tile: one row per
    /// block left the per-head gated norm (48 heads × 2048 prompt rows) bound by block overhead.
    #[cutile::entry()]
    fn norm_rows<const D: i32, const GATED: i32>(
        out: &mut Tensor<f32, { [16, D] }>,
        x: &Tensor<f32, { [-1, D] }>,
        weight: &Tensor<f32, { [-1] }>,
        gate: &Tensor<f32, { [-1, D] }>,
        epsilon: f32,
        offset: f32,
    ) {
        let x = x.load_like(out);
        let square: Tile<f32, { [16] }> = reduce_sum(x * x, 1i32);
        let d: f32 = convert_scalar(D);
        let variance: Tile<f32, { [16] }> =
            square / d.broadcast(shape![16]) + epsilon.broadcast(shape![16]);
        let inverse = rsqrt(variance, ftz::Disabled)
            .reshape(shape![16, 1])
            .broadcast(shape![16, D]);
        let weights = weight
            .partition(shape![D])
            .load([0i32])
            .reshape(shape![1, D])
            .broadcast(shape![16, D]);
        let mut result = x * inverse * (weights + offset.broadcast(shape![16, D]));
        if GATED == 1 {
            let z = gate.load_like(out);
            let negative: Tile<f32, { [16, D] }> = 0.0f32.broadcast(shape![16, D]) - z;
            let denominator = 1.0f32.broadcast(shape![16, D]) + exp(negative);
            result = result * z / denominator;
        }
        out.store(result);
    }

    #[cutile::entry()]
    fn split<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        x: &Tensor<f32, { [-1, D] }>,
        part: i32,
    ) {
        let pid = get_tile_block_id();
        out.store(x.partition(shape![1, B]).load([pid.0, part]));
    }

    #[cutile::entry()]
    fn rope<const D: i32, const HALF: i32>(
        out: &mut Tensor<f32, { [1, HALF] }>,
        x: &Tensor<f32, { [-1, D] }>,
        frequencies: &Tensor<f32, { [HALF] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, HALF]);
        let value = xp.load([pid.0, pid.1]);
        if pid.1 < 2 {
            let p: Tile<f32, { [1] }> = convert_tile(metadata.partition(shape![1]).load([0i32]));
            let position: f32 = tile_to_scalar(p.reshape(shape![]));
            let angles =
                frequencies.partition(shape![HALF]).load([0i32]) * position.broadcast(shape![HALF]);
            let c = cos(angles).reshape(shape![1, HALF]);
            let s = sin(angles).reshape(shape![1, HALF]);
            let other = xp.load([pid.0, 1i32 - pid.1]);
            if pid.1 == 0 {
                out.store(value * c - other * s);
            } else {
                out.store(value * c + other * s);
            }
        } else {
            out.store(value);
        }
    }
    #[cutile::entry()]
    fn prefill_rope<const D: i32, const HALF: i32, const HEADS: i32>(
        out: &mut Tensor<f32, { [1, HALF] }>,
        x: &Tensor<f32, { [-1, D] }>,
        frequencies: &Tensor<f32, { [HALF] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, HALF]);
        let value = xp.load([pid.0, pid.1]);
        if pid.1 < 2 {
            let p: Tile<f32, { [1] }> = convert_tile(metadata.partition(shape![1]).load([0i32]));
            let lane: f32 = convert_scalar(pid.0 / HEADS);
            let base: f32 = tile_to_scalar(p.reshape(shape![]));
            let position = base + lane;
            let angles =
                frequencies.partition(shape![HALF]).load([0i32]) * position.broadcast(shape![HALF]);
            let c = cos(angles).reshape(shape![1, HALF]);
            let s = sin(angles).reshape(shape![1, HALF]);
            let other = xp.load([pid.0, 1i32 - pid.1]);
            if pid.1 == 0 {
                out.store(value * c - other * s);
            } else {
                out.store(value * c + other * s);
            }
        } else {
            out.store(value);
        }
    }
    /// Three-axis `RoPE` for a prompt chunk: one `(t, h, w)` triple per lane.
    #[cutile::entry()]
    fn prefill_mrope<const D: i32, const HALF: i32, const HEADS: i32>(
        out: &mut Tensor<f32, { [1, HALF] }>,
        x: &Tensor<f32, { [-1, D] }>,
        frequencies: &Tensor<f32, { [HALF] }>,
        axes: &Tensor<i32, { [HALF] }>,
        rotary: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, HALF]);
        let value = xp.load([pid.0, pid.1]);
        if pid.1 < 2 {
            let lane: i32 = pid.0 / HEADS;
            let base = lane * 3;
            let positions = rotary.partition(shape![1]);
            let temporal: Tile<f32, { [1] }> = convert_tile(positions.load([base]));
            let height: Tile<f32, { [1] }> = convert_tile(positions.load([base + 1i32]));
            let width: Tile<f32, { [1] }> = convert_tile(positions.load([base + 2i32]));
            let selected_axis: Tile<i32, { [HALF] }> = axes.partition(shape![HALF]).load([0i32]);
            let position = select(
                eq_tile(selected_axis, 1i32.broadcast(shape![HALF])),
                height.broadcast(shape![HALF]),
                temporal.broadcast(shape![HALF]),
            );
            let position = select(
                eq_tile(selected_axis, 2i32.broadcast(shape![HALF])),
                width.broadcast(shape![HALF]),
                position,
            );
            let angles = frequencies.partition(shape![HALF]).load([0i32]) * position;
            let c = cos(angles).reshape(shape![1, HALF]);
            let s = sin(angles).reshape(shape![1, HALF]);
            let other = xp.load([pid.0, 1i32 - pid.1]);
            if pid.1 == 0 {
                out.store(value * c - other * s);
            } else {
                out.store(value * c + other * s);
            }
        } else {
            out.store(value);
        }
    }
    /// Three-axis `RoPE` with a provider-built frequency-to-axis map.
    #[cutile::entry()]
    fn mrope<const D: i32, const HALF: i32>(
        out: &mut Tensor<f32, { [1, HALF] }>,
        x: &Tensor<f32, { [-1, D] }>,
        frequencies: &Tensor<f32, { [HALF] }>,
        axes: &Tensor<i32, { [HALF] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, HALF]);
        let value = xp.load([pid.0, pid.1]);
        if pid.1 < 2 {
            let positions = metadata.partition(shape![1]);
            let temporal: Tile<f32, { [1] }> = convert_tile(positions.load([4i32]));
            let height: Tile<f32, { [1] }> = convert_tile(positions.load([5i32]));
            let width: Tile<f32, { [1] }> = convert_tile(positions.load([6i32]));
            let selected_axis: Tile<i32, { [HALF] }> = axes.partition(shape![HALF]).load([0i32]);
            let position = select(
                eq_tile(selected_axis, 1i32.broadcast(shape![HALF])),
                height.broadcast(shape![HALF]),
                temporal.broadcast(shape![HALF]),
            );
            let position = select(
                eq_tile(selected_axis, 2i32.broadcast(shape![HALF])),
                width.broadcast(shape![HALF]),
                position,
            );
            let angles = frequencies.partition(shape![HALF]).load([0i32]) * position;
            let c = cos(angles).reshape(shape![1, HALF]);
            let s = sin(angles).reshape(shape![1, HALF]);
            let other = xp.load([pid.0, 1i32 - pid.1]);
            if pid.1 == 0 {
                out.store(value * c - other * s);
            } else {
                out.store(value * c + other * s);
            }
        } else {
            out.store(value);
        }
    }

    /// [`fusion_norm`] for every lane of a prompt chunk: output row `2 * lane` normalizes the
    /// lane's embedding and row `2 * lane + 1` its external hidden, so the output viewed as
    /// `[lanes, 2D]` is the fused projection input.
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL requires statement control flow"
    )]
    #[cutile::entry()]
    fn fusion_norm_rows<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        embedding: &Tensor<f32, { [-1, D] }>,
        hidden: &Tensor<f32, { [-1, D] }>,
        norms: &Tensor<f32, { [2, D] }>,
        epsilon: f32,
        offset: f32,
    ) {
        let pid = get_tile_block_id();
        let lane = pid.0 / 2i32;
        let which = pid.0 % 2i32;
        let mut input = embedding
            .partition(shape![1, B])
            .load([lane, 0i32])
            .reshape(shape![B]);
        if which == 1i32 {
            input = hidden
                .partition(shape![1, B])
                .load([lane, 0i32])
                .reshape(shape![B]);
        }
        let square: Tile<f32, { [] }> = reduce_sum(input * input, 0i32);
        let dimension: f32 = convert_scalar(D);
        let inverse = rsqrt(
            square.reshape(shape![1]) / dimension.broadcast(shape![1])
                + epsilon.broadcast(shape![1]),
            ftz::Disabled,
        );
        let weight = norms
            .partition(shape![1, B])
            .load([which, 0i32])
            .reshape(shape![B]);
        out.store(
            (input * inverse.broadcast(shape![B]) * (weight + offset.broadcast(shape![B])))
                .reshape(shape![1, B]),
        );
    }

    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL requires statement control flow"
    )]
    #[cutile::entry()]
    fn fusion_norm<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        embedding: &Tensor<f32, { [-1] }>,
        hidden: &Tensor<f32, { [-1] }>,
        norms: &Tensor<f32, { [2, D] }>,
        epsilon: f32,
        offset: f32,
    ) {
        let pid = get_tile_block_id();
        let mut input = embedding.partition(shape![B]).load([0i32]);
        if pid.0 == 1 {
            input = hidden.partition(shape![B]).load([0i32]);
        }
        let square: Tile<f32, { [] }> = reduce_sum(input * input, 0i32);
        let dimension: f32 = convert_scalar(D);
        let inverse = rsqrt(
            square.reshape(shape![1]) / dimension.broadcast(shape![1])
                + epsilon.broadcast(shape![1]),
            ftz::Disabled,
        );
        let weight = norms
            .partition(shape![1, B])
            .load([pid.0, 0i32])
            .reshape(shape![B]);
        out.store(
            (input * inverse.broadcast(shape![B]) * (weight + offset.broadcast(shape![B])))
                .reshape(shape![1, B]),
        );
    }
}
