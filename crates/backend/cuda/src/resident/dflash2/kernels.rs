//! `DFlash2` draft kernels: everything the draft does that the target graph does not.
#[cutile::module]
pub(crate) mod draft {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3, bf16, cmp_ordering, cmpf,
        constant, convert_scalar, convert_tile, cos, eq_tile, exp, f8e4m3fn, ftz, ge_tile,
        get_tile_block_id, iota, le_tile, lt_tile, mmaf, predicate, reduce_max, reduce_min,
        reduce_sum, rsqrt, select, sin, tile_to_scalar,
    };

    /// Smallest finite score; masked entries never win a reduction against a real one.
    const MASKED: f32 = -3.0e38;
    /// Index larger than any vocabulary entry, so it never wins a minimum.
    const UNUSED: i32 = 2_147_483_647;
    /// Keys one attention step reads from the ring.
    const KEYS: i32 = 64;

    /// Gather one embedding row per output row; `tokens` holds one id per row.
    #[cutile::entry()]
    fn embed<const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        table: &Tensor<bf16, { [-1, -1] }>,
        tokens: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let token: i32 =
            tile_to_scalar(tokens.partition(shape![1]).load([pid.0]).reshape(shape![]));
        let row: Tile<f32, { [1, B] }> =
            convert_tile(table.partition(shape![1, B]).load([token, pid.1]));
        out.store(row);
    }

    /// `RMSNorm` of one `D`-wide row, loaded as one power-of-two `B`-wide tile whose padding
    /// reads as zero.
    #[cutile::entry()]
    fn norm<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        epsilon: f32,
    ) {
        let pid = get_tile_block_id();
        let x = x.partition(shape![1, B]).load([pid.0, 0i32]);
        let square: Tile<f32, { [1] }> = reduce_sum(x * x, 1i32);
        let d: f32 = convert_scalar(D);
        let inverse = rsqrt(
            square / d.broadcast(shape![1]) + epsilon.broadcast(shape![1]),
            ftz::Disabled,
        )
        .reshape(shape![1, 1])
        .broadcast(shape![1, B]);
        let weight = weight
            .partition(shape![B])
            .load([0i32])
            .reshape(shape![1, B]);
        out.store(x * inverse * weight);
    }

    /// Residual update and the next layer's `RMSNorm`: `stream = residual + delta`,
    /// `out = norm(stream)`, over a `B`-wide padded tile of a `D`-wide row.
    #[cutile::entry()]
    fn add_norm<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        stream: &mut Tensor<f32, { [1, B] }>,
        residual: &Tensor<f32, { [-1, -1] }>,
        delta: &Tensor<f32, { [-1, -1] }>,
        weight: &Tensor<f32, { [-1] }>,
        epsilon: f32,
    ) {
        let pid = get_tile_block_id();
        let x = residual.partition(shape![1, B]).load([pid.0, 0i32])
            + delta.partition(shape![1, B]).load([pid.0, 0i32]);
        let square: Tile<f32, { [1] }> = reduce_sum(x * x, 1i32);
        let d: f32 = convert_scalar(D);
        let inverse = rsqrt(
            square / d.broadcast(shape![1]) + epsilon.broadcast(shape![1]),
            ftz::Disabled,
        )
        .reshape(shape![1, 1])
        .broadcast(shape![1, B]);
        let weight = weight
            .partition(shape![B])
            .load([0i32])
            .reshape(shape![1, B]);
        stream.store(x);
        out.store(x * inverse * weight);
    }

    /// Two-tap grouped convolution inside each `BLOCK`-row block (the reference's
    /// `_grouped_conv`): `out[r] = (base[s,0] + c[r,s,0]) x[r] + (base[s,1] + c[r,s,1]) x[r-1]`,
    /// where the previous row only counts inside the same block and a coefficient covers
    /// `GROUP` consecutive channels. `coefficients` is `[rows, side, tap, groups]` flattened and
    /// `base` is `[side * tap, channels]`.
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL requires statement control flow"
    )]
    #[cutile::entry()]
    fn grouped_conv<
        const B: i32,
        const GROUP: i32,
        const GB: i32,
        const GROUPS: i32,
        const BLOCK: i32,
    >(
        out: &mut Tensor<f32, { [1, B] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        coefficients: &Tensor<f32, { [-1, -1] }>,
        base: &Tensor<f32, { [-1, -1] }>,
        side: i32,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, B]);
        let current = xp.load([pid.0, pid.1]);
        let mut previous_row = pid.0 - 1i32;
        let mut gate = 1.0f32;
        if pid.0 % BLOCK == 0i32 {
            previous_row = pid.0;
            gate = 0.0f32;
        }
        let previous = xp.load([previous_row, pid.1]);
        let cp = coefficients.partition(shape![1, GB]);
        let first = cp
            .load([pid.0, side * 2i32 * GROUPS / GB + pid.1])
            .reshape(shape![GB, 1])
            .broadcast(shape![GB, GROUP])
            .reshape(shape![1, B]);
        let second = cp
            .load([pid.0, (side * 2i32 + 1i32) * GROUPS / GB + pid.1])
            .reshape(shape![GB, 1])
            .broadcast(shape![GB, GROUP])
            .reshape(shape![1, B]);
        let bp = base.partition(shape![1, B]);
        let b0 = bp.load([side * 2i32, pid.1]);
        let b1 = bp.load([side * 2i32 + 1i32, pid.1]);
        out.store((b0 + first) * current + (b1 + second) * previous * gate.broadcast(shape![1, B]));
    }

    /// Neox `RoPE` over `[rows * HEADS, 2 * HALF]`, one position per row.
    #[cutile::entry()]
    fn rope_rows<const HALF: i32, const HEADS: i32>(
        out: &mut Tensor<f32, { [1, HALF] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        frequencies: &Tensor<f32, { [HALF] }>,
        positions: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, HALF]);
        let value = xp.load([pid.0, pid.1]);
        let other = xp.load([pid.0, 1i32 - pid.1]);
        let p: Tile<f32, { [1] }> =
            convert_tile(positions.partition(shape![1]).load([pid.0 / HEADS]));
        let position: f32 = tile_to_scalar(p.reshape(shape![]));
        let angles =
            frequencies.partition(shape![HALF]).load([0i32]) * position.broadcast(shape![HALF]);
        let c = cos(angles).reshape(shape![1, HALF]);
        let s = sin(angles).reshape(shape![1, HALF]);
        if pid.1 == 0i32 {
            out.store(value * c - other * s);
        } else {
            out.store(value * c + other * s);
        }
    }

    /// `SiLU(gate) * up`, elementwise over flat tiles.
    #[cutile::entry()]
    fn silu_mul(
        out: &mut Tensor<f32, { [256] }>,
        gate: &Tensor<f32, { [-1] }>,
        up: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let x = gate.partition(shape![256]).load([pid.0]);
        let negative: Tile<f32, { [256] }> = 0.0f32.broadcast(shape![256]) - x;
        let denominator = 1.0f32.broadcast(shape![256]) + exp(negative);
        out.store(x / denominator * up.partition(shape![256]).load([pid.0]));
    }

    /// Store `ROWS` rows of K or V into one head of the per-slot ring: row `r` lands at
    /// `slots[r] * RING + positions[r] % RING`; a negative position skips the row.
    #[cutile::entry()]
    fn ring_store<const D: i32, const CAP: i32, const RING: i32, const ROWS: i32>(
        ring: &mut Tensor<bf16, { [1, CAP, D] }>,
        rows: &Tensor<f32, { [-1, -1, D] }>,
        slots: &Tensor<i32, { [-1] }>,
        positions: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let rp = rows.partition(shape![1, 1, D]);
        let sp = slots.partition(shape![1]);
        let pp = positions.partition(shape![1]);
        for row in 0i32..ROWS {
            let position: i32 = tile_to_scalar(pp.load([row]).reshape(shape![]));
            if position >= 0i32 {
                let slot: i32 = tile_to_scalar(sp.load([row]).reshape(shape![]));
                let value: Tile<bf16, { [1, 1, D] }> = convert_tile(rp.load([row, pid.0, 0i32]));
                ring.partition_mut(shape![1, 1, D])
                    .store(value, [0i32, slot * RING + position % RING, 0i32]);
            }
        }
    }

    /// Non-causal block attention over the ring for one (slot, KV head): the block's `BLOCK`
    /// queries see every key of their own block and the history inside the sliding `WINDOW`
    /// (`query - key < WINDOW`). The block's keys are already in the ring; `bases[slot]` is the
    /// block's first position, negative for an idle slot, and `rings[slot]` the ring it reads.
    /// `q` and `out` are `[rows, kv_heads, GROUP * D]`, so one tile holds the block's queries of
    /// one KV head in `(token, head)` order.
    #[cutile::entry()]
    fn block_attention<
        const D: i32,
        const GROUP: i32,
        const GD: i32,
        const BLOCK: i32,
        const QR: i32,
        const RING: i32,
        const WINDOW: i32,
    >(
        out: &mut Tensor<f32, { [BLOCK, 1, GD] }>,
        q: &Tensor<f32, { [-1, -1, GD] }>,
        keys: &Tensor<bf16, { [-1, -1, D] }>,
        values: &Tensor<bf16, { [-1, -1, D] }>,
        bases: &Tensor<i32, { [-1] }>,
        rings: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let base: i32 = tile_to_scalar(bases.partition(shape![1]).load([pid.0]).reshape(shape![]));
        if base >= 0i32 {
            let ring: i32 =
                tile_to_scalar(rings.partition(shape![1]).load([pid.0]).reshape(shape![]));
            let d: f32 = convert_scalar(D);
            let scale: Tile<f32, { [QR, D] }> = rsqrt(d.broadcast(shape![QR, D]), ftz::Disabled);
            let query: Tile<f32, { [QR, D] }> = q
                .partition(shape![BLOCK, 1, GD])
                .load([pid.0, pid.1, 0i32])
                .reshape(shape![QR, D]);
            let query: Tile<bf16, { [QR, D] }> = convert_tile(query * scale);
            let offsets: Tile<i32, { [BLOCK] }> = iota(shape![BLOCK]);
            let query_positions = (offsets + base.broadcast(shape![BLOCK]))
                .reshape(shape![BLOCK, 1])
                .broadcast(shape![BLOCK, GROUP])
                .reshape(shape![QR, 1])
                .broadcast(shape![QR, KEYS]);
            let earliest = query_positions - (WINDOW - 1i32).broadcast(shape![QR, KEYS]);
            let last = base + BLOCK - 1i32;
            let mut first = last + 1i32 - WINDOW;
            if first < 0i32 {
                first = 0i32;
            }
            let tiles = RING / KEYS;
            let kp = keys.partition(shape![1, KEYS, D]);
            let vp = values.partition(shape![1, KEYS, D]);
            let mut accumulator: Tile<f32, { [QR, D] }> = constant(0.0f32, shape![QR, D]);
            let mut maximum: Tile<f32, { [QR] }> = constant(MASKED, shape![QR]);
            let mut total: Tile<f32, { [QR] }> = constant(0.0f32, shape![QR]);
            let zero_scores: Tile<f32, { [QR, KEYS] }> = constant(0.0f32, shape![QR, KEYS]);
            let zero_values: Tile<f32, { [QR, D] }> = constant(0.0f32, shape![QR, D]);
            let masked: Tile<f32, { [QR, KEYS] }> = constant(MASKED, shape![QR, KEYS]);
            // The DSL lowers only half-open ranges; `..=` would silently drop the last tile.
            let end = last / KEYS + 1i32;
            for tile in (first / KEYS)..end {
                let physical = ring * tiles + tile % tiles;
                let key: Tile<bf16, { [KEYS, D] }> =
                    kp.load([pid.1, physical, 0i32]).reshape(shape![KEYS, D]);
                let value: Tile<bf16, { [KEYS, D] }> =
                    vp.load([pid.1, physical, 0i32]).reshape(shape![KEYS, D]);
                let scores = mmaf(query, key.transpose(), zero_scores);
                let key_offsets: Tile<i32, { [KEYS] }> = iota(shape![KEYS]);
                let key_positions = (key_offsets + (tile * KEYS).broadcast(shape![KEYS]))
                    .reshape(shape![1, KEYS])
                    .broadcast(shape![QR, KEYS]);
                let valid = le_tile(key_positions, last.broadcast(shape![QR, KEYS]))
                    & ge_tile(key_positions, earliest)
                    & ge_tile(key_positions, 0i32.broadcast(shape![QR, KEYS]));
                let scores = select(valid, scores, masked);
                let tile_maximum: Tile<f32, { [QR] }> = reduce_max(scores, 1i32);
                // Materialize both shapes for cuTile's type checker before combining them.
                let tile_maximum = tile_maximum.reshape(shape![QR]);
                let previous = maximum.reshape(shape![QR]);
                let greater = cmpf(
                    tile_maximum,
                    previous,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let next: Tile<f32, { [QR] }> = select(greater, tile_maximum, previous);
                let rescale = exp(previous - next);
                let probabilities = select(
                    valid,
                    exp(scores - next.reshape(shape![QR, 1]).broadcast(shape![QR, KEYS])),
                    zero_scores,
                );
                let tile_total: Tile<f32, { [QR] }> = reduce_sum(probabilities, 1i32);
                let tile_total = tile_total.reshape(shape![QR]);
                let rescale = rescale.reshape(shape![QR]);
                total = total.reshape(shape![QR]) * rescale + tile_total;
                let weights: Tile<bf16, { [QR, KEYS] }> = convert_tile(probabilities);
                let contribution: Tile<f32, { [QR, D] }> = mmaf(weights, value, zero_values);
                accumulator = accumulator.reshape(shape![QR, D])
                    * rescale.reshape(shape![QR, 1]).broadcast(shape![QR, D])
                    + contribution.reshape(shape![QR, D]);
                maximum = next.reshape(shape![QR]);
            }
            let result = accumulator.reshape(shape![QR, D])
                / total.reshape(shape![QR, 1]).broadcast(shape![QR, D]);
            out.store(result.reshape(shape![BLOCK, 1, GD]));
        } else {
            out.store(0.0f32.broadcast(shape![BLOCK, 1, GD]));
        }
    }

    /// Split-K window of a per-channel FP8 GEMM: block row `pid.0` of `partials`
    /// (`[splits * 32, N]`) is split `pid.0`, which accumulates K tiles
    /// `[pid.0 * k_tiles, (pid.0 + 1) * k_tiles)` of row tile zero, unscaled.
    #[cutile::entry()]
    fn fp8_split<const K: i32>(
        partials: &mut Tensor<f32, { [32, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<f8e4m3fn, { [-1, K] }>,
        k_tiles: i32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![32, 64]);
        let wp = weight.partition(shape![64, 64]);
        let total = (K + 63i32) / 64i32;
        let first = pid.0 * k_tiles;
        let mut last = first + k_tiles;
        if last > total {
            last = total;
        }
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for k in first..last {
            let x: Tile<bf16, { [32, 64] }> = convert_tile(xp.load([0i32, k]));
            let w: Tile<bf16, { [64, 64] }> = convert_tile(wp.load([pid.1, k]));
            acc = mmaf(x, w.transpose(), acc);
        }
        partials.store(acc);
    }

    /// Sum the split windows of [`fp8_split`] in split order and apply the channel scales.
    #[cutile::entry()]
    fn reduce_scaled(
        out: &mut Tensor<f32, { [32, 64] }>,
        partials: &Tensor<f32, { [-1, -1] }>,
        scale: &Tensor<f32, { [-1] }>,
        splits: i32,
    ) {
        let pid = get_tile_block_id();
        let pp = partials.partition(shape![32, 64]);
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for split in 0i32..splits {
            acc = acc + pp.load([split, pid.1]);
        }
        let scales = scale
            .partition(shape![64])
            .load([pid.1])
            .reshape(shape![1, 64])
            .broadcast(shape![32, 64]);
        out.store(acc * scales);
    }

    /// Block-scaled FP8 GEMM against F32 activations: each 128-column K block's partial product
    /// is scaled by its own weight scale before it joins the accumulator. `scale` is the
    /// row-expanded `[rows, K / 128]` table the loader builds.
    #[cutile::entry()]
    fn fp8_block<const K: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<f8e4m3fn, { [-1, K] }>,
        scale: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![32, 64]);
        let wp = weight.partition(shape![64, 64]);
        let sp = scale.partition(shape![64, 1]);
        let zero: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        let mut acc: Tile<f32, { [32, 64] }> = zero;
        for block in 0i32..(K / 128i32) {
            let mut part = zero;
            for half in 0i32..2i32 {
                let k = block * 2i32 + half;
                let x: Tile<bf16, { [32, 64] }> = convert_tile(xp.load([pid.0, k]));
                let w: Tile<bf16, { [64, 64] }> = convert_tile(wp.load([pid.1, k]));
                part = mmaf(x, w.transpose(), part);
            }
            let scales = sp
                .load([pid.1, block])
                .reshape(shape![1, 64])
                .broadcast(shape![32, 64]);
            acc = acc + part * scales;
        }
        out.store(acc);
    }

    /// The `K` largest logits of one `B`-wide vocabulary block, in descending order; ties keep
    /// the lower token id first.
    #[cutile::entry()]
    fn top_partial<const V: i32, const B: i32, const K: i32>(
        values: &mut Tensor<f32, { [1, K] }>,
        indices: &mut Tensor<i32, { [1, K] }>,
        logits: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let ids: Tile<i32, { [B] }> = iota(shape![B]) + (pid.1 * B).broadcast(shape![B]);
        let loaded = logits
            .partition(shape![1, B])
            .load([pid.0, pid.1])
            .reshape(shape![B]);
        let masked: Tile<f32, { [B] }> = constant(MASKED, shape![B]);
        let mut x = select(lt_tile(ids, V.broadcast(shape![B])), loaded, masked);
        let slots: Tile<i32, { [K] }> = iota(shape![K]);
        let mut best_values: Tile<f32, { [K] }> = constant(MASKED, shape![K]);
        let mut best_ids: Tile<i32, { [K] }> = constant(0i32, shape![K]);
        for k in 0i32..K {
            let best: Tile<f32, { [] }> = reduce_max(x, 0i32);
            let best = best.reshape(shape![1]);
            let tied = eq_tile(x, best.broadcast(shape![B]));
            let token: Tile<i32, { [] }> =
                reduce_min(select(tied, ids, UNUSED.broadcast(shape![B])), 0i32);
            let token = token.reshape(shape![1]);
            let here = eq_tile(slots, k.broadcast(shape![K]));
            best_values = select(here, best.broadcast(shape![K]), best_values);
            best_ids = select(here, token.broadcast(shape![K]), best_ids);
            x = select(eq_tile(ids, token.broadcast(shape![B])), masked, x);
        }
        values.store(best_values.reshape(shape![1, K]));
        indices.store(best_ids.reshape(shape![1, K]));
    }

    /// Merge one row's `N` block candidates into its final `K` (`B` is `N` rounded up to a
    /// power of two).
    #[cutile::entry()]
    fn top_finish<const N: i32, const B: i32, const K: i32>(
        values: &mut Tensor<f32, { [1, K] }>,
        indices: &mut Tensor<i32, { [1, K] }>,
        partial_values: &Tensor<f32, { [-1, -1] }>,
        partial_indices: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let lanes: Tile<i32, { [B] }> = iota(shape![B]);
        let present = lt_tile(lanes, N.broadcast(shape![B]));
        let masked: Tile<f32, { [B] }> = constant(MASKED, shape![B]);
        let mut x = select(
            present,
            partial_values
                .partition(shape![1, B])
                .load([pid.0, 0i32])
                .reshape(shape![B]),
            masked,
        );
        let ids = select(
            present,
            partial_indices
                .partition(shape![1, B])
                .load([pid.0, 0i32])
                .reshape(shape![B]),
            UNUSED.broadcast(shape![B]),
        );
        let slots: Tile<i32, { [K] }> = iota(shape![K]);
        let mut best_values: Tile<f32, { [K] }> = constant(MASKED, shape![K]);
        let mut best_ids: Tile<i32, { [K] }> = constant(0i32, shape![K]);
        for k in 0i32..K {
            let best: Tile<f32, { [] }> = reduce_max(x, 0i32);
            let best = best.reshape(shape![1]);
            let tied = eq_tile(x, best.broadcast(shape![B]));
            let token: Tile<i32, { [] }> =
                reduce_min(select(tied, ids, UNUSED.broadcast(shape![B])), 0i32);
            let token = token.reshape(shape![1]);
            let here = eq_tile(slots, k.broadcast(shape![K]));
            best_values = select(here, best.broadcast(shape![K]), best_values);
            best_ids = select(here, token.broadcast(shape![K]), best_ids);
            x = select(eq_tile(ids, token.broadcast(shape![B])), masked, x);
        }
        values.store(best_values.reshape(shape![1, K]));
        indices.store(best_ids.reshape(shape![1, K]));
    }

    /// Greedy walk of the candidate lattice for one slot (the reference's `_score_edges` plus
    /// `_selector_walk_kernel` at temperature zero). Step `s` scores row `slot * BLOCK + s`'s
    /// `K` candidates against the token chosen one step earlier (the anchor for the first step):
    /// `unary[c] + Σ_r predecessor[previous, r] * hidden[r] * successor[c, r]`, and keeps the
    /// first maximum. `tokens[slot]` receives the anchor followed by the `BLOCK - 1` choices.
    #[cutile::entry()]
    fn select_walk<const R: i32, const K: i32, const BLOCK: i32>(
        tokens: &mut Tensor<i32, { [1, BLOCK] }>,
        candidates: &Tensor<i32, { [-1, K] }>,
        unary: &Tensor<f32, { [-1, K] }>,
        hidden: &Tensor<f32, { [-1, R] }>,
        predecessor: &Tensor<bf16, { [-1, R] }>,
        successor: &Tensor<bf16, { [-1, R] }>,
        anchors: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut previous: i32 =
            tile_to_scalar(anchors.partition(shape![1]).load([pid.0]).reshape(shape![]));
        let steps: Tile<i32, { [BLOCK] }> = iota(shape![BLOCK]);
        let mut chosen = select(
            eq_tile(steps, 0i32.broadcast(shape![BLOCK])),
            previous.broadcast(shape![BLOCK]),
            0i32.broadcast(shape![BLOCK]),
        );
        let lanes: Tile<i32, { [K] }> = iota(shape![K]);
        let pp = predecessor.partition(shape![1, R]);
        let sp = successor.partition(shape![1, R]);
        for step in 1i32..BLOCK {
            let row = pid.0 * BLOCK + step;
            let h = hidden.partition(shape![1, R]).load([row, 0i32]);
            let before: Tile<f32, { [1, R] }> = convert_tile(pp.load([previous, 0i32]));
            let weighted = before * h;
            let ids = candidates
                .partition(shape![1, K])
                .load([row, 0i32])
                .reshape(shape![K]);
            let logits = unary
                .partition(shape![1, K])
                .load([row, 0i32])
                .reshape(shape![K]);
            let mut best = MASKED;
            let mut token = 0i32;
            for lane in 0i32..K {
                let here = eq_tile(lanes, lane.broadcast(shape![K]));
                let candidate: Tile<i32, { [] }> =
                    reduce_max(select(here, ids, (-1i32).broadcast(shape![K])), 0i32);
                let candidate: i32 = tile_to_scalar(candidate);
                let logit: Tile<f32, { [] }> =
                    reduce_max(select(here, logits, MASKED.broadcast(shape![K])), 0i32);
                let after: Tile<f32, { [1, R] }> = convert_tile(sp.load([candidate, 0i32]));
                let pair: Tile<f32, { [1] }> = reduce_sum(weighted * after, 1i32);
                let pair: f32 = tile_to_scalar(pair.reshape(shape![]));
                let logit: f32 = tile_to_scalar(logit);
                let score = pair + logit;
                if score > best {
                    best = score;
                    token = candidate;
                }
            }
            chosen = select(
                eq_tile(steps, step.broadcast(shape![BLOCK])),
                token.broadcast(shape![BLOCK]),
                chosen,
            );
            previous = token;
        }
        tokens.store(chosen.reshape(shape![1, BLOCK]));
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_dflash2_kernels.rs"]
mod tests;
