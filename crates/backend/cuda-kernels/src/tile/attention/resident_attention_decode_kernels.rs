//! Split long KV reductions across independent CTAs before stable softmax merging.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        Array_3, BroadcastScalar, ElementType, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        Shape_4, StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tensor_4, Tile_0, Tile_1,
        Tile_2, bf16, cat, cmp_ordering, cmpf, constant, convert_scalar, convert_tile, exp, ftz,
        ge_tile, get_tile_block_id, iota, le_tile, max_tile, mmaf, predicate, reduce_max,
        reduce_sum, rsqrt, select, tile_to_scalar,
    };
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow and exclusive ranges"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[cutile::entry()]
    fn partial<E: ElementType, const D: i32, const GROUP: i32, const PARTS: i32, const BT: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        maxima: &mut Tensor<f32, { [1, 1] }>,
        sums: &mut Tensor<f32, { [1, 1] }>,
        q: &Tensor<f32, { [-1, D] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
        table: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        if position >= 0 {
            let mut start = 0i32;
            if window > 0 && position + 1i32 > window {
                start = position + 1i32 - window;
            }
            let head = pid.0 / PARTS;
            let split = pid.0 % PARTS;
            let query = q
                .partition(shape![1, D])
                .load([head, 0i32])
                .reshape(shape![1, D])
                .broadcast(shape![32, D]);
            let d: f32 = convert_scalar(D);
            let scale: Tile<f32, { [32] }> = rsqrt(d.broadcast(shape![32]), ftz::Disabled);
            let kp = keys.partition(shape![1, 32, D]);
            let vp = values.partition(shape![1, 32, D]);
            let blocks = table.partition(shape![1]);
            let mut numerator: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
            let mut denominator: Tile<f32, { [1] }> = constant(0.0f32, shape![1]);
            let mut maximum: Tile<f32, { [1] }> = constant(-1.0e30f32, shape![1]);
            let first = start / 32i32;
            let end = position / 32i32 + 1i32;
            let block_count = (end - first + PARTS - 1i32) / PARTS;
            let first = first + split * block_count;
            let mut last = first + block_count;
            if last > end {
                last = end;
            }
            for block in first..last {
                let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
                let key: Tile<f32, { [32, D] }> = convert_tile(
                    kp.load([head / GROUP, physical, 0i32])
                        .reshape(shape![32, D]),
                );
                let value: Tile<f32, { [32, D] }> = convert_tile(
                    vp.load([head / GROUP, physical, 0i32])
                        .reshape(shape![32, D]),
                );
                let key: Tile<f32, { [32, D] }> = key * k_scale.broadcast(shape![32, D]);
                let value: Tile<f32, { [32, D] }> = value * v_scale.broadcast(shape![32, D]);
                let score: Tile<f32, { [32] }> = reduce_sum(query * key, 1i32);
                let score = score * scale;
                let offsets: Tile<i32, { [32] }> = iota(shape![32]);
                let positions = offsets + (block * 32i32).broadcast(shape![32]);
                let valid = ge_tile(positions, start.broadcast(shape![32]))
                    & le_tile(positions, position.broadcast(shape![32]));
                let masked: Tile<f32, { [32] }> = constant(-1.0e30f32, shape![32]);
                let score = select(valid, score, masked);
                let block_max: Tile<f32, { [] }> = reduce_max(score, 0i32);
                let block_max = block_max.reshape(shape![1]);
                let greater = cmpf(
                    block_max,
                    maximum,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let next_max: Tile<f32, { [1] }> = select(greater, block_max, maximum);
                let old_scale: Tile<f32, { [1] }> = exp(maximum - next_max);
                let probabilities: Tile<f32, { [32] }> =
                    exp(score - next_max.broadcast(shape![32]));
                let block_sum: Tile<f32, { [] }> = reduce_sum(probabilities, 0i32);
                let contribution: Tile<f32, { [D] }> = reduce_sum(
                    value
                        * probabilities
                            .reshape(shape![32, 1])
                            .broadcast(shape![32, D]),
                    0i32,
                );
                numerator =
                    numerator * old_scale.broadcast(shape![D]) + contribution.reshape(shape![D]);
                denominator = denominator * old_scale + block_sum.reshape(shape![1]);
                maximum = next_max;
            }
            out.store(numerator.reshape(shape![1, D]));
            maxima.store(maximum.reshape(shape![1, 1]));
            sums.store(denominator.reshape(shape![1, 1]));
        } else {
            out.store(0.0f32.broadcast(shape![1, D]));
            maxima.store((-1.0e30f32).broadcast(shape![1, 1]));
            sums.store(0.0f32.broadcast(shape![1, 1]));
        }
    }
    /// Split-KV decode attention for grouped queries: one CTA per (KV head, split) reads each KV
    /// block once and scores the group's query heads (padded to 16 rows; `q` is viewed as
    /// `[kv_heads, group, D]`) with BF16 tensor-core products.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[expect(
        clippy::useless_let_if_seq,
        clippy::similar_names,
        reason = "cuTile DSL uses statement control flow; lane 0 names mirror the three-lane kernel"
    )]
    #[cutile::entry()]
    fn partial_gqa<E: ElementType, const D: i32, const PARTS: i32, const BT: i32>(
        out0: &mut Tensor<f32, { [1, 16, D] }>,
        maxima0: &mut Tensor<f32, { [1, 16] }>,
        sums0: &mut Tensor<f32, { [1, 16] }>,
        q0: &Tensor<f32, { [-1, -1, D] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        table: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let kv = pid.0 / PARTS;
        let split = pid.0 % PARTS;
        let d: f32 = convert_scalar(D);
        let scale: f32 = tile_to_scalar(
            (rsqrt(d.broadcast(shape![1]), ftz::Disabled) * k_scale.broadcast(shape![1]))
                .reshape(shape![]),
        );
        let zero_scores: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        let zero_rows: Tile<f32, { [16, D] }> = constant(0.0f32, shape![16, D]);
        let masked: Tile<f32, { [16, 32] }> = constant(-1.0e30f32, shape![16, 32]);
        let position0: i32 = tile_to_scalar(
            metadata0
                .partition(shape![1])
                .load([2i32])
                .reshape(shape![]),
        );
        let mut start0 = 0i32;
        if window > 0 && position0 + 1i32 > window {
            start0 = position0 + 1i32 - window;
        }
        let query0: Tile<f32, { [16, D] }> = q0
            .partition(shape![1, 16, D])
            .load([kv, 0i32, 0i32])
            .reshape(shape![16, D]);
        let query0: Tile<bf16, { [16, D] }> = convert_tile(query0);
        let mut numerator0 = zero_rows;
        let mut denominator0: Tile<f32, { [16] }> = constant(0.0f32, shape![16]);
        let mut maximum0: Tile<f32, { [16] }> = constant(-1.0e30f32, shape![16]);
        // Blocks up to the furthest lane; earlier lanes mask the rows past their own position.
        let position = position0;
        let start = start0;
        let mut first = 0i32;
        let mut last = 0i32;
        if position >= 0i32 {
            let lowest = start / 32i32;
            let end = position / 32i32 + 1i32;
            let block_count = (end - lowest + PARTS - 1i32) / PARTS;
            first = lowest + split * block_count;
            last = first + block_count;
            if last > end {
                last = end;
            }
        }
        // Keys load already transposed for the score product.
        let kp = keys.partition_permuted(shape![1, D, 32], const_array![0, 2, 1]);
        let vp = values.partition(shape![1, 32, D]);
        let blocks = table.partition(shape![1]);
        let offsets: Tile<i32, { [32] }> = iota(shape![32]);
        for block in first..last {
            let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
            let key_t: Tile<bf16, { [D, 32] }> = convert_tile(
                kp.load_pipelined::<4>([kv, 0i32, physical])
                    .reshape(shape![D, 32]),
            );
            let value: Tile<bf16, { [32, D] }> = convert_tile(
                vp.load_pipelined::<4>([kv, physical, 0i32])
                    .reshape(shape![32, D]),
            );
            let positions = offsets + (block * 32i32).broadcast(shape![32]);
            let raw0: Tile<f32, { [16, 32] }> = mmaf(query0, key_t, zero_scores);
            let valid0 = ge_tile(positions, start0.broadcast(shape![32]))
                & le_tile(positions, position0.broadcast(shape![32]));
            let scores0 = select(
                valid0.reshape(shape![1, 32]).broadcast(shape![16, 32]),
                raw0 * scale.broadcast(shape![16, 32]),
                masked,
            );
            let block_max0: Tile<f32, { [16] }> = reduce_max(scores0, 1i32);
            let next0 = max_tile(block_max0, maximum0);
            let old0 = exp(maximum0 - next0);
            let p0 = exp(scores0 - next0.reshape(shape![16, 1]).broadcast(shape![16, 32]));
            let sum0: Tile<f32, { [16] }> = reduce_sum(p0, 1i32);
            let w0: Tile<bf16, { [16, 32] }> = convert_tile(p0);
            let c0: Tile<f32, { [16, D] }> = mmaf(w0, value, zero_rows);
            numerator0 = numerator0 * old0.reshape(shape![16, 1]).broadcast(shape![16, D]) + c0;
            denominator0 = denominator0 * old0 + sum0;
            maximum0 = next0;
        }
        out0.store((numerator0 * v_scale.broadcast(shape![16, D])).reshape(shape![1, 16, D]));
        maxima0.store(maximum0.reshape(shape![1, 16]));
        sums0.store(denominator0.reshape(shape![1, 16]));
    }

    fn query_position(metadata: &Tensor<i32, { [-1] }>) -> i32 {
        tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]))
    }

    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow for scalar window bounds"
    )]
    const fn window_start(position: i32, window: i32) -> i32 {
        let mut start = 0i32;
        if window > 0 && position + 1i32 > window {
            start = position + 1i32 - window;
        }
        start
    }

    fn grouped_scale<const D: i32>(k_scale: f32) -> f32 {
        let d: f32 = convert_scalar(D);
        tile_to_scalar(
            (rsqrt(d.broadcast(shape![1]), ftz::Disabled) * k_scale.broadcast(shape![1]))
                .reshape(shape![]),
        )
    }

    /// One query tile for the three verification lanes: lane `l` owns rows `8 l .. 8 l + GROUP`
    /// and the unused rows are zero.
    fn stacked_query<const D: i32>(
        q0: &Tensor<f32, { [-1, -1, D] }>,
        q1: &Tensor<f32, { [-1, -1, D] }>,
        q2: &Tensor<f32, { [-1, -1, D] }>,
        kv: i32,
    ) -> Tile<bf16, { [32, D] }> {
        let first: Tile<f32, { [8, D] }> = q0
            .partition(shape![1, 8, D])
            .load([kv, 0i32, 0i32])
            .reshape(shape![8, D]);
        let second: Tile<f32, { [8, D] }> = q1
            .partition(shape![1, 8, D])
            .load([kv, 0i32, 0i32])
            .reshape(shape![8, D]);
        let third: Tile<f32, { [8, D] }> = q2
            .partition(shape![1, 8, D])
            .load([kv, 0i32, 0i32])
            .reshape(shape![8, D]);
        let pad: Tile<f32, { [8, D] }> = constant(0.0f32, shape![8, D]);
        let low: Tile<f32, { [16, D] }> = cat(first, second, 0i32);
        let high: Tile<f32, { [16, D] }> = cat(third, pad, 0i32);
        let rows: Tile<f32, { [32, D] }> = cat(low, high, 0i32);
        convert_tile(rows)
    }

    /// Per-row value of a three-lane quantity in the stacked layout of [`stacked_query`].
    fn stacked_rows(first: i32, second: i32, third: i32) -> Tile<i32, { [32, 32] }> {
        let rows: Tile<i32, { [32] }> = iota(shape![32]);
        let later = ge_tile(rows, 8i32.broadcast(shape![32]));
        let last = ge_tile(rows, 16i32.broadcast(shape![32]));
        let early: Tile<i32, { [32] }> = select(
            later,
            second.broadcast(shape![32]),
            first.broadcast(shape![32]),
        );
        let value: Tile<i32, { [32] }> = select(last, third.broadcast(shape![32]), early);
        value.reshape(shape![32, 1]).broadcast(shape![32, 32])
    }

    const fn furthest_position(first: i32, second: i32, third: i32) -> i32 {
        let mut result = first;
        if second > result {
            result = second;
        }
        if third > result {
            result = third;
        }
        result
    }

    const fn earliest_start(first: i32, second: i32, third: i32) -> i32 {
        let mut result = first;
        if second < result {
            result = second;
        }
        if third < result {
            result = third;
        }
        result
    }

    /// [`partial_gqa`] for the three lanes of an MTP verification step: the lanes' query heads
    /// are stacked into one 32-row tile, so each KV block is read once and scored for every lane
    /// with one product, each row masked to its own lane's position.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[expect(
        clippy::similar_names,
        reason = "Matching query and position names identify each of the three verification lanes"
    )]
    #[cutile::entry()]
    fn partial_gqa3<E: ElementType, const D: i32, const PARTS: i32, const BT: i32>(
        out: &mut Tensor<f32, { [1, 32, D] }>,
        maxima: &mut Tensor<f32, { [1, 32] }>,
        sums: &mut Tensor<f32, { [1, 32] }>,
        q0: &Tensor<f32, { [-1, -1, D] }>,
        q1: &Tensor<f32, { [-1, -1, D] }>,
        q2: &Tensor<f32, { [-1, -1, D] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
        keys: &Tensor<E, { [-1, -1, D] }>,
        values: &Tensor<E, { [-1, -1, D] }>,
        table: &Tensor<i32, { [-1] }>,
        window: i32,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let kv = pid.0 / PARTS;
        let split = pid.0 % PARTS;
        let scale = grouped_scale::<D>(k_scale);
        let position0 = query_position(metadata0);
        let position1 = query_position(metadata1);
        let position2 = query_position(metadata2);
        let start0 = window_start(position0, window);
        let start1 = window_start(position1, window);
        let start2 = window_start(position2, window);
        let query = stacked_query::<D>(q0, q1, q2, kv);
        let row_first = stacked_rows(start0, start1, start2);
        let row_last = stacked_rows(position0, position1, position2);
        let zero_scores: Tile<f32, { [32, 32] }> = constant(0.0f32, shape![32, 32]);
        let zero_rows: Tile<f32, { [32, D] }> = constant(0.0f32, shape![32, D]);
        let masked: Tile<f32, { [32, 32] }> = constant(-1.0e30f32, shape![32, 32]);
        let mut numerator = zero_rows;
        let mut denominator: Tile<f32, { [32] }> = constant(0.0f32, shape![32]);
        let mut maximum: Tile<f32, { [32] }> = constant(-1.0e30f32, shape![32]);
        // Blocks up to the furthest lane; earlier lanes mask the rows past their own position.
        let position = furthest_position(position0, position1, position2);
        let start = earliest_start(start0, start1, start2);
        let mut first = 0i32;
        let mut last = 0i32;
        if position >= 0i32 {
            let lowest = start / 32i32;
            let end = position / 32i32 + 1i32;
            let block_count = (end - lowest + PARTS - 1i32) / PARTS;
            first = lowest + split * block_count;
            last = first + block_count;
            if last > end {
                last = end;
            }
        }
        // Keys load already transposed for the score product.
        let kp = keys.partition_permuted(shape![1, D, 32], const_array![0, 2, 1]);
        let vp = values.partition(shape![1, 32, D]);
        let blocks = table.partition(shape![1]);
        let offsets: Tile<i32, { [32] }> = iota(shape![32]);
        for block in first..last {
            let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
            let key_t: Tile<bf16, { [D, 32] }> = convert_tile(
                kp.load_pipelined::<4>([kv, 0i32, physical])
                    .reshape(shape![D, 32]),
            );
            let value: Tile<bf16, { [32, D] }> = convert_tile(
                vp.load_pipelined::<4>([kv, physical, 0i32])
                    .reshape(shape![32, D]),
            );
            let positions = offsets + (block * 32i32).broadcast(shape![32]);
            let columns: Tile<i32, { [32, 32] }> =
                positions.reshape(shape![1, 32]).broadcast(shape![32, 32]);
            let valid = ge_tile(columns, row_first) & le_tile(columns, row_last);
            let raw: Tile<f32, { [32, 32] }> = mmaf(query, key_t, zero_scores);
            let scores = select(valid, raw * scale.broadcast(shape![32, 32]), masked);
            let block_max: Tile<f32, { [32] }> = reduce_max(scores, 1i32);
            let next = max_tile(block_max, maximum);
            let old = exp(maximum - next);
            let p = exp(scores - next.reshape(shape![32, 1]).broadcast(shape![32, 32]));
            let sum: Tile<f32, { [32] }> = reduce_sum(p, 1i32);
            let weights: Tile<bf16, { [32, 32] }> = convert_tile(p);
            let product: Tile<f32, { [32, D] }> = mmaf(weights, value, zero_rows);
            numerator = numerator * old.reshape(shape![32, 1]).broadcast(shape![32, D]) + product;
            denominator = denominator * old + sum;
            maximum = next;
        }
        out.store((numerator * v_scale.broadcast(shape![32, D])).reshape(shape![1, 32, D]));
        maxima.store(maximum.reshape(shape![1, 32]));
        sums.store(denominator.reshape(shape![1, 32]));
    }

    /// Merge grouped splits: head `h` reads row `offset + h % GROUP` of its KV head's partials.
    #[cutile::entry()]
    fn merge_gqa<const D: i32, const PARTS: i32, const GROUP: i32, const ROWS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        numerator: &Tensor<f32, { [-1, PARTS, ROWS, D] }>,
        maxima: &Tensor<f32, { [-1, PARTS, ROWS] }>,
        sums: &Tensor<f32, { [-1, PARTS, ROWS] }>,
        offset: i32,
    ) {
        let pid = get_tile_block_id();
        let kv = pid.0 / GROUP;
        let row = offset + pid.0 % GROUP;
        let maxima = maxima
            .partition(shape![1, PARTS, 1])
            .load([kv, 0i32, row])
            .reshape(shape![PARTS]);
        let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0i32);
        let weights = exp(maxima - maximum.reshape(shape![1]).broadcast(shape![PARTS]));
        let sums = sums
            .partition(shape![1, PARTS, 1])
            .load([kv, 0i32, row])
            .reshape(shape![PARTS]);
        let total: Tile<f32, { [] }> = reduce_sum(sums * weights, 0i32);
        let total = max_tile(total.reshape(shape![1]), 1.0e-20f32.broadcast(shape![1]));
        let values = numerator
            .partition(shape![1, PARTS, 1, D])
            .load([kv, 0i32, row, 0i32])
            .reshape(shape![PARTS, D]);
        let sum: Tile<f32, { [D] }> = reduce_sum(
            values
                * weights
                    .reshape(shape![PARTS, 1])
                    .broadcast(shape![PARTS, D]),
            0i32,
        );
        out.store((sum / total.broadcast(shape![D])).reshape(shape![1, D]));
    }

    #[cutile::entry()]
    fn merge<const D: i32, const PARTS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        numerator: &Tensor<f32, { [-1, PARTS, D] }>,
        maxima: &Tensor<f32, { [-1, PARTS] }>,
        sums: &Tensor<f32, { [-1, PARTS] }>,
    ) {
        let pid = get_tile_block_id();
        let maxima = maxima
            .partition(shape![1, PARTS])
            .load([pid.0, 0i32])
            .reshape(shape![PARTS]);
        let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0i32);
        let weights = exp(maxima - maximum.reshape(shape![1]).broadcast(shape![PARTS]));
        let sums = sums
            .partition(shape![1, PARTS])
            .load([pid.0, 0i32])
            .reshape(shape![PARTS]);
        let total: Tile<f32, { [] }> = reduce_sum(sums * weights, 0i32);
        let total = max_tile(total.reshape(shape![1]), 1.0e-20f32.broadcast(shape![1]));
        let values = numerator
            .partition(shape![1, PARTS, D])
            .load([pid.0, 0i32, 0i32])
            .reshape(shape![PARTS, D]);
        let sum: Tile<f32, { [D] }> = reduce_sum(
            values
                * weights
                    .reshape(shape![PARTS, 1])
                    .broadcast(shape![PARTS, D]),
            0i32,
        );
        out.store((sum / total.broadcast(shape![D])).reshape(shape![1, D]));
    }
}
