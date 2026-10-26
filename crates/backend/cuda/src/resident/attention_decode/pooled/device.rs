//! Original per-slot attention arithmetic with a group of independent slots per launch.
#[cutile::module]
pub(super) mod kernels {
    use cutile::core::{
        Array_3, BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, Shape_3, Shape_4,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tensor_4, Tile_0, Tile_1, Tile_2,
        Tile_3, bf16, bitcast, cat, constant, convert_scalar, convert_tile, exp, f8e4m3fn, ftz,
        ge_tile, get_tile_block_id, iota, le_tile, max_tile, min_tile, mmaf, reduce_max,
        reduce_sum, rsqrt, select, tile_to_scalar,
    };

    /// Header, metadata and page tables share one integer binding without rounding scale bits.
    #[cutile::entry()]
    fn pack_control<const C: i32>(
        out: &mut Tensor<i32, { [1, C] }>,
        t0: &Tensor<i32, { [-1] }>,
        t1: &Tensor<i32, { [-1] }>,
        t2: &Tensor<i32, { [-1] }>,
        t3: &Tensor<i32, { [-1] }>,
        parameters: &Tensor<i32, { [32] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        if pid.0 == 0 {
            out.store(
                parameters
                    .partition(shape![C])
                    .load([0i32])
                    .reshape(shape![1, C]),
            );
        } else if pid.0 == 1 {
            out.store(
                metadata
                    .partition(shape![C])
                    .load([0i32])
                    .reshape(shape![1, C]),
            );
        } else if pid.0 == 2 {
            out.store(t0.partition(shape![C]).load([0i32]).reshape(shape![1, C]));
        } else if pid.0 == 3 {
            out.store(t1.partition(shape![C]).load([0i32]).reshape(shape![1, C]));
        } else if pid.0 == 4 {
            out.store(t2.partition(shape![C]).load([0i32]).reshape(shape![1, C]));
        } else {
            out.store(t3.partition(shape![C]).load([0i32]).reshape(shape![1, C]));
        }
    }

    /// One launch packs every slot's control row: row 0 header, row 1 metadata,
    /// rows 2..2+slots the stacked page tables. Grid rows come from the output partition.
    #[cutile::entry()]
    fn pack_control_batch<const C: i32>(
        out: &mut Tensor<i32, { [1, C] }>,
        tables: &Tensor<i32, { [-1, C] }>,
        parameters: &Tensor<i32, { [32] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        if pid.0 == 0 {
            out.store(
                parameters
                    .partition(shape![C])
                    .load([0i32])
                    .reshape(shape![1, C]),
            );
        } else if pid.0 == 1 {
            out.store(
                metadata
                    .partition(shape![C])
                    .load([0i32])
                    .reshape(shape![1, C]),
            );
        } else {
            out.store(tables.partition(shape![1, C]).load([pid.0 - 2i32, 0i32]));
        }
    }

    fn word(control: &Tensor<i32, { [-1, -1] }>, row: i32, column: i32) -> i32 {
        tile_to_scalar(
            control
                .partition(shape![1, 1])
                .load([row, column])
                .reshape(shape![]),
        )
    }

    fn scale(control: &Tensor<i32, { [-1, -1] }>, column: i32) -> f32 {
        let raw: Tile<f32, { [1, 1] }> =
            bitcast(control.partition(shape![1, 1]).load([0i32, column]));
        tile_to_scalar(raw.reshape(shape![]))
    }

    /// One CTA owns a whole KV head, retaining exclusive writes while walking its candidates.
    #[cutile::entry()]
    fn append_pages<
        const D: i32,
        const CAP: i32,
        const KH: i32,
        const LANES: i32,
        const MF: i32,
    >(
        keys: &mut Tensor<f8e4m3fn, { [1, CAP, D] }>,
        values: &mut Tensor<f8e4m3fn, { [1, CAP, D] }>,
        input_keys: &Tensor<f32, { [-1, KH, D] }>,
        input_values: &Tensor<f32, { [-1, KH, D] }>,
        control: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let ks = scale(control, 0i32);
        let vs = scale(control, 1i32);
        for lane in 0i32..LANES {
            let position = word(control, 1i32, lane * MF + 2i32);
            if position >= 0i32 {
                let physical = word(control, lane / 3i32 + 2i32, position / 32i32);
                let row = physical * 32i32 + position % 32i32;
                let key = input_keys
                    .partition(shape![1, 1, D])
                    .load([lane, pid.0, 0i32]);
                let value = input_values
                    .partition(shape![1, 1, D])
                    .load([lane, pid.0, 0i32]);
                let low = (-448.0f32).broadcast(shape![1, 1, D]);
                let high = 448.0f32.broadcast(shape![1, 1, D]);
                let normalized_key =
                    min_tile(max_tile(key / ks.broadcast(shape![1, 1, D]), low), high);
                let normalized_value =
                    min_tile(max_tile(value / vs.broadcast(shape![1, 1, D]), low), high);
                let packed_key: Tile<f8e4m3fn, { [1, 1, D] }> = convert_tile(normalized_key);
                let packed_value: Tile<f8e4m3fn, { [1, 1, D] }> = convert_tile(normalized_value);
                keys.partition_mut(shape![1, 1, D])
                    .store(packed_key, [0i32, row, 0i32]);
                values
                    .partition_mut(shape![1, 1, D])
                    .store(packed_value, [0i32, row, 0i32]);
            }
        }
    }

    fn window_start(position: i32, window: i32) -> i32 {
        let candidate = position + 1i32 - window;
        let valid = ge_tile(window.broadcast(shape![1]), 1i32.broadcast(shape![1]))
            & ge_tile(candidate.broadcast(shape![1]), 1i32.broadcast(shape![1]));
        tile_to_scalar(
            select(
                valid,
                candidate.broadcast(shape![1]),
                0i32.broadcast(shape![1]),
            )
            .reshape(shape![]),
        )
    }

    fn grouped_scale<const D: i32>(k_scale: f32) -> f32 {
        let d: f32 = convert_scalar(D);
        tile_to_scalar(
            (rsqrt(d.broadcast(shape![1]), ftz::Disabled) * k_scale.broadcast(shape![1]))
                .reshape(shape![]),
        )
    }

    fn stacked_query<const D: i32, const KH: i32>(
        queries: &Tensor<f32, { [-1, -1, D] }>,
        slot: i32,
        kv: i32,
    ) -> Tile<bf16, { [32, D] }> {
        let base = slot * 3i32 * KH + kv;
        let first: Tile<f32, { [8, D] }> = queries
            .partition(shape![1, 8, D])
            .load([base, 0i32, 0i32])
            .reshape(shape![8, D]);
        let second: Tile<f32, { [8, D] }> = queries
            .partition(shape![1, 8, D])
            .load([base + KH, 0i32, 0i32])
            .reshape(shape![8, D]);
        let third: Tile<f32, { [8, D] }> = queries
            .partition(shape![1, 8, D])
            .load([base + 2i32 * KH, 0i32, 0i32])
            .reshape(shape![8, D]);
        let pad: Tile<f32, { [8, D] }> = constant(0.0f32, shape![8, D]);
        let low: Tile<f32, { [16, D] }> = cat(first, second, 0i32);
        let high: Tile<f32, { [16, D] }> = cat(third, pad, 0i32);
        let rows: Tile<f32, { [32, D] }> = cat(low, high, 0i32);
        convert_tile(rows)
    }

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

    #[cutile::entry()]
    fn partial_slots<const D: i32, const KH: i32, const PARTS: i32, const MF: i32>(
        out: &mut Tensor<f32, { [1, 32, D] }>,
        peaks: &mut Tensor<f32, { [1, 32] }>,
        masses: &mut Tensor<f32, { [1, 32] }>,
        queries: &Tensor<f32, { [-1, -1, D] }>,
        keys: &Tensor<f8e4m3fn, { [-1, -1, D] }>,
        values: &Tensor<f8e4m3fn, { [-1, -1, D] }>,
        control: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let slot = pid.0 / (KH * PARTS);
        let kv = pid.0 / PARTS % KH;
        let split = pid.0 % PARTS;
        let factor = grouped_scale::<D>(scale(control, 0i32));
        let amplitude = scale(control, 1i32);
        let window = word(control, 0i32, 2i32);
        let p0 = word(control, 1i32, slot * 3i32 * MF + 2i32);
        let p1 = word(control, 1i32, (slot * 3i32 + 1i32) * MF + 2i32);
        let p2 = word(control, 1i32, (slot * 3i32 + 2i32) * MF + 2i32);
        let s0 = window_start(p0, window);
        let s1 = window_start(p1, window);
        let s2 = window_start(p2, window);
        let tile = stacked_query::<D, KH>(queries, slot, kv);
        let row_first = stacked_rows(s0, s1, s2);
        let row_last = stacked_rows(p0, p1, p2);
        let zero_scores: Tile<f32, { [32, 32] }> = constant(0.0f32, shape![32, 32]);
        let zero_rows: Tile<f32, { [32, D] }> = constant(0.0f32, shape![32, D]);
        let masked: Tile<f32, { [32, 32] }> = constant(-1.0e30f32, shape![32, 32]);
        let mut numerator = zero_rows;
        let mut denominator: Tile<f32, { [32] }> = constant(0.0f32, shape![32]);
        let mut maximum: Tile<f32, { [32] }> = constant(-1.0e30f32, shape![32]);
        let position = furthest_position(p0, p1, p2);
        let start = earliest_start(s0, s1, s2);
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
        let kp = keys.partition_permuted(shape![1, D, 32], const_array![0, 2, 1]);
        let vp = values.partition(shape![1, 32, D]);
        let blocks = control.partition(shape![1, 1]);
        let offsets: Tile<i32, { [32] }> = iota(shape![32]);
        for block in first..last {
            let physical: i32 = tile_to_scalar(blocks.load([slot + 2i32, block]).reshape(shape![]));
            let key_t: Tile<bf16, { [D, 32] }> = convert_tile(
                kp.load_pipelined::<4>([kv, 0i32, physical])
                    .reshape(shape![D, 32]),
            );
            let block_values: Tile<bf16, { [32, D] }> = convert_tile(
                vp.load_pipelined::<4>([kv, physical, 0i32])
                    .reshape(shape![32, D]),
            );
            let key_rows = offsets + (block * 32i32).broadcast(shape![32]);
            let columns: Tile<i32, { [32, 32] }> =
                key_rows.reshape(shape![1, 32]).broadcast(shape![32, 32]);
            let valid = ge_tile(columns, row_first) & le_tile(columns, row_last);
            let raw: Tile<f32, { [32, 32] }> = mmaf(tile, key_t, zero_scores);
            let scores = select(valid, raw * factor.broadcast(shape![32, 32]), masked);
            let block_max: Tile<f32, { [32] }> = reduce_max(scores, 1i32);
            let next = max_tile(block_max, maximum);
            let old = exp(maximum - next);
            let p = exp(scores - next.reshape(shape![32, 1]).broadcast(shape![32, 32]));
            let sum: Tile<f32, { [32] }> = reduce_sum(p, 1i32);
            let weights: Tile<bf16, { [32, 32] }> = convert_tile(p);
            let product: Tile<f32, { [32, D] }> = mmaf(weights, block_values, zero_rows);
            numerator = numerator * old.reshape(shape![32, 1]).broadcast(shape![32, D]) + product;
            denominator = denominator * old + sum;
            maximum = next;
        }
        out.store((numerator * amplitude.broadcast(shape![32, D])).reshape(shape![1, 32, D]));
        peaks.store(maximum.reshape(shape![1, 32]));
        masses.store(denominator.reshape(shape![1, 32]));
    }

    #[cutile::entry()]
    fn merge_slots<const D: i32, const PARTS: i32, const GROUP: i32, const HEADS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        numerator: &Tensor<f32, { [-1, PARTS, 32, D] }>,
        peaks: &Tensor<f32, { [-1, PARTS, 32] }>,
        masses: &Tensor<f32, { [-1, PARTS, 32] }>,
        first_slot: i32,
        batch: i32,
    ) {
        let pid = get_tile_block_id();
        let slot = pid.0 / (3i32 * HEADS);
        if slot >= first_slot && slot < first_slot + batch {
            let head = pid.0 % HEADS;
            let lane = pid.0 / HEADS % 3i32;
            let kv = (slot - first_slot) * (HEADS / GROUP) + head / GROUP;
            let row = lane * 8i32 + head % GROUP;
            let maxima = peaks
                .partition(shape![1, PARTS, 1])
                .load([kv, 0i32, row])
                .reshape(shape![PARTS]);
            let maximum: Tile<f32, { [] }> = reduce_max(maxima, 0i32);
            let weights = exp(maxima - maximum.reshape(shape![1]).broadcast(shape![PARTS]));
            let sums = masses
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
    }
}
