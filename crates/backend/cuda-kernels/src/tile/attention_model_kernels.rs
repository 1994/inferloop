#[cutile::module]
pub mod attention {
    use cutile::core::{
        BroadcastScalar, ElementType, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3,
        bf16, cmp_ordering, cmpf, constant, convert_scalar, convert_tile, exp, ftz, ge_tile,
        get_tile_block_id, iota, le_tile, max_tile, min_tile, predicate, reduce_max, reduce_sum,
        rsqrt, select, tile_to_scalar,
    };

    #[cutile::entry()]
    fn embedding<const D: i32, const OVERRIDE: i32>(
        out: &mut Tensor<f32, { [256] }>,
        weight: &Tensor<bf16, { [-1, D] }>,
        external: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let mp = metadata.partition(shape![1]);
        let override_input: i32 = tile_to_scalar(mp.load([3i32]).reshape(shape![]));
        if OVERRIDE == 1 && override_input == 1 {
            out.store(external.load_like(out));
        } else {
            let token: i32 = tile_to_scalar(mp.load([1i32]).reshape(shape![]));
            let pid = get_tile_block_id();
            let w = weight.partition(shape![1, 256]).load([token, pid.0]);
            let value: Tile<f32, { [256] }> = convert_tile(w.reshape(shape![256]));
            out.store(value);
        }
    }

    /// One gather for every prompt lane: row `pid.0` reads token id `metadata[row][1]`
    /// and copies the 256-wide weight slice at column tile `pid.1`. Pure copy + convert,
    /// bit-identical to the per-lane `embedding` kernel.
    #[cutile::entry()]
    fn embedding_batch<const D: i32, const FIELDS: i32>(
        out: &mut Tensor<f32, { [1, 256] }>,
        weight: &Tensor<bf16, { [-1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let token: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([pid.0 * FIELDS + 1i32])
                .reshape(shape![]),
        );
        let w = weight.partition(shape![1, 256]).load([token, pid.1]);
        let value: Tile<f32, { [1, 256] }> = convert_tile(w);
        out.store(value);
    }

    /// Target twin of `embedding_batch`: row `pid.0` copies its token's 256-wide weight slice at
    /// column tile `pid.1`, or that lane's external hidden row when its metadata marks an
    /// override. Bit-identical to the per-lane `embedding` kernel with `OVERRIDE = 1`.
    #[cutile::entry()]
    fn embedding_rows<const D: i32, const FIELDS: i32>(
        out: &mut Tensor<f32, { [1, 256] }>,
        weight: &Tensor<bf16, { [-1, D] }>,
        external: &Tensor<f32, { [-1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let fields = metadata.partition(shape![1]);
        let override_input: i32 =
            tile_to_scalar(fields.load([pid.0 * FIELDS + 3i32]).reshape(shape![]));
        if override_input == 1 {
            let row: Tile<f32, { [1, 256] }> =
                external.partition(shape![1, 256]).load([pid.0, pid.1]);
            out.store(row);
        } else {
            let token: i32 = tile_to_scalar(fields.load([pid.0 * FIELDS + 1i32]).reshape(shape![]));
            let w = weight.partition(shape![1, 256]).load([token, pid.1]);
            let value: Tile<f32, { [1, 256] }> = convert_tile(w);
            out.store(value);
        }
    }

    #[cutile::entry()]
    fn append<E: ElementType, const D: i32, const CAP: i32, const QUANT: i32>(
        keys: &mut Tensor<E, { [1, CAP, D] }>,
        values: &mut Tensor<E, { [1, CAP, D] }>,
        k: &Tensor<f32, { [-1, D] }>,
        v: &Tensor<f32, { [-1, D] }>,
        metadata: &Tensor<i32, { [-1] }>,
        table: &Tensor<i32, { [-1] }>,
        k_scale: f32,
        v_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        if position >= 0 {
            // Logical block to physical block; an identity table keeps the same row.
            let physical: i32 = tile_to_scalar(
                table
                    .partition(shape![1])
                    .load([position / 32i32])
                    .reshape(shape![]),
            );
            let row = physical * 32i32 + position % 32i32;
            let mut k = k
                .partition(shape![1, D])
                .load([pid.0, 0i32])
                .reshape(shape![1, 1, D]);
            let mut v = v
                .partition(shape![1, D])
                .load([pid.0, 0i32])
                .reshape(shape![1, 1, D]);
            if QUANT == 1 {
                let low = (-448.0f32).broadcast(shape![1, 1, D]);
                let high = 448.0f32.broadcast(shape![1, 1, D]);
                k = min_tile(max_tile(k / k_scale.broadcast(shape![1, 1, D]), low), high);
                v = min_tile(max_tile(v / v_scale.broadcast(shape![1, 1, D]), low), high);
            }
            let k: Tile<E, { [1, 1, D] }> = convert_tile(k);
            let v: Tile<E, { [1, 1, D] }> = convert_tile(v);
            keys.partition_mut(shape![1, 1, D])
                .store(k, [0i32, row, 0i32]);
            values
                .partition_mut(shape![1, 1, D])
                .store(v, [0i32, row, 0i32]);
        }
    }

    #[expect(
        clippy::useless_let_if_seq,
        clippy::range_plus_one,
        reason = "cuTile DSL uses statement control flow and exclusive ranges"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA entry takes explicit tensor bindings and independent KV scales"
    )]
    #[cutile::entry()]
    fn decode<E: ElementType, const D: i32, const GROUP: i32, const BT: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
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
            let query = q
                .load_like(out)
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
            for block in (start / 32i32)..(position / 32i32 + 1i32) {
                let physical: i32 = tile_to_scalar(blocks.load([block]).reshape(shape![]));
                let key: Tile<f32, { [32, D] }> = convert_tile(
                    kp.load([pid.0 / GROUP, physical, 0i32])
                        .reshape(shape![32, D]),
                );
                let value: Tile<f32, { [32, D] }> = convert_tile(
                    vp.load([pid.0 / GROUP, physical, 0i32])
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
            out.store((numerator / denominator.broadcast(shape![D])).reshape(shape![1, D]));
        } else {
            out.store(0.0f32.broadcast(shape![1, D]));
        }
    }
}
