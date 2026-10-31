//! Token-chunk recurrent execution: keep the state in registers across prompt rows.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3,
        bf16, cmp_ordering, cmpf, constant, convert_scalar, convert_tile, eq_tile, exp, extract,
        ftz, ge_tile, get_tile_block_id, gt_tile, iota, log, lt_tile, mmaf, num_tiles, predicate,
        reduce_sum, reverse, rsqrt, scan_sum, select, tile_to_scalar,
    };

    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds explicit tensors and the padding metadata"
    )]
    #[cutile::entry()]
    fn delta<const KH: i32, const VH: i32, const D: i32, const LANES: i32, const SW: i32>(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        out: &mut Tensor<f32, { [1, LANES, SW] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
        beta: &Tensor<f32, { [-1, -1] }>,
        alpha: &Tensor<f32, { [-1, -1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        // One block of the value dimension. The state's columns are mutually independent - every
        // reduction below is over the key dimension - so the grid gains D / SW times more blocks
        // without changing any summation order.
        let mut old = state.load_like(state).reshape(shape![D, SW]);
        for lane in 0i32..LANES {
            if lane < count && base + lane + offset >= 0 {
                let key = pid.0 / (VH / KH);
                let qp = qkv.partition(shape![1, 1, D]);
                let mut q = qp.load([lane, key, 0i32]).reshape(shape![D]);
                let mut k = qp.load([lane, KH + key, 0i32]).reshape(shape![D]);
                let vp = qkv.partition(shape![1, 1, SW]);
                let v = vp
                    .load([lane, 2i32 * KH + pid.0, pid.2])
                    .reshape(shape![SW]);
                let qs: Tile<f32, { [] }> = reduce_sum(q * q, 0i32);
                let ks: Tile<f32, { [] }> = reduce_sum(k * k, 0i32);
                let qs = qs.reshape(shape![1]);
                let ks = ks.reshape(shape![1]);
                let qinv = rsqrt(qs + 1e-6f32.broadcast(shape![1]), ftz::Disabled);
                let kinv = rsqrt(ks + 1e-6f32.broadcast(shape![1]), ftz::Disabled);
                let d: f32 = convert_scalar(D);
                let dinv = rsqrt(d.broadcast(shape![1]), ftz::Disabled);
                q = q * (qinv * dinv).broadcast(shape![D]);
                k = k * kinv.broadcast(shape![D]);
                let one = 1.0f32.broadcast(shape![1]);
                let zero = 0.0f32.broadcast(shape![1]);
                let bt = beta
                    .partition(shape![1, 1])
                    .load([lane, pid.0])
                    .reshape(shape![1]);
                let bt = one / (one + exp(zero - bt));
                let at = alpha
                    .partition(shape![1, 1])
                    .load([lane, pid.0])
                    .reshape(shape![1])
                    + bias.partition(shape![1]).load([pid.0]);
                let soft = log(one + exp(at));
                let large = cmpf(
                    at,
                    20.0f32.broadcast(shape![1]),
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let soft = select(large, at, soft);
                let decay = exp(zero - exp(a_log.partition(shape![1]).load([pid.0])) * soft);
                let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
                let kb = k.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let predicted: Tile<f32, { [SW] }> = reduce_sum(decayed * kb, 0i32);
                let predicted = predicted.reshape(shape![SW]);
                let diff = (v - predicted) * bt.broadcast(shape![SW]);
                let updated: Tile<f32, { [D, SW] }> =
                    decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
                let qb = q.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let result: Tile<f32, { [SW] }> = reduce_sum(updated * qb, 0i32);
                old = updated;
                out.partition_mut(shape![1, 1, SW])
                    .store(result.reshape(shape![1, 1, SW]), [0i32, lane, 0i32]);
            } else {
                out.partition_mut(shape![1, 1, SW])
                    .store(0.0f32.broadcast(shape![1, 1, SW]), [0i32, lane, 0i32]);
            }
        }
        state.store(old.reshape(shape![1, D, SW]));
    }
    // Preserve the original scalar kernel's F32 addition tree: eight terms per stripe,
    // stripes combined as d + (c + (a + b)), then the 2/1 butterfly across key rows.
    // Prediction uses a different tree; sharing one helper changes the output last bits.
    fn scalar128_readout_sum<const SW: i32>(
        products: Tile<f32, { [128, SW] }>,
    ) -> Tile<f32, { [SW] }> {
        let groups = products.reshape(shape![8, 16, SW]);
        let zero = 0i32.broadcast(shape![]);
        let slab0: Tile<f32, { [1, 16, SW] }> = extract(groups, [zero, zero, zero]);
        let slab1: Tile<f32, { [1, 16, SW] }> =
            extract(groups, [1i32.broadcast(shape![]), zero, zero]);
        let slab2: Tile<f32, { [1, 16, SW] }> =
            extract(groups, [2i32.broadcast(shape![]), zero, zero]);
        let slab3: Tile<f32, { [1, 16, SW] }> =
            extract(groups, [3i32.broadcast(shape![]), zero, zero]);
        let slab4: Tile<f32, { [1, 16, SW] }> =
            extract(groups, [4i32.broadcast(shape![]), zero, zero]);
        let slab5: Tile<f32, { [1, 16, SW] }> =
            extract(groups, [5i32.broadcast(shape![]), zero, zero]);
        let slab6: Tile<f32, { [1, 16, SW] }> =
            extract(groups, [6i32.broadcast(shape![]), zero, zero]);
        let slab7: Tile<f32, { [1, 16, SW] }> =
            extract(groups, [7i32.broadcast(shape![]), zero, zero]);
        let partial = ((((((((0.0f32.broadcast(shape![1, 16, SW]) + slab0) + slab1) + slab2)
            + slab3)
            + slab4)
            + slab5)
            + slab6)
            + slab7)
            .reshape(shape![4, 4, SW]);
        let stripe0: Tile<f32, { [1, 4, SW] }> = extract(partial, [zero, zero, zero]);
        let stripe1: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [1i32.broadcast(shape![]), zero, zero]);
        let stripe2: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [2i32.broadcast(shape![]), zero, zero]);
        let stripe3: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [3i32.broadcast(shape![]), zero, zero]);
        let partial = (stripe3 + (stripe2 + (stripe0 + stripe1))).reshape(shape![4, SW]);
        let worker0: Tile<f32, { [1, SW] }> = extract(partial, [zero, zero]);
        let worker1: Tile<f32, { [1, SW] }> = extract(partial, [1i32.broadcast(shape![]), zero]);
        let worker2: Tile<f32, { [1, SW] }> = extract(partial, [2i32.broadcast(shape![]), zero]);
        let worker3: Tile<f32, { [1, SW] }> = extract(partial, [3i32.broadcast(shape![]), zero]);
        ((worker0 + worker2) + (worker1 + worker3)).reshape(shape![SW])
    }

    // Fold four 32-row slabs before reducing. The scalar kernel does these additions
    // before its eight-term worker sum; a flat 128-row reduction changes recurrent state.
    fn scalar128_prediction_sum<const SW: i32>(
        products: Tile<f32, { [128, SW] }>,
    ) -> Tile<f32, { [SW] }> {
        let groups = products.reshape(shape![4, 32, SW]);
        let zero = 0i32.broadcast(shape![]);
        let slab0: Tile<f32, { [1, 32, SW] }> = extract(groups, [zero, zero, zero]);
        let slab1: Tile<f32, { [1, 32, SW] }> =
            extract(groups, [1i32.broadcast(shape![]), zero, zero]);
        let slab2: Tile<f32, { [1, 32, SW] }> =
            extract(groups, [2i32.broadcast(shape![]), zero, zero]);
        let slab3: Tile<f32, { [1, 32, SW] }> =
            extract(groups, [3i32.broadcast(shape![]), zero, zero]);
        let partial = (((0.0f32.broadcast(shape![1, 32, SW]) + slab0) + slab1) + slab2) + slab3;
        reduce_sum(partial.reshape(shape![32, SW]), 0i32)
    }

    // Fold four 32-row slabs, then combine the eight worker terms in the scalar tree.
    // The tree is explicit so changing the value tile cannot reassociate the sum.
    // The scalar kernel does these additions
    // before its eight-term worker sum; a flat 128-row reduction changes recurrent state.
    fn scalar128_prediction_split_sum<const SW: i32>(
        products: Tile<f32, { [128, SW] }>,
    ) -> Tile<f32, { [SW] }> {
        let groups = products.reshape(shape![4, 32, SW]);
        let zero = 0i32.broadcast(shape![]);
        let slab0: Tile<f32, { [1, 32, SW] }> = extract(groups, [zero, zero, zero]);
        let slab1: Tile<f32, { [1, 32, SW] }> =
            extract(groups, [1i32.broadcast(shape![]), zero, zero]);
        let slab2: Tile<f32, { [1, 32, SW] }> =
            extract(groups, [2i32.broadcast(shape![]), zero, zero]);
        let slab3: Tile<f32, { [1, 32, SW] }> =
            extract(groups, [3i32.broadcast(shape![]), zero, zero]);
        let partial = (((0.0f32.broadcast(shape![1, 32, SW]) + slab0) + slab1) + slab2) + slab3;
        let partial = partial.reshape(shape![8, 4, SW]);
        let term0: Tile<f32, { [1, 4, SW] }> = extract(partial, [zero, zero, zero]);
        let term1: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [1i32.broadcast(shape![]), zero, zero]);
        let term2: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [2i32.broadcast(shape![]), zero, zero]);
        let term3: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [3i32.broadcast(shape![]), zero, zero]);
        let term4: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [4i32.broadcast(shape![]), zero, zero]);
        let term5: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [5i32.broadcast(shape![]), zero, zero]);
        let term6: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [6i32.broadcast(shape![]), zero, zero]);
        let term7: Tile<f32, { [1, 4, SW] }> =
            extract(partial, [7i32.broadcast(shape![]), zero, zero]);
        let partial = (term7 + (term6 + (term5 + (term4 + (term3 + (term2 + (term0 + term1)))))))
            .reshape(shape![4, SW]);
        let worker0: Tile<f32, { [1, SW] }> = extract(partial, [zero, zero]);
        let worker1: Tile<f32, { [1, SW] }> = extract(partial, [1i32.broadcast(shape![]), zero]);
        let worker2: Tile<f32, { [1, SW] }> = extract(partial, [2i32.broadcast(shape![]), zero]);
        let worker3: Tile<f32, { [1, SW] }> = extract(partial, [3i32.broadcast(shape![]), zero]);
        ((worker0 + worker2) + (worker1 + worker3)).reshape(shape![SW])
    }

    // Keep the scalar Q/K norm tree when value columns are split among more blocks.
    fn scalar128_norm_sum(products: Tile<f32, { [128] }>) -> Tile<f32, { [] }> {
        let terms = products.reshape(shape![32, 4]);
        let zero = 0i32.broadcast(shape![]);
        let partial = 0.0f32.broadcast(shape![1, 4]);
        let row: Tile<f32, { [1, 4] }> = extract(terms, [zero, zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [1i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [2i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [3i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [4i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [5i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [6i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [7i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [8i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [9i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [10i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [11i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [12i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [13i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [14i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [15i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [16i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [17i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [18i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [19i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [20i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [21i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [22i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [23i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [24i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [25i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [26i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [27i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [28i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [29i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [30i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let row: Tile<f32, { [1, 4] }> = extract(terms, [31i32.broadcast(shape![]), zero]);
        let partial = partial + row;
        let partial = partial.reshape(shape![4]);
        let worker0: Tile<f32, { [1] }> = extract(partial, [zero]);
        let worker1: Tile<f32, { [1] }> = extract(partial, [1i32.broadcast(shape![])]);
        let worker2: Tile<f32, { [1] }> = extract(partial, [2i32.broadcast(shape![])]);
        let worker3: Tile<f32, { [1] }> = extract(partial, [3i32.broadcast(shape![])]);
        ((worker0 + worker2) + (worker1 + worker3)).reshape(shape![])
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds explicit tensors and the padding metadata"
    )]
    #[cutile::entry()]
    // D = 128 at captured 64/256-token widths. Split columns use explicit scalar trees.
    // Keep the state across tokens while preserving both scalar reduction trees.
    fn delta_grouped<
        const KH: i32,
        const VH: i32,
        const D: i32,
        const LANES: i32,
        const SW: i32,
    >(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        out: &mut Tensor<f32, { [1, LANES, SW] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
        beta: &Tensor<f32, { [-1, -1] }>,
        alpha: &Tensor<f32, { [-1, -1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        // One block of the value dimension. The state's columns are mutually independent - every
        // reduction below is over the key dimension - so the grid gains D / SW times more blocks
        // without changing any summation order.
        let mut old = state.load_like(state).reshape(shape![D, SW]);
        for lane in 0i32..LANES {
            if lane < count && base + lane + offset >= 0 {
                let key = pid.0 / (VH / KH);
                let qp = qkv.partition(shape![1, 1, D]);
                let mut q = qp.load([lane, key, 0i32]).reshape(shape![D]);
                let mut k = qp.load([lane, KH + key, 0i32]).reshape(shape![D]);
                let vp = qkv.partition(shape![1, 1, SW]);
                let v = vp
                    .load([lane, 2i32 * KH + pid.0, pid.2])
                    .reshape(shape![SW]);
                let qs: Tile<f32, { [] }> = if SW == 128 {
                    reduce_sum(q * q, 0i32)
                } else {
                    scalar128_norm_sum((q * q).reshape(shape![128]))
                };
                let ks: Tile<f32, { [] }> = if SW == 128 {
                    reduce_sum(k * k, 0i32)
                } else {
                    scalar128_norm_sum((k * k).reshape(shape![128]))
                };
                let qs = qs.reshape(shape![1]);
                let ks = ks.reshape(shape![1]);
                let qinv = rsqrt(qs + 1e-6f32.broadcast(shape![1]), ftz::Disabled);
                let kinv = rsqrt(ks + 1e-6f32.broadcast(shape![1]), ftz::Disabled);
                let d: f32 = convert_scalar(D);
                let dinv = rsqrt(d.broadcast(shape![1]), ftz::Disabled);
                q = q * (qinv * dinv).broadcast(shape![D]);
                k = k * kinv.broadcast(shape![D]);
                let one = 1.0f32.broadcast(shape![1]);
                let zero = 0.0f32.broadcast(shape![1]);
                let bt = beta
                    .partition(shape![1, 1])
                    .load([lane, pid.0])
                    .reshape(shape![1]);
                let bt = one / (one + exp(zero - bt));
                let at = alpha
                    .partition(shape![1, 1])
                    .load([lane, pid.0])
                    .reshape(shape![1])
                    + bias.partition(shape![1]).load([pid.0]);
                let soft = log(one + exp(at));
                let large = cmpf(
                    at,
                    20.0f32.broadcast(shape![1]),
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                );
                let soft = select(large, at, soft);
                let decay = exp(zero - exp(a_log.partition(shape![1]).load([pid.0])) * soft);
                let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
                let kb = k.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let predicted: Tile<f32, { [SW] }> = if SW == 128 {
                    scalar128_prediction_sum((decayed * kb).reshape(shape![128, SW]))
                } else {
                    scalar128_prediction_split_sum((decayed * kb).reshape(shape![128, SW]))
                };
                let predicted = predicted.reshape(shape![SW]);
                let diff = (v - predicted) * bt.broadcast(shape![SW]);
                let updated: Tile<f32, { [D, SW] }> =
                    decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
                let qb = q.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let result: Tile<f32, { [SW] }> =
                    scalar128_readout_sum((updated * qb).reshape(shape![128, SW]));
                old = updated;
                out.partition_mut(shape![1, 1, SW])
                    .store(result.reshape(shape![1, 1, SW]), [0i32, lane, 0i32]);
            } else {
                out.partition_mut(shape![1, 1, SW])
                    .store(0.0f32.broadcast(shape![1, 1, SW]), [0i32, lane, 0i32]);
            }
        }
        state.store(old.reshape(shape![1, D, SW]));
    }
    // One F32 normalization per Q/K head and token, preserving the scalar norm tree.
    #[cutile::entry()]
    fn normalize_shared<const KH: i32, const D: i32>(
        normalized: &mut Tensor<f32, { [1, 1, D] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
    ) {
        let pid = get_tile_block_id();
        let x = qkv
            .partition(shape![1, 1, D])
            .load([pid.0, pid.1, 0i32])
            .reshape(shape![D]);
        let square = scalar128_norm_sum((x * x).reshape(shape![128])).reshape(shape![1]);
        let inverse = rsqrt(square + 1e-6f32.broadcast(shape![1]), ftz::Disabled);
        let factor = if pid.1 < KH {
            let d: f32 = convert_scalar(D);
            inverse * rsqrt(d.broadcast(shape![1]), ftz::Disabled)
        } else {
            inverse
        };
        let result = x * factor.broadcast(shape![D]);
        normalized.store(result.reshape(shape![1, 1, D]));
    }

    // Beta and decay are independent of the state columns; each is evaluated once.
    #[cutile::entry()]
    fn gates_shared(
        betas: &mut Tensor<f32, { [1, 1] }>,
        decays: &mut Tensor<f32, { [1, 1] }>,
        beta: &Tensor<f32, { [-1, -1] }>,
        alpha: &Tensor<f32, { [-1, -1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let one = 1.0f32.broadcast(shape![1]);
        let zero = 0.0f32.broadcast(shape![1]);
        let bt = beta
            .partition(shape![1, 1])
            .load([pid.0, pid.1])
            .reshape(shape![1]);
        let bt = one / (one + exp(zero - bt));
        let at = alpha
            .partition(shape![1, 1])
            .load([pid.0, pid.1])
            .reshape(shape![1])
            + bias.partition(shape![1]).load([pid.1]);
        let soft = log(one + exp(at));
        let large = cmpf(
            at,
            20.0f32.broadcast(shape![1]),
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let soft = select(large, at, soft);
        let decay = exp(zero - exp(a_log.partition(shape![1]).load([pid.1])) * soft);
        betas.store(bt.reshape(shape![1, 1]));
        decays.store(decay.reshape(shape![1, 1]));
    }

    #[cutile::entry()]
    // D = 128 at captured 64/256-token widths. Split columns use explicit scalar trees.
    // Keep the state across tokens while preserving both scalar reduction trees.
    fn delta_precomputed<
        const KH: i32,
        const VH: i32,
        const D: i32,
        const LANES: i32,
        const SW: i32,
    >(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        out: &mut Tensor<f32, { [1, LANES, SW] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
        normalized: &Tensor<f32, { [-1, -1, D] }>,
        betas: &Tensor<f32, { [-1, -1] }>,
        decays: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        // One block of the value dimension. The state's columns are mutually independent - every
        // reduction below is over the key dimension - so the grid gains D / SW times more blocks
        // without changing any summation order.
        // Prepared inputs have LANES padded rows. Clear padding once, then keep
        // checked loads in a straight-line active loop so bounds checks can move
        // to its preheader without changing the F32 operation tree.
        let qp = normalized.partition(shape![1, 1, D]);
        let vp = qkv.partition(shape![1, 1, SW]);
        let bp = betas.partition(shape![1, 1]);
        let dp = decays.partition(shape![1, 1]);
        let key = pid.0 / (VH / KH);
        let key_value = KH + key;
        let value_head = 2i32 * KH + pid.0;
        let rows = num_tiles(&qp, 0i32);
        let limit = if count < rows { count } else { rows };
        let displacement = base + offset;
        let first = if displacement < 0i32 {
            0i32 - displacement
        } else {
            0i32
        };
        let mut old = state.load_like(state).reshape(shape![D, SW]);
        out.store(0.0f32.broadcast(shape![1, LANES, SW]));
        if first < limit {
            for lane in first..limit {
                let q = qp.load([lane, key, 0i32]).reshape(shape![D]);
                let k = qp.load([lane, key_value, 0i32]).reshape(shape![D]);
                let v = vp.load([lane, value_head, pid.2]).reshape(shape![SW]);
                let bt = bp.load([lane, pid.0]).reshape(shape![1]);
                let decay = dp.load([lane, pid.0]).reshape(shape![1]);
                let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
                let kb = k.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let predicted: Tile<f32, { [SW] }> = if SW == 128 {
                    scalar128_prediction_sum((decayed * kb).reshape(shape![128, SW]))
                } else {
                    scalar128_prediction_split_sum((decayed * kb).reshape(shape![128, SW]))
                };
                let predicted = predicted.reshape(shape![SW]);
                let diff = (v - predicted) * bt.broadcast(shape![SW]);
                let updated: Tile<f32, { [D, SW] }> =
                    decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
                let qb = q.reshape(shape![D, 1]).broadcast(shape![D, SW]);
                let result: Tile<f32, { [SW] }> =
                    scalar128_readout_sum((updated * qb).reshape(shape![128, SW]));
                old = updated;
                out.partition_mut(shape![1, 1, SW])
                    .store(result.reshape(shape![1, 1, SW]), [0i32, lane, 0i32]);
            }
        }
        state.store(old.reshape(shape![1, D, SW]));
    }
    // Experimental WY factorization. Each chunk solves the causal token interaction system
    // before updating the recurrent state with matrix products. F32 operands use six BF16
    // component products with F32 accumulation.
    fn precise_mma<const M: i32, const K: i32, const N: i32>(
        left: Tile<f32, { [M, K] }>,
        right: Tile<f32, { [K, N] }>,
    ) -> Tile<f32, { [M, N] }> {
        let lh: Tile<bf16, { [M, K] }> = convert_tile(left);
        let rh: Tile<bf16, { [K, N] }> = convert_tile(right);
        let lr: Tile<f32, { [M, K] }> = convert_tile(lh);
        let rr: Tile<f32, { [K, N] }> = convert_tile(rh);
        let ll: Tile<bf16, { [M, K] }> = convert_tile(left - lr);
        let rl: Tile<bf16, { [K, N] }> = convert_tile(right - rr);
        // A third BF16 component retains the remaining F32 significand bits. Include
        // all products through second residual order; three products omit ll*rl and
        // both third components, giving roughly 16-bit operand accuracy.
        let lf: Tile<f32, { [M, K] }> = convert_tile(ll);
        let rf: Tile<f32, { [K, N] }> = convert_tile(rl);
        let lll: Tile<bf16, { [M, K] }> = convert_tile((left - lr) - lf);
        let rll: Tile<bf16, { [K, N] }> = convert_tile((right - rr) - rf);
        let zero: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        let low = mmaf(lll, rh, zero);
        let low = mmaf(lh, rll, low);
        let low = mmaf(ll, rl, low);
        let low = mmaf(ll, rh, low);
        let low = mmaf(lh, rl, low);
        mmaf(lh, rh, low)
    }

    fn triangular_inverse(
        interactions: Tile<f32, { [32, 32] }>,
        row_indices: Tile<i32, { [32, 32] }>,
        col_indices: Tile<i32, { [32, 32] }>,
        lower: Tile<bool, { [32, 32] }>,
    ) -> Tile<f32, { [32, 32] }> {
        let identity = select(
            eq_tile(row_indices, col_indices),
            1.0f32.broadcast(shape![32, 32]),
            0.0f32.broadcast(shape![32, 32]),
        );
        // Invert 2x2 diagonal blocks first, then merge adjacent inverted blocks:
        // [[L11,0],[L21,L22]]^-1 has cross block -inv(L22)*L21*inv(L11).
        // Squaring the entire strictly-lower matrix creates huge binomial terms for
        // correlated keys with weak decay and then subtracts them, losing precision.
        let two = 2i32.broadcast(shape![32, 32]);
        let pair = eq_tile(row_indices / two, col_indices / two) & lower;
        let mut inverse = identity - select(pair, interactions, 0.0f32.broadcast(shape![32, 32]));
        let mut half = 2i32;
        for _level in 0i32..4i32 {
            let width = (2i32 * half).broadcast(shape![32, 32]);
            let row_block = row_indices / width;
            let col_block = col_indices / width;
            let middle = row_block * width + half.broadcast(shape![32, 32]);
            let cross = eq_tile(row_block, col_block)
                & ge_tile(row_indices, middle)
                & lt_tile(col_indices, middle);
            let coupling = select(cross, interactions, 0.0f32.broadcast(shape![32, 32]));
            inverse = inverse - precise_mma(precise_mma(inverse, coupling), inverse);
            half = half * 2i32;
        }
        inverse
    }

    fn cumulative_decay(
        alpha: &Tensor<f32, { [-1, -1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        active: Tile<bool, { [32] }>,
        chunk: i32,
        head: i32,
    ) -> Tile<f32, { [32] }> {
        let a = alpha
            .partition(shape![32, 1])
            .load([chunk, head])
            .reshape(shape![32])
            + bias.partition(shape![1]).load([head]).broadcast(shape![32]);
        let soft = log(1.0f32.broadcast(shape![32]) + exp(a));
        let soft = select(gt_tile(a, 20.0f32.broadcast(shape![32])), a, soft);
        let g = 0.0f32.broadcast(shape![32])
            - exp(a_log.partition(shape![1]).load([head])).broadcast(shape![32]) * soft;
        let g = select(active, g, 0.0f32.broadcast(shape![32]));
        let g = scan_sum(g, 0i32, reverse::Forward, 0.0f32);
        g
    }

    #[expect(clippy::too_many_arguments, reason = "explicit CUDA tensor bindings")]
    #[expect(
        clippy::many_single_char_names,
        reason = "WY symbols match the validated matrix derivation"
    )]
    #[cutile::entry()]
    fn delta_wy<const KH: i32, const VH: i32, const D: i32, const LANES: i32, const SW: i32>(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        out: &mut Tensor<f32, { [1, LANES, SW] }>,
        qkv: &Tensor<f32, { [-1, -1, D] }>,
        beta: &Tensor<f32, { [-1, -1] }>,
        alpha: &Tensor<f32, { [-1, -1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let mut old = state.load_like(state).reshape(shape![D, SW]);
        let indices: Tile<i32, { [32] }> = iota(shape![32]);
        let row_indices = indices.reshape(shape![32, 1]).broadcast(shape![32, 32]);
        let col_indices = indices.reshape(shape![1, 32]).broadcast(shape![32, 32]);
        let lower = gt_tile(row_indices, col_indices);
        let causal = ge_tile(row_indices, col_indices);
        for chunk in 0i32..((LANES + 31i32) / 32i32) {
            let rows = indices + (chunk * 32i32).broadcast(shape![32]);
            let active = lt_tile(rows, count.broadcast(shape![32]))
                & ge_tile(
                    rows + (base + offset).broadcast(shape![32]),
                    0i32.broadcast(shape![32]),
                );
            let valid = active.reshape(shape![32, 1]).broadcast(shape![32, D]);
            let key = pid.0 / (VH / KH);
            let qp = qkv.partition(shape![32, 1, D]);
            let q = qp.load([chunk, key, 0i32]).reshape(shape![32, D]);
            let k = qp.load([chunk, KH + key, 0i32]).reshape(shape![32, D]);
            let qnorm: Tile<f32, { [32] }> = reduce_sum(q * q, 1i32);
            let knorm: Tile<f32, { [32] }> = reduce_sum(k * k, 1i32);
            let qinv = rsqrt(qnorm + 1e-6f32.broadcast(shape![32]), ftz::Disabled);
            let kinv = rsqrt(knorm + 1e-6f32.broadcast(shape![32]), ftz::Disabled);
            let d: f32 = convert_scalar(D);
            let dinv = rsqrt(d.broadcast(shape![32]), ftz::Disabled);
            let q = select(
                valid,
                q * (qinv * dinv)
                    .reshape(shape![32, 1])
                    .broadcast(shape![32, D]),
                0.0f32.broadcast(shape![32, D]),
            );
            let k = select(
                valid,
                k * kinv.reshape(shape![32, 1]).broadcast(shape![32, D]),
                0.0f32.broadcast(shape![32, D]),
            );
            let v = qkv
                .partition(shape![32, 1, SW])
                .load([chunk, 2i32 * KH + pid.0, pid.2])
                .reshape(shape![32, SW]);
            let b = beta
                .partition(shape![32, 1])
                .load([chunk, pid.0])
                .reshape(shape![32]);
            let b = 1.0f32.broadcast(shape![32])
                / (1.0f32.broadcast(shape![32]) + exp(0.0f32.broadcast(shape![32]) - b));
            let b = select(active, b, 0.0f32.broadcast(shape![32]));
            let g = cumulative_decay(alpha, a_log, bias, active, chunk, pid.0);
            let decay = exp(g);
            let gap = g.reshape(shape![32, 1]).broadcast(shape![32, 32])
                - g.reshape(shape![1, 32]).broadcast(shape![32, 32]);
            // Mask before exponentiation: upper-triangle gaps can overflow even though the
            // causal coefficients themselves are bounded by one.
            let relative = exp(select(causal, gap, 0.0f32.broadcast(shape![32, 32])));
            let kk = precise_mma(k, k.transpose());
            let interactions = select(
                lower,
                kk * relative * b.reshape(shape![32, 1]).broadcast(shape![32, 32]),
                0.0f32.broadcast(shape![32, 32]),
            );
            let projected =
                precise_mma(k, old) * decay.reshape(shape![32, 1]).broadcast(shape![32, SW]);
            let rhs = (v - projected) * b.reshape(shape![32, 1]).broadcast(shape![32, SW]);
            // Solve (I + A) * updates = rhs, preserving unit diagonal structure.
            let inverse = triangular_inverse(interactions, row_indices, col_indices, lower);
            let updates = precise_mma(inverse, rhs);
            let qk = precise_mma(q, k.transpose());
            let probabilities = select(causal, qk * relative, 0.0f32.broadcast(shape![32, 32]));
            let output = precise_mma(q, old)
                * decay.reshape(shape![32, 1]).broadcast(shape![32, SW])
                + precise_mma(probabilities, updates);
            let output = select(
                active.reshape(shape![32, 1]).broadcast(shape![32, SW]),
                output,
                0.0f32.broadcast(shape![32, SW]),
            );
            out.partition_mut(shape![1, 32, SW])
                .store(output.reshape(shape![1, 32, SW]), [0i32, chunk, 0i32]);
            let last: Tile<f32, { [1] }> = extract(g, [31i32.broadcast(shape![])]);
            let weights = exp(last.broadcast(shape![32]) - g);
            old = old * exp(last).reshape(shape![1, 1]).broadcast(shape![D, SW])
                + precise_mma(
                    k.transpose(),
                    updates * weights.reshape(shape![32, 1]).broadcast(shape![32, SW]),
                );
        }
        state.store(old.reshape(shape![1, D, SW]));
    }
    #[cutile::entry()]
    fn transpose<const D: i32>(
        out: &mut Tensor<f32, { [1, 1, D] }>,
        input: &Tensor<f32, { [-1, -1, D] }>,
    ) {
        let pid = get_tile_block_id();
        out.store(input.partition(shape![1, 1, D]).load([pid.1, pid.0, 0i32]));
    }
}
