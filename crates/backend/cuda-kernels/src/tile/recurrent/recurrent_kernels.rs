#[cutile::module]
pub mod recurrent {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2,
        cmp_ordering, cmpf, convert_scalar, exp, ftz, get_tile_block_id, log, predicate,
        reduce_sum, rsqrt, select, tile_to_scalar,
    };

    #[cutile::entry()]
    fn conv4(
        out: &mut Tensor<f32, { [128] }>,
        h0: &mut Tensor<f32, { [128] }>,
        h1: &mut Tensor<f32, { [128] }>,
        h2: &mut Tensor<f32, { [128] }>,
        x: &Tensor<f32, { [-1] }>,
        w: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        if position >= 0 {
            let pid = get_tile_block_id();
            let x = x.load_like(out);
            let oldest = h0.load_like(out);
            let middle = h1.load_like(out);
            let newest = h2.load_like(out);
            let wp = w.partition(shape![128, 1]);
            let wa = wp.load([pid.0, 0i32]).reshape(shape![128]);
            let wb = wp.load([pid.0, 1i32]).reshape(shape![128]);
            let wc = wp.load([pid.0, 2i32]).reshape(shape![128]);
            let wd = wp.load([pid.0, 3i32]).reshape(shape![128]);
            let value = oldest * wa + middle * wb + newest * wc + x * wd;
            let negative: Tile<f32, { [128] }> = 0.0f32.broadcast(shape![128]) - value;
            out.store(value / (1.0f32.broadcast(shape![128]) + exp(negative)));
            h0.store(middle);
            h1.store(newest);
            h2.store(x);
        } else {
            out.store(0.0f32.broadcast(shape![128]));
        }
    }

    /// Lane-batched twin of [`delta`]: block `pid.0` is `lane * VH + value_head`, so one
    /// replay advances every busy lane instead of one launch per lane.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds explicit tensors and the padding metadata"
    )]
    #[cutile::entry()]
    fn delta_lanes<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        out: &mut Tensor<f32, { [1, 1, SW] }>,
        state: &mut Tensor<f32, { [1, D, SW] }>,
        qkv: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        alpha: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let lane = pid.0 / VH;
        let head = pid.0 % VH;
        let position: i32 = tile_to_scalar(
            metadata
                .partition(shape![1])
                .load([lane * 4 + 2])
                .reshape(shape![]),
        );
        if position >= 0 {
            let key = head / (VH / KH);
            let stride = 2 * KH + VH;
            let qp = qkv.partition(shape![D]);
            let mut q = qp.load([lane * stride + key]);
            let mut k = qp.load([lane * stride + KH + key]);
            let vp = qkv.partition(shape![SW]);
            let v = vp.load([(lane * stride + 2 * KH + head) * (D / SW) + pid.2]);
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
            let bt = beta.partition(shape![1]).load([lane * VH + head]);
            let bt = one / (one + exp(zero - bt));
            let at = alpha.partition(shape![1]).load([lane * VH + head])
                + bias.partition(shape![1]).load([head]);
            let soft = log(one + exp(at));
            let large = cmpf(
                at,
                20.0f32.broadcast(shape![1]),
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            );
            let soft = select(large, at, soft);
            let decay = exp(zero - exp(a_log.partition(shape![1]).load([head])) * soft);
            let old = state.load_like(state).reshape(shape![D, SW]);
            let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
            let kb = k.reshape(shape![D, 1]).broadcast(shape![D, SW]);
            let predicted: Tile<f32, { [SW] }> = reduce_sum(decayed * kb, 0i32);
            let predicted = predicted.reshape(shape![SW]);
            let diff = (v - predicted) * bt.broadcast(shape![SW]);
            let updated: Tile<f32, { [D, SW] }> =
                decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
            let qb = q.reshape(shape![D, 1]).broadcast(shape![D, SW]);
            let result: Tile<f32, { [SW] }> = reduce_sum(updated * qb, 0i32);
            state.store(updated.reshape(shape![1, D, SW]));
            out.store(result.reshape(shape![1, 1, SW]));
        } else {
            out.store(0.0f32.broadcast(shape![1, 1, SW]));
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds explicit tensors and the padding metadata"
    )]
    #[cutile::entry()]
    fn delta<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        out: &mut Tensor<f32, { [1, 1, SW] }>,
        state: &mut Tensor<f32, { [1, D, SW] }>,
        qkv: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        alpha: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let position: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([2i32]).reshape(shape![]));
        if position >= 0 {
            let pid = get_tile_block_id();
            let key = pid.0 / (VH / KH);
            let qp = qkv.partition(shape![D]);
            let mut q = qp.load([key]);
            let mut k = qp.load([KH + key]);
            // One block of the value dimension. The state's columns are mutually independent -
            // every reduction below is over the key dimension - so splitting them changes no
            // summation order while giving the grid SW times more blocks to fill the device.
            let vp = qkv.partition(shape![SW]);
            let v = vp.load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
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
            let bt = beta.partition(shape![1]).load([pid.0]);
            let bt = one / (one + exp(zero - bt));
            let at =
                alpha.partition(shape![1]).load([pid.0]) + bias.partition(shape![1]).load([pid.0]);
            let soft = log(one + exp(at));
            let large = cmpf(
                at,
                20.0f32.broadcast(shape![1]),
                predicate::GreaterThan,
                cmp_ordering::Ordered,
            );
            let soft = select(large, at, soft);
            let decay = exp(zero - exp(a_log.partition(shape![1]).load([pid.0])) * soft);
            let old = state.load_like(state).reshape(shape![D, SW]);
            let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
            let kb = k.reshape(shape![D, 1]).broadcast(shape![D, SW]);
            let predicted: Tile<f32, { [SW] }> = reduce_sum(decayed * kb, 0i32);
            let predicted = predicted.reshape(shape![SW]);
            let diff = (v - predicted) * bt.broadcast(shape![SW]);
            let updated: Tile<f32, { [D, SW] }> =
                decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
            let qb = q.reshape(shape![D, 1]).broadcast(shape![D, SW]);
            let result: Tile<f32, { [SW] }> = reduce_sum(updated * qb, 0i32);
            state.store(updated.reshape(shape![1, D, SW]));
            out.store(result.reshape(shape![1, 1, SW]));
        } else {
            out.store(0.0f32.broadcast(shape![1, 1, SW]));
        }
    }
}
