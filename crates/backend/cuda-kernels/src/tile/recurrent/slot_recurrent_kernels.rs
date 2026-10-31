//! Snapshot-free recurrent verification for the pooled speculation graph.
//!
//! The verification lanes of one slot are scored in one launch per recurrent node that reads the
//! slot's state and leaves it untouched; once acceptance is known, one commit launch applies the
//! accepted prefix. Rolling back therefore needs no per-lane state snapshot, which was two full
//! recurrent state copies (about 300 MB on the 27B) per slot.
#[cutile::module]
pub mod fused {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tensor_3, Tile_0, Tile_1, Tile_2, Tile_3,
        cmp_ordering, cmpf, convert_scalar, exp, ftz, ge_tile, get_tile_block_id, log, predicate,
        reduce_sum, rsqrt, select, tile_to_scalar,
    };

    /// Whether a lane's metadata marks a real token rather than padding.
    fn lane_active(metadata: &Tensor<i32, { [-1] }>) -> Tile<bool, { [1] }> {
        let position = metadata.partition(shape![1]).load([2i32]);
        ge_tile(position, 0i32.broadcast(shape![1]))
    }

    fn batch_active<const MF: i32>(
        metadata: &Tensor<i32, { [-1] }>,
        row: i32,
    ) -> Tile<bool, { [1] }> {
        let position = metadata.partition(shape![1]).load([row * MF + 2i32]);
        ge_tile(position, 0i32.broadcast(shape![1]))
    }

    /// Short-convolution outputs of one slot's three verification lanes, each over the window
    /// its predecessors would leave, without shifting the state: [`conv_commit3`] shifts it by
    /// the accepted prefix. Matches `conv4`'s arithmetic; a padding lane outputs zero.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[cutile::entry()]
    fn conv_verify3(
        out0: &mut Tensor<f32, { [128] }>,
        out1: &mut Tensor<f32, { [128] }>,
        out2: &mut Tensor<f32, { [128] }>,
        h0: &Tensor<f32, { [-1] }>,
        h1: &Tensor<f32, { [-1] }>,
        h2: &Tensor<f32, { [-1] }>,
        x0: &Tensor<f32, { [-1] }>,
        x1: &Tensor<f32, { [-1] }>,
        x2: &Tensor<f32, { [-1] }>,
        w: &Tensor<f32, { [-1, 4] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let oldest: Tile<f32, { [128] }> = h0.partition(shape![128]).load([pid.0]);
        let middle: Tile<f32, { [128] }> = h1.partition(shape![128]).load([pid.0]);
        let newest: Tile<f32, { [128] }> = h2.partition(shape![128]).load([pid.0]);
        let first: Tile<f32, { [128] }> = x0.partition(shape![128]).load([pid.0]);
        let second: Tile<f32, { [128] }> = x1.partition(shape![128]).load([pid.0]);
        let third: Tile<f32, { [128] }> = x2.partition(shape![128]).load([pid.0]);
        let wp = w.partition(shape![128, 1]);
        let wa = wp.load([pid.0, 0i32]).reshape(shape![128]);
        let wb = wp.load([pid.0, 1i32]).reshape(shape![128]);
        let wc = wp.load([pid.0, 2i32]).reshape(shape![128]);
        let wd = wp.load([pid.0, 3i32]).reshape(shape![128]);
        let one: Tile<f32, { [128] }> = 1.0f32.broadcast(shape![128]);
        let zero: Tile<f32, { [128] }> = 0.0f32.broadcast(shape![128]);
        let value0 = oldest * wa + middle * wb + newest * wc + first * wd;
        let value1 = middle * wa + newest * wb + first * wc + second * wd;
        let value2 = newest * wa + first * wb + second * wc + third * wd;
        let silu0 = value0 / (one + exp(zero - value0));
        let silu1 = value1 / (one + exp(zero - value1));
        let silu2 = value2 / (one + exp(zero - value2));
        out0.store(select(
            lane_active(metadata0).broadcast(shape![128]),
            silu0,
            zero,
        ));
        out1.store(select(
            lane_active(metadata1).broadcast(shape![128]),
            silu1,
            zero,
        ));
        out2.store(select(
            lane_active(metadata2).broadcast(shape![128]),
            silu2,
            zero,
        ));
    }

    /// Shift one slot's convolution window by its first `count` verification lanes' inputs; a
    /// padding lane is never applied.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[cutile::entry()]
    fn conv_commit3(
        h0: &mut Tensor<f32, { [128] }>,
        h1: &mut Tensor<f32, { [128] }>,
        h2: &mut Tensor<f32, { [128] }>,
        x0: &Tensor<f32, { [-1] }>,
        x1: &Tensor<f32, { [-1] }>,
        x2: &Tensor<f32, { [-1] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
        count: i32,
    ) {
        let pid = get_tile_block_id();
        let oldest: Tile<f32, { [128] }> = h0.load_like(h0);
        let middle: Tile<f32, { [128] }> = h1.load_like(h1);
        let newest: Tile<f32, { [128] }> = h2.load_like(h2);
        let first: Tile<f32, { [128] }> = x0.partition(shape![128]).load([pid.0]);
        let second: Tile<f32, { [128] }> = x1.partition(shape![128]).load([pid.0]);
        let third: Tile<f32, { [128] }> = x2.partition(shape![128]).load([pid.0]);
        let accepted = count.broadcast(shape![1]);
        let one: Tile<bool, { [1] }> =
            lane_active(metadata0) & ge_tile(accepted, 1i32.broadcast(shape![1]));
        let two: Tile<bool, { [1] }> =
            one & lane_active(metadata1) & ge_tile(accepted, 2i32.broadcast(shape![1]));
        let three: Tile<bool, { [1] }> =
            two & lane_active(metadata2) & ge_tile(accepted, 3i32.broadcast(shape![1]));
        let one = one.broadcast(shape![128]);
        let two = two.broadcast(shape![128]);
        let three = three.broadcast(shape![128]);
        // The window after n applied lanes is [h0, h1, h2, x0, x1, x2][n..n + 3].
        h0.store(select(
            three,
            first,
            select(two, newest, select(one, middle, oldest)),
        ));
        h1.store(select(
            three,
            second,
            select(two, first, select(one, newest, middle)),
        ));
        h2.store(select(
            three,
            third,
            select(two, second, select(one, first, newest)),
        ));
    }

    /// Normalize a query or key with the reference delta kernel's reduction order.
    fn normalize<const D: i32>(value: Tile<f32, { [D] }>) -> Tile<f32, { [D] }> {
        // Materialize the specialized shape at the helper boundary for cuTile's type checker.
        let value = value.reshape(shape![D]);
        let square: Tile<f32, { [] }> = reduce_sum(value * value, 0i32);
        let inverse = rsqrt(
            square.reshape(shape![1]) + 1e-6f32.broadcast(shape![1]),
            ftz::Disabled,
        );
        // Resolve arithmetic before cuTile applies the helper's generic return type.
        let normalized = value * inverse.broadcast(shape![D]);
        normalized.reshape(shape![D])
    }

    fn sigmoid(value: Tile<f32, { [1] }>) -> Tile<f32, { [1] }> {
        let one = 1.0f32.broadcast(shape![1]);
        one / (one + exp(0.0f32.broadcast(shape![1]) - value))
    }

    fn delta_decay(alpha: Tile<f32, { [1] }>, rate: Tile<f32, { [1] }>) -> Tile<f32, { [1] }> {
        let one = 1.0f32.broadcast(shape![1]);
        let large = cmpf(
            alpha,
            20.0f32.broadcast(shape![1]),
            predicate::GreaterThan,
            cmp_ordering::Ordered,
        );
        let soft = select(large, alpha, log(one + exp(alpha)));
        exp(0.0f32.broadcast(shape![1]) - rate * soft)
    }

    /// One lane's state transition, shared by verification and accepted-prefix commit.
    fn delta_update<const D: i32, const SW: i32>(
        old: Tile<f32, { [D, SW] }>,
        key: Tile<f32, { [D] }>,
        value: Tile<f32, { [SW] }>,
        beta: Tile<f32, { [1] }>,
        decay: Tile<f32, { [1] }>,
        active: Tile<bool, { [1] }>,
    ) -> Tile<f32, { [D, SW] }> {
        let old = old.reshape(shape![D, SW]);
        let key = normalize(key.reshape(shape![D]));
        let value = value.reshape(shape![SW]);
        let decayed = old * decay.reshape(shape![1, 1]).broadcast(shape![D, SW]);
        let kb = key.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let predicted: Tile<f32, { [SW] }> = reduce_sum(decayed * kb, 0i32);
        let diff = (value - predicted.reshape(shape![SW])) * beta.broadcast(shape![SW]);
        let next: Tile<f32, { [D, SW] }> =
            decayed + kb * diff.reshape(shape![1, SW]).broadcast(shape![D, SW]);
        select(
            active.reshape(shape![1, 1]).broadcast(shape![D, SW]),
            next,
            old,
        )
    }

    fn delta_output<const D: i32, const SW: i32>(
        state: Tile<f32, { [D, SW] }>,
        query: Tile<f32, { [D] }>,
        active: Tile<bool, { [1] }>,
    ) -> Tile<f32, { [1, 1, SW] }> {
        let state = state.reshape(shape![D, SW]);
        let query = query.reshape(shape![D]);
        let square: Tile<f32, { [] }> = reduce_sum(query * query, 0i32);
        let inverse = rsqrt(
            square.reshape(shape![1]) + 1e-6f32.broadcast(shape![1]),
            ftz::Disabled,
        );
        let d: f32 = convert_scalar(D);
        let dinv = rsqrt(d.broadcast(shape![1]), ftz::Disabled);
        let query = query * (inverse * dinv).broadcast(shape![D]);
        let qb = query.reshape(shape![D, 1]).broadcast(shape![D, SW]);
        let result: Tile<f32, { [SW] }> = reduce_sum(state * qb, 0i32);
        select(
            active.reshape(shape![1, 1, 1]).broadcast(shape![1, 1, SW]),
            result.reshape(shape![1, 1, SW]),
            0.0f32.broadcast(shape![1, 1, SW]),
        )
    }

    /// Score three lanes without changing the slot state; commit applies the accepted prefix.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[cutile::entry()]
    fn delta_verify3<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        out0: &mut Tensor<f32, { [1, 1, SW] }>,
        out1: &mut Tensor<f32, { [1, 1, SW] }>,
        out2: &mut Tensor<f32, { [1, 1, SW] }>,
        state: &Tensor<f32, { [-1, D, -1] }>,
        qkv0: &Tensor<f32, { [-1] }>,
        qkv1: &Tensor<f32, { [-1] }>,
        qkv2: &Tensor<f32, { [-1] }>,
        beta0: &Tensor<f32, { [-1] }>,
        beta1: &Tensor<f32, { [-1] }>,
        beta2: &Tensor<f32, { [-1] }>,
        alpha0: &Tensor<f32, { [-1] }>,
        alpha1: &Tensor<f32, { [-1] }>,
        alpha2: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let key = pid.0 / (VH / KH);
        let head_bias = bias.partition(shape![1]).load([pid.0]);
        let rate = exp(a_log.partition(shape![1]).load([pid.0]));
        let initial: Tile<f32, { [D, SW] }> = state
            .partition(shape![1, D, SW])
            .load([pid.0, 0i32, pid.2])
            .reshape(shape![D, SW]);
        let active0 = lane_active(metadata0);
        let qp0 = qkv0.partition(shape![D]);
        let k0: Tile<f32, { [D] }> = qp0.load([KH + key]);
        let v0: Tile<f32, { [SW] }> = qkv0
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let bt0 = sigmoid(beta0.partition(shape![1]).load([pid.0]));
        let decay0 = delta_decay(alpha0.partition(shape![1]).load([pid.0]) + head_bias, rate);
        let after0 = delta_update(initial, k0, v0, bt0, decay0, active0);
        let q0: Tile<f32, { [D] }> = qp0.load([key]);
        out0.store(delta_output(after0, q0, active0));
        let active1 = lane_active(metadata1);
        let qp1 = qkv1.partition(shape![D]);
        let k1: Tile<f32, { [D] }> = qp1.load([KH + key]);
        let v1: Tile<f32, { [SW] }> = qkv1
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let bt1 = sigmoid(beta1.partition(shape![1]).load([pid.0]));
        let decay1 = delta_decay(alpha1.partition(shape![1]).load([pid.0]) + head_bias, rate);
        let after1 = delta_update(after0, k1, v1, bt1, decay1, active1);
        let q1: Tile<f32, { [D] }> = qp1.load([key]);
        out1.store(delta_output(after1, q1, active1));
        let active2 = lane_active(metadata2);
        let qp2 = qkv2.partition(shape![D]);
        let k2: Tile<f32, { [D] }> = qp2.load([KH + key]);
        let v2: Tile<f32, { [SW] }> = qkv2
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let bt2 = sigmoid(beta2.partition(shape![1]).load([pid.0]));
        let decay2 = delta_decay(alpha2.partition(shape![1]).load([pid.0]) + head_bias, rate);
        let after2 = delta_update(after1, k2, v2, bt2, decay2, active2);
        let q2: Tile<f32, { [D] }> = qp2.load([key]);
        out2.store(delta_output(after2, q2, active2));
    }

    /// Keep the immutable rate and bias in one launch operand.
    #[cutile::entry()]
    fn pack_parameters<const VH: i32>(
        out: &mut Tensor<f32, { [2, 1] }>,
        rate: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let rate = rate.partition(shape![1]).load([pid.1]);
        let bias = bias.partition(shape![1]).load([pid.1]);
        out.partition_mut(shape![1, 1])
            .store(rate.reshape(shape![1, 1]), [0i32, 0i32]);
        out.partition_mut(shape![1, 1])
            .store(bias.reshape(shape![1, 1]), [1i32, 0i32]);
    }

    /// The slot axis shares one launch; each CTA retains the original per-head reduction order.
    #[cutile::entry()]
    fn delta_verify_batch<
        const KH: i32,
        const VH: i32,
        const D: i32,
        const SW: i32,
        const MF: i32,
    >(
        out: &mut Tensor<f32, { [1, 3, SW] }>,
        state: &Tensor<f32, { [-1, D, D] }>,
        qkv: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        alpha: &Tensor<f32, { [-1] }>,
        parameters: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let slot = pid.0;
        let head = pid.2 / (D / SW);
        let split = pid.2 % (D / SW);
        let row0 = slot * 3i32;
        let row1 = row0 + 1i32;
        let row2 = row0 + 2i32;
        let base0 = row0 * (2i32 * KH + VH);
        let base1 = row1 * (2i32 * KH + VH);
        let base2 = row2 * (2i32 * KH + VH);
        let key = head / (VH / KH);
        let head_bias = parameters.partition(shape![1]).load([VH + head]);
        let rate = exp(parameters.partition(shape![1]).load([head]));
        let initial: Tile<f32, { [D, SW] }> = state
            .partition(shape![1, D, SW])
            .load([slot * VH + head, 0i32, split])
            .reshape(shape![D, SW]);
        let active0 = batch_active::<MF>(metadata, row0);
        let qp0 = qkv.partition(shape![D]);
        let k0: Tile<f32, { [D] }> = qp0.load([base0 + KH + key]);
        let v0: Tile<f32, { [SW] }> = qkv
            .partition(shape![SW])
            .load([(base0 + 2i32 * KH + head) * (D / SW) + split]);
        let bt0 = sigmoid(beta.partition(shape![1]).load([row0 * VH + head]));
        let decay0 = delta_decay(
            alpha.partition(shape![1]).load([row0 * VH + head]) + head_bias,
            rate,
        );
        let after0 = delta_update(initial, k0, v0, bt0, decay0, active0);
        let q0: Tile<f32, { [D] }> = qp0.load([base0 + key]);
        out.partition_mut(shape![1, 1, SW])
            .store(delta_output(after0, q0, active0), [0i32, 0i32, 0i32]);
        let active1 = batch_active::<MF>(metadata, row1);
        let qp1 = qkv.partition(shape![D]);
        let k1: Tile<f32, { [D] }> = qp1.load([base1 + KH + key]);
        let v1: Tile<f32, { [SW] }> = qkv
            .partition(shape![SW])
            .load([(base1 + 2i32 * KH + head) * (D / SW) + split]);
        let bt1 = sigmoid(beta.partition(shape![1]).load([row1 * VH + head]));
        let decay1 = delta_decay(
            alpha.partition(shape![1]).load([row1 * VH + head]) + head_bias,
            rate,
        );
        let after1 = delta_update(after0, k1, v1, bt1, decay1, active1);
        let q1: Tile<f32, { [D] }> = qp1.load([base1 + key]);
        out.partition_mut(shape![1, 1, SW])
            .store(delta_output(after1, q1, active1), [0i32, 1i32, 0i32]);
        let active2 = batch_active::<MF>(metadata, row2);
        let qp2 = qkv.partition(shape![D]);
        let k2: Tile<f32, { [D] }> = qp2.load([base2 + KH + key]);
        let v2: Tile<f32, { [SW] }> = qkv
            .partition(shape![SW])
            .load([(base2 + 2i32 * KH + head) * (D / SW) + split]);
        let bt2 = sigmoid(beta.partition(shape![1]).load([row2 * VH + head]));
        let decay2 = delta_decay(
            alpha.partition(shape![1]).load([row2 * VH + head]) + head_bias,
            rate,
        );
        let after2 = delta_update(after1, k2, v2, bt2, decay2, active2);
        let q2: Tile<f32, { [D] }> = qp2.load([base2 + key]);
        out.partition_mut(shape![1, 1, SW])
            .store(delta_output(after2, q2, active2), [0i32, 2i32, 0i32]);
    }

    /// Apply the accepted prefix, ignoring padding and every lane after the first padding.
    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA state kernel binds three verification lanes' rows explicitly"
    )]
    #[cutile::entry()]
    fn delta_commit3<const KH: i32, const VH: i32, const D: i32, const SW: i32>(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        qkv0: &Tensor<f32, { [-1] }>,
        qkv1: &Tensor<f32, { [-1] }>,
        qkv2: &Tensor<f32, { [-1] }>,
        beta0: &Tensor<f32, { [-1] }>,
        beta1: &Tensor<f32, { [-1] }>,
        beta2: &Tensor<f32, { [-1] }>,
        alpha0: &Tensor<f32, { [-1] }>,
        alpha1: &Tensor<f32, { [-1] }>,
        alpha2: &Tensor<f32, { [-1] }>,
        a_log: &Tensor<f32, { [-1] }>,
        bias: &Tensor<f32, { [-1] }>,
        metadata0: &Tensor<i32, { [-1] }>,
        metadata1: &Tensor<i32, { [-1] }>,
        metadata2: &Tensor<i32, { [-1] }>,
        count: i32,
    ) {
        let pid = get_tile_block_id();
        let key = pid.0 / (VH / KH);
        let head_bias = bias.partition(shape![1]).load([pid.0]);
        let rate = exp(a_log.partition(shape![1]).load([pid.0]));
        let initial: Tile<f32, { [D, SW] }> = state.load_like(state).reshape(shape![D, SW]);
        let applied0 =
            lane_active(metadata0) & ge_tile(count.broadcast(shape![1]), 1i32.broadcast(shape![1]));
        let qp0 = qkv0.partition(shape![D]);
        let k0: Tile<f32, { [D] }> = qp0.load([KH + key]);
        let v0: Tile<f32, { [SW] }> = qkv0
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let bt0 = sigmoid(beta0.partition(shape![1]).load([pid.0]));
        let decay0 = delta_decay(alpha0.partition(shape![1]).load([pid.0]) + head_bias, rate);
        let after0 = delta_update(initial, k0, v0, bt0, decay0, applied0);
        let applied1 = applied0
            & lane_active(metadata1)
            & ge_tile(count.broadcast(shape![1]), 2i32.broadcast(shape![1]));
        let qp1 = qkv1.partition(shape![D]);
        let k1: Tile<f32, { [D] }> = qp1.load([KH + key]);
        let v1: Tile<f32, { [SW] }> = qkv1
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let bt1 = sigmoid(beta1.partition(shape![1]).load([pid.0]));
        let decay1 = delta_decay(alpha1.partition(shape![1]).load([pid.0]) + head_bias, rate);
        let after1 = delta_update(after0, k1, v1, bt1, decay1, applied1);
        let applied2 = applied1
            & lane_active(metadata2)
            & ge_tile(count.broadcast(shape![1]), 3i32.broadcast(shape![1]));
        let qp2 = qkv2.partition(shape![D]);
        let k2: Tile<f32, { [D] }> = qp2.load([KH + key]);
        let v2: Tile<f32, { [SW] }> = qkv2
            .partition(shape![SW])
            .load([(2i32 * KH + pid.0) * (D / SW) + pid.2]);
        let bt2 = sigmoid(beta2.partition(shape![1]).load([pid.0]));
        let decay2 = delta_decay(alpha2.partition(shape![1]).load([pid.0]) + head_bias, rate);
        let after2 = delta_update(after1, k2, v2, bt2, decay2, applied2);
        state.store(after2.reshape(shape![1, D, SW]));
    }

    /// Commit independent per-slot accepted prefixes in a single state-family launch.
    #[cutile::entry()]
    fn delta_commit_batch<
        const KH: i32,
        const VH: i32,
        const D: i32,
        const SW: i32,
        const MF: i32,
    >(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        qkv: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        alpha: &Tensor<f32, { [-1] }>,
        parameters: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        counts: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let slot = pid.0;
        let head = pid.1;
        let row0 = slot * 3i32;
        let row1 = row0 + 1i32;
        let row2 = row0 + 2i32;
        let base0 = row0 * (2i32 * KH + VH);
        let base1 = row1 * (2i32 * KH + VH);
        let base2 = row2 * (2i32 * KH + VH);
        let count = counts.partition(shape![1]).load([slot]);
        let enabled: bool =
            tile_to_scalar(ge_tile(count, 1i32.broadcast(shape![1])).reshape(shape![]));
        if enabled {
            let key = head / (VH / KH);
            let head_bias = parameters.partition(shape![1]).load([VH + head]);
            let rate = exp(parameters.partition(shape![1]).load([head]));
            let initial: Tile<f32, { [D, SW] }> = state.load_like(state).reshape(shape![D, SW]);
            let applied0 =
                batch_active::<MF>(metadata, row0) & ge_tile(count, 1i32.broadcast(shape![1]));
            let qp0 = qkv.partition(shape![D]);
            let k0: Tile<f32, { [D] }> = qp0.load([base0 + KH + key]);
            let v0: Tile<f32, { [SW] }> =
                qkv.partition(shape![SW])
                    .load([(base0 + 2i32 * KH + head) * (D / SW) + pid.2]);
            let bt0 = sigmoid(beta.partition(shape![1]).load([row0 * VH + head]));
            let decay0 = delta_decay(
                alpha.partition(shape![1]).load([row0 * VH + head]) + head_bias,
                rate,
            );
            let after0 = delta_update(initial, k0, v0, bt0, decay0, applied0);
            let applied1 = applied0
                & batch_active::<MF>(metadata, row1)
                & ge_tile(count, 2i32.broadcast(shape![1]));
            let qp1 = qkv.partition(shape![D]);
            let k1: Tile<f32, { [D] }> = qp1.load([base1 + KH + key]);
            let v1: Tile<f32, { [SW] }> =
                qkv.partition(shape![SW])
                    .load([(base1 + 2i32 * KH + head) * (D / SW) + pid.2]);
            let bt1 = sigmoid(beta.partition(shape![1]).load([row1 * VH + head]));
            let decay1 = delta_decay(
                alpha.partition(shape![1]).load([row1 * VH + head]) + head_bias,
                rate,
            );
            let after1 = delta_update(after0, k1, v1, bt1, decay1, applied1);
            let applied2 = applied1
                & batch_active::<MF>(metadata, row2)
                & ge_tile(count, 3i32.broadcast(shape![1]));
            let qp2 = qkv.partition(shape![D]);
            let k2: Tile<f32, { [D] }> = qp2.load([base2 + KH + key]);
            let v2: Tile<f32, { [SW] }> =
                qkv.partition(shape![SW])
                    .load([(base2 + 2i32 * KH + head) * (D / SW) + pid.2]);
            let bt2 = sigmoid(beta.partition(shape![1]).load([row2 * VH + head]));
            let decay2 = delta_decay(
                alpha.partition(shape![1]).load([row2 * VH + head]) + head_bias,
                rate,
            );
            let after2 = delta_update(after1, k2, v2, bt2, decay2, applied2);
            state.store(after2.reshape(shape![1, D, SW]));
        }
    }

    /// [`delta_verify_batch`] for `L` candidate lanes per slot: the lanes run in order over
    /// one register-resident state, with the three-lane kernel's per-lane arithmetic.
    #[cutile::entry()]
    fn delta_verify_lanes<
        const KH: i32,
        const VH: i32,
        const D: i32,
        const SW: i32,
        const MF: i32,
        const L: i32,
    >(
        out: &mut Tensor<f32, { [1, L, SW] }>,
        state: &Tensor<f32, { [-1, D, D] }>,
        qkv: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        alpha: &Tensor<f32, { [-1] }>,
        parameters: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let slot = pid.0;
        let head = pid.2 / (D / SW);
        let split = pid.2 % (D / SW);
        let key = head / (VH / KH);
        let head_bias = parameters.partition(shape![1]).load([VH + head]);
        let rate = exp(parameters.partition(shape![1]).load([head]));
        let mut current: Tile<f32, { [D, SW] }> = state
            .partition(shape![1, D, SW])
            .load([slot * VH + head, 0i32, split])
            .reshape(shape![D, SW]);
        let qp = qkv.partition(shape![D]);
        let vp = qkv.partition(shape![SW]);
        for lane in 0i32..L {
            let row = slot * L + lane;
            let base = row * (2i32 * KH + VH);
            let active = batch_active::<MF>(metadata, row);
            let k: Tile<f32, { [D] }> = qp.load([base + KH + key]);
            let v: Tile<f32, { [SW] }> = vp.load([(base + 2i32 * KH + head) * (D / SW) + split]);
            let bt = sigmoid(beta.partition(shape![1]).load([row * VH + head]));
            let decay = delta_decay(
                alpha.partition(shape![1]).load([row * VH + head]) + head_bias,
                rate,
            );
            current = delta_update(current, k, v, bt, decay, active);
            let q: Tile<f32, { [D] }> = qp.load([base + key]);
            out.partition_mut(shape![1, 1, SW])
                .store(delta_output(current, q, active), [0i32, lane, 0i32]);
        }
    }

    /// [`delta_commit_batch`] for `L` candidate lanes per slot.
    #[cutile::entry()]
    fn delta_commit_lanes<
        const KH: i32,
        const VH: i32,
        const D: i32,
        const SW: i32,
        const MF: i32,
        const L: i32,
    >(
        state: &mut Tensor<f32, { [1, D, SW] }>,
        qkv: &Tensor<f32, { [-1] }>,
        beta: &Tensor<f32, { [-1] }>,
        alpha: &Tensor<f32, { [-1] }>,
        parameters: &Tensor<f32, { [-1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        counts: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let slot = pid.0;
        let head = pid.1;
        let count = counts.partition(shape![1]).load([slot]);
        let enabled: bool =
            tile_to_scalar(ge_tile(count, 1i32.broadcast(shape![1])).reshape(shape![]));
        if enabled {
            let key = head / (VH / KH);
            let head_bias = parameters.partition(shape![1]).load([VH + head]);
            let rate = exp(parameters.partition(shape![1]).load([head]));
            let mut current: Tile<f32, { [D, SW] }> = state.load_like(state).reshape(shape![D, SW]);
            let mut applied: Tile<bool, { [1] }> = ge_tile(count, 1i32.broadcast(shape![1]));
            let qp = qkv.partition(shape![D]);
            let vp = qkv.partition(shape![SW]);
            for lane in 0i32..L {
                let row = slot * L + lane;
                let base = row * (2i32 * KH + VH);
                applied = applied
                    & batch_active::<MF>(metadata, row)
                    & ge_tile(count, (lane + 1i32).broadcast(shape![1]));
                let k: Tile<f32, { [D] }> = qp.load([base + KH + key]);
                let v: Tile<f32, { [SW] }> =
                    vp.load([(base + 2i32 * KH + head) * (D / SW) + pid.2]);
                let bt = sigmoid(beta.partition(shape![1]).load([row * VH + head]));
                let decay = delta_decay(
                    alpha.partition(shape![1]).load([row * VH + head]) + head_bias,
                    rate,
                );
                current = delta_update(current, k, v, bt, decay, applied);
            }
            state.store(current.reshape(shape![1, D, SW]));
        }
    }

    /// [`conv_commit_batch`] for `L` candidate lanes per slot: the window after `n` applied
    /// lanes is the last three of `[h0, h1, h2, x0, ..., x(n-1)]`.
    #[cutile::entry()]
    fn conv_commit_lanes<const MF: i32, const L: i32>(
        h0: &mut Tensor<f32, { [1, 128] }>,
        h1: &mut Tensor<f32, { [1, 128] }>,
        h2: &mut Tensor<f32, { [1, 128] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        counts: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let count = counts.partition(shape![1]).load([pid.0]);
        let enabled: bool =
            tile_to_scalar(ge_tile(count, 1i32.broadcast(shape![1])).reshape(shape![]));
        if enabled {
            let mut oldest = h0.load_like(h0).reshape(shape![128]);
            let mut middle = h1.load_like(h1).reshape(shape![128]);
            let mut newest = h2.load_like(h2).reshape(shape![128]);
            let mut applied: Tile<bool, { [1] }> = ge_tile(count, 1i32.broadcast(shape![1]));
            for lane in 0i32..L {
                let row = pid.0 * L + lane;
                applied = applied
                    & batch_active::<MF>(metadata, row)
                    & ge_tile(count, (lane + 1i32).broadcast(shape![1]));
                let shift = applied.broadcast(shape![128]);
                let next = input
                    .partition(shape![1, 128])
                    .load([row, pid.1])
                    .reshape(shape![128]);
                oldest = select(shift, middle, oldest);
                middle = select(shift, newest, middle);
                newest = select(shift, next, newest);
            }
            h0.store(oldest.reshape(shape![1, 128]));
            h1.store(middle.reshape(shape![1, 128]));
            h2.store(newest.reshape(shape![1, 128]));
        }
    }

    /// Shift each slot's window by its own accepted prefix; other slots keep their state.
    #[cutile::entry()]
    fn conv_commit_batch<const MF: i32>(
        h0: &mut Tensor<f32, { [1, 128] }>,
        h1: &mut Tensor<f32, { [1, 128] }>,
        h2: &mut Tensor<f32, { [1, 128] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
        counts: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0 * 3i32;
        let count = counts.partition(shape![1]).load([pid.0]);
        let enabled: bool =
            tile_to_scalar(ge_tile(count, 1i32.broadcast(shape![1])).reshape(shape![]));
        if enabled {
            let oldest = h0.load_like(h0).reshape(shape![128]);
            let middle = h1.load_like(h1).reshape(shape![128]);
            let newest = h2.load_like(h2).reshape(shape![128]);
            let first = input
                .partition(shape![1, 128])
                .load([row, pid.1])
                .reshape(shape![128]);
            let second = input
                .partition(shape![1, 128])
                .load([row + 1i32, pid.1])
                .reshape(shape![128]);
            let third = input
                .partition(shape![1, 128])
                .load([row + 2i32, pid.1])
                .reshape(shape![128]);
            let one = batch_active::<MF>(metadata, row) & ge_tile(count, 1i32.broadcast(shape![1]));
            let two = one
                & batch_active::<MF>(metadata, row + 1i32)
                & ge_tile(count, 2i32.broadcast(shape![1]));
            let three = two
                & batch_active::<MF>(metadata, row + 2i32)
                & ge_tile(count, 3i32.broadcast(shape![1]));
            let one = one.broadcast(shape![128]);
            let two = two.broadcast(shape![128]);
            let three = three.broadcast(shape![128]);
            h0.store(
                select(
                    three,
                    first,
                    select(two, newest, select(one, middle, oldest)),
                )
                .reshape(shape![1, 128]),
            );
            h1.store(
                select(
                    three,
                    second,
                    select(two, first, select(one, newest, middle)),
                )
                .reshape(shape![1, 128]),
            );
            h2.store(
                select(
                    three,
                    third,
                    select(two, second, select(one, first, newest)),
                )
                .reshape(shape![1, 128]),
            );
        }
    }
}
