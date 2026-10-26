//! Snapshot-free recurrent verification for the pooled speculation graph.
//!
//! The verification lanes of one slot are scored in one launch per recurrent node that reads the
//! slot's state and leaves it untouched; once acceptance is known, one commit launch applies the
//! accepted prefix. Rolling back therefore needs no per-lane state snapshot, which was two full
//! recurrent state copies (about 300 MB on the 27B) per slot.
#[cutile::module]
pub(crate) mod fused {
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

use super::{ActivationArena, ProgramWeights, batch::States, capture::error};
use crate::constants::{CONV_KERNEL_SIZE, CONV_KERNEL_TILE, GDN_PROJECTED_INPUTS};
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::{Error, TensorId};
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Candidate lanes of one slot that the fused kernels cover.
pub(super) const LANES: usize = 3;

/// One slot's verification lanes in the batched arena and their per-lane metadata.
pub(super) struct SlotLanes<'a> {
    pub lanes: [usize; LANES],
    pub metadata: [&'a Tensor<i32>; LANES],
}

/// One node input's candidate rows for a slot.
type StagedInputs = Vec<[Tensor<f32>; LANES]>;

/// Commit views precede their owning batched buffers in drop order. Verification copies
/// each input once for all slots; commits keep the same per-slot/candidate row layout.
#[derive(Default)]
pub(super) struct Staging {
    entries: BTreeMap<(usize, usize), StagedInputs>,
    roots: BTreeMap<(usize, usize), Tensor<f32>>,
}

impl Staging {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn root(&self, index: usize, input: usize) -> Option<&Tensor<f32>> {
        self.roots.get(&(index, input))
    }

    pub(super) fn get(&self, key: &(usize, usize)) -> Option<&StagedInputs> {
        self.entries.get(key)
    }

    /// Preserve every slot's commit input before the arena reuses its allocation.
    pub(super) fn record_node(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &ActivationArena,
        index: usize,
    ) -> Result<(), DeviceError> {
        for (input, &id) in commit_inputs(node).iter().enumerate() {
            let source = arena.get(id).map_err(error)?;
            let target = self
                .roots
                .get_mut(&(index, input))
                .ok_or_else(|| error("batched recurrent staging input"))?;
            if source.size() != target.size() {
                return Err(error("batched recurrent staging geometry"));
            }
            scope.record(api::memcpy(target, source))?;
        }
        Ok(())
    }
}

/// Activations a covered node's commit re-reads; weights are read in place.
pub(super) fn commit_inputs(node: &TensorNode) -> &[TensorId] {
    match node.op {
        TensorOp::Conv { .. } => &node.inputs[..1],
        _ => &node.inputs[..GDN_PROJECTED_INPUTS],
    }
}

/// Allocate the staging rows of every covered node for `slots` slots.
/// # Errors
/// Rejects unbatched activations and failed device allocation.
pub(super) fn allocate_staging(
    device: &CudaDevice,
    nodes: &[TensorNode],
    arena: &ActivationArena,
    slots: usize,
) -> infer_core::Result<Staging> {
    let lanes = slots
        .checked_mul(LANES)
        .ok_or_else(|| Error::invalid("batched recurrent staging overflow"))?;
    if lanes != arena.lanes() {
        return Err(Error::invalid("batched recurrent staging lanes"));
    }
    let mut staging = Staging::new();
    for (index, node) in nodes.iter().enumerate().filter(|(_, node)| covers(node)) {
        for (input, &id) in commit_inputs(node).iter().enumerate() {
            let mut root = api::zeros::<f32>(&[arena.get(id)?.size()])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            for slot in 0..slots {
                let base = slot * LANES;
                let (first, owner) = staging_row(root, lanes, base)?;
                let (second, owner) = staging_row(owner, lanes, base + 1)?;
                let (third, owner) = staging_row(owner, lanes, base + 2)?;
                root = owner;
                staging
                    .entries
                    .entry((slot, index))
                    .or_default()
                    .push([first, second, third]);
            }
            staging.roots.insert((index, input), root);
        }
    }
    Ok(staging)
}

/// Borrow a row while returning the unique parent for its persistent staging anchor.
fn staging_row(
    root: Tensor<f32>,
    lanes: usize,
    lane: usize,
) -> infer_core::Result<(Tensor<f32>, Tensor<f32>)> {
    let (row, owner) = super::arena::row_tensor(root, lanes, lane)?;
    let root =
        Arc::try_unwrap(owner).map_err(|_| Error::invariant("unique recurrent staging owner"))?;
    Ok((row, root))
}

/// Value-dimension block of the fused kernels: one CTA per value head per split. Every
/// reduction is over the key dimension, so the split changes no summation order.
pub(super) const fn fused_block(value_dim: usize, split: usize) -> usize {
    if split > 1 && value_dim.is_multiple_of(split) {
        value_dim / split
    } else {
        value_dim
    }
}

/// Value-dimension split of the fused verify and commit kernels for one covered geometry.
///
/// The target is one wave of CTAs: `head_count × slots × split` should cover the SM count.
/// Only splits that divide the actual value width and stay within the supported small set are
/// legal; the lower divisor wins ties so a small device never over-partitions a tail.
pub(super) fn value_split(
    multiprocessors: u32,
    heads: usize,
    slots: usize,
    value_dim: usize,
) -> usize {
    const LEGAL: [usize; 4] = [1, 2, 4, 8];
    let lanes = heads.max(1).saturating_mul(slots.max(1));
    let target = usize::try_from(multiprocessors)
        .unwrap_or(usize::MAX)
        .max(1);
    let desired = target.div_ceil(lanes).max(1);
    LEGAL
        .into_iter()
        .filter(|split| value_dim.is_multiple_of(*split))
        .min_by_key(|split| split.abs_diff(desired))
        .unwrap_or(1)
}

/// Whether `node` is a recurrent state writer the fused verify/commit pair covers.
pub(super) const fn covers(node: &TensorNode) -> bool {
    node.outputs.len() == 1
        && node.states.len() == 1
        && match node.op {
            TensorOp::Conv { kernel, .. } => kernel == CONV_KERNEL_SIZE,
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                key_dim == value_dim
                    && key_dim.is_power_of_two()
                    && key_heads > 0
                    && value_heads.is_multiple_of(key_heads)
            }
            _ => false,
        }
}

/// A weight constant, or one lane's row of a batched activation.
fn lane_input<'a>(
    weights: &'a ProgramWeights,
    arena: &'a ActivationArena,
    id: TensorId,
    lane: usize,
) -> Result<TensorView<'a, f32>, DeviceError> {
    weights.constants.get(&id).map_or_else(
        || arena.row(id, lane).map_err(error),
        |constant| constant.view(&[constant.size()]).map_err(error),
    )
}

fn rows<'a>(
    weights: &'a ProgramWeights,
    arena: &'a ActivationArena,
    id: TensorId,
    slot: &SlotLanes<'_>,
) -> Result<[TensorView<'a, f32>; LANES], DeviceError> {
    let [first, second, third] = slot.lanes;
    Ok([
        lane_input(weights, arena, id, first)?,
        lane_input(weights, arena, id, second)?,
        lane_input(weights, arena, id, third)?,
    ])
}

/// Record the outputs of one slot's verification lanes for a covered recurrent node, reading
/// but never writing the slot's state.
#[expect(
    unsafe_code,
    reason = "Audited row alias: the three output rows are disjoint lanes of one arena slot, \
              which outlives the captured graph"
)]
pub(super) fn record_verify(
    scope: &Scope,
    node: &TensorNode,
    arena: &mut ActivationArena,
    states: &States,
    slot: &SlotLanes<'_>,
    weights: &ProgramWeights,
    value_split: usize,
) -> Result<(), DeviceError> {
    let index = arena.slot(node.outputs[0]).map_err(error)?;
    let buffer = arena.buffers[index]
        .take()
        .ok_or_else(|| error("missing slot verify output"))?;
    let lanes = arena.lanes();
    let recorded = (|| {
        let state = states
            .get(&node.states[0])
            .ok_or_else(|| error("slot verify state"))?;
        let [first, second, third] = slot.lanes;
        // SAFETY: see the expectation above.
        let o0 = unsafe { super::arena::row_view(&buffer, lanes, first) }.map_err(error)?;
        // SAFETY: see the expectation above.
        let o1 = unsafe { super::arena::row_view(&buffer, lanes, second) }.map_err(error)?;
        // SAFETY: see the expectation above.
        let o2 = unsafe { super::arena::row_view(&buffer, lanes, third) }.map_err(error)?;
        let [m0, m1, m2] = slot.metadata;
        match node.op {
            TensorOp::Conv { channels, .. } => {
                let [h0, h1, h2] = state.as_slice() else {
                    return Err(error("convolution state layout"));
                };
                let (mut o0, mut o1, mut o2) = (o0, o1, o2);
                let x = rows(weights, arena, node.inputs[0], slot)?;
                let weight = lane_input(weights, arena, node.inputs[1], slot.lanes[0])?;
                let w = weight.view(&[channels, CONV_KERNEL_SIZE])?;
                let tile = CONV_KERNEL_TILE;
                scope.record(fused::conv_verify3(
                    (&mut o0).partition([tile]),
                    (&mut o1).partition([tile]),
                    (&mut o2).partition([tile]),
                    &h0.view(&[h0.size()])?,
                    &h1.view(&[h1.size()])?,
                    &h2.view(&[h2.size()])?,
                    &x[0],
                    &x[1],
                    &x[2],
                    &w,
                    m0,
                    m1,
                    m2,
                ))?;
            }
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            } => {
                let [state] = state.as_slice() else {
                    return Err(error("delta state layout"));
                };
                let block = fused_block(value_dim, value_split);
                let shape = [value_heads, 1, value_dim];
                let (mut o0, mut o1, mut o2) = (
                    o0.reshape(&shape)?,
                    o1.reshape(&shape)?,
                    o2.reshape(&shape)?,
                );
                let qkv = rows(weights, arena, node.inputs[0], slot)?;
                let beta = rows(weights, arena, node.inputs[1], slot)?;
                let alpha = rows(weights, arena, node.inputs[2], slot)?;
                scope.record(
                    fused::delta_verify3(
                        (&mut o0).partition([1, 1, block]),
                        (&mut o1).partition([1, 1, block]),
                        (&mut o2).partition([1, 1, block]),
                        &state.view(&[value_heads, key_dim, value_dim])?,
                        &qkv[0],
                        &qkv[1],
                        &qkv[2],
                        &beta[0],
                        &beta[1],
                        &beta[2],
                        &alpha[0],
                        &alpha[1],
                        &alpha[2],
                        &lane_input(weights, arena, node.inputs[3], slot.lanes[0])?,
                        &lane_input(weights, arena, node.inputs[4], slot.lanes[0])?,
                        m0,
                        m1,
                        m2,
                    )
                    .generics(vec![
                        key_heads.to_string(),
                        value_heads.to_string(),
                        key_dim.to_string(),
                        block.to_string(),
                    ]),
                )?;
            }
            _ => return Err(error("slot verify recurrent node")),
        }
        Ok(())
    })();
    arena.buffers[index] = Some(buffer);
    recorded
}

/// Record the commit of one slot's first `count` verification lanes for a covered node.
#[expect(
    clippy::too_many_arguments,
    reason = "The captured commit node binds scope, geometry, state, lanes and staging explicitly"
)]
pub(super) fn record_commit(
    scope: &Scope,
    node: &TensorNode,
    arena: &ActivationArena,
    states: &mut States,
    slot: &SlotLanes<'_>,
    weights: &ProgramWeights,
    value_split: usize,
    (staged, count): (&[[Tensor<f32>; LANES]], usize),
) -> Result<(), DeviceError> {
    let count = i32::try_from(count).map_err(error)?;
    let state = states
        .get_mut(&node.states[0])
        .ok_or_else(|| error("slot commit state"))?;
    let [m0, m1, m2] = slot.metadata;
    match node.op {
        TensorOp::Conv { .. } => {
            let [h0, h1, h2] = state.as_mut_slice() else {
                return Err(error("convolution state layout"));
            };
            let [x] = staged else {
                return Err(error("convolution staging layout"));
            };
            let tile = CONV_KERNEL_TILE;
            scope.record(fused::conv_commit3(
                h0.partition([tile]),
                h1.partition([tile]),
                h2.partition([tile]),
                &x[0],
                &x[1],
                &x[2],
                m0,
                m1,
                m2,
                count,
            ))?;
        }
        TensorOp::Delta {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } => {
            let [state] = state.as_mut_slice() else {
                return Err(error("delta state layout"));
            };
            let block = fused_block(value_dim, value_split);
            let [qkv, beta, alpha] = staged else {
                return Err(error("delta staging layout"));
            };
            scope.record(
                fused::delta_commit3(
                    state.partition([1, key_dim, block]),
                    &qkv[0],
                    &qkv[1],
                    &qkv[2],
                    &beta[0],
                    &beta[1],
                    &beta[2],
                    &alpha[0],
                    &alpha[1],
                    &alpha[2],
                    &lane_input(weights, arena, node.inputs[3], slot.lanes[0])?,
                    &lane_input(weights, arena, node.inputs[4], slot.lanes[0])?,
                    m0,
                    m1,
                    m2,
                    count,
                )
                .generics(vec![
                    key_heads.to_string(),
                    value_heads.to_string(),
                    key_dim.to_string(),
                    block.to_string(),
                ]),
            )?;
        }
        _ => return Err(error("slot commit recurrent node")),
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_recurrent.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/slot_staging.rs"]
mod staging_tests;
