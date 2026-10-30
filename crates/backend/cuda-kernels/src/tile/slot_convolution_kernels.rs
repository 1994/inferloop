//! Read-only three-candidate convolution verification across the pool's slot axis.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, Tensor_1, Tensor_2, Tensor_3, Tile_1,
        exp, ge_tile, get_tile_block_id, select,
    };

    fn active<const MF: i32>(metadata: &Tensor<i32, { [-1] }>, row: i32) -> Tile<bool, { [128] }> {
        let position = metadata.partition(shape![1]).load([row * MF + 2i32]);
        ge_tile(position, 0i32.broadcast(shape![1])).broadcast(shape![128])
    }

    /// [`verify`] for `L` candidate lanes per slot, sliding the window over the lanes.
    #[cutile::entry()]
    fn verify_lanes<const C: i32, const MF: i32, const L: i32>(
        out: &mut Tensor<f32, { [1, L, 128] }>,
        input: &Tensor<f32, { [-1, L, C] }>,
        h0: &Tensor<f32, { [-1, C] }>,
        h1: &Tensor<f32, { [-1, C] }>,
        h2: &Tensor<f32, { [-1, C] }>,
        weight: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let mut oldest = h0
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let mut middle = h1
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let mut newest = h2
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let inputs = input.partition(shape![1, 1, 128]);
        let weights = weight.partition(shape![128, 1]);
        let wa = weights.load([pid.2, 0i32]).reshape(shape![128]);
        let wb = weights.load([pid.2, 1i32]).reshape(shape![128]);
        let wc = weights.load([pid.2, 2i32]).reshape(shape![128]);
        let wd = weights.load([pid.2, 3i32]).reshape(shape![128]);
        let one = 1.0f32.broadcast(shape![128]);
        let zero = 0.0f32.broadcast(shape![128]);
        for lane in 0i32..L {
            let current = inputs.load([pid.0, lane, pid.2]).reshape(shape![128]);
            let value = oldest * wa + middle * wb + newest * wc + current * wd;
            let silu = value / (one + exp(zero - value));
            out.partition_mut(shape![1, 1, 128]).store(
                select(active::<MF>(metadata, pid.0 * L + lane), silu, zero)
                    .reshape(shape![1, 1, 128]),
                [0i32, lane, 0i32],
            );
            oldest = middle;
            middle = newest;
            newest = current;
        }
    }

    #[cutile::entry()]
    fn verify<const C: i32, const MF: i32>(
        out: &mut Tensor<f32, { [1, 3, 128] }>,
        input: &Tensor<f32, { [-1, 3, C] }>,
        h0: &Tensor<f32, { [-1, C] }>,
        h1: &Tensor<f32, { [-1, C] }>,
        h2: &Tensor<f32, { [-1, C] }>,
        weight: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let oldest = h0
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let middle = h1
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let newest = h2
            .partition(shape![1, 128])
            .load([pid.0, pid.2])
            .reshape(shape![128]);
        let inputs = input.partition(shape![1, 1, 128]);
        let first = inputs.load([pid.0, 0i32, pid.2]).reshape(shape![128]);
        let second = inputs.load([pid.0, 1i32, pid.2]).reshape(shape![128]);
        let third = inputs.load([pid.0, 2i32, pid.2]).reshape(shape![128]);
        let weights = weight.partition(shape![128, 1]);
        let wa = weights.load([pid.2, 0i32]).reshape(shape![128]);
        let wb = weights.load([pid.2, 1i32]).reshape(shape![128]);
        let wc = weights.load([pid.2, 2i32]).reshape(shape![128]);
        let wd = weights.load([pid.2, 3i32]).reshape(shape![128]);
        let one = 1.0f32.broadcast(shape![128]);
        let zero = 0.0f32.broadcast(shape![128]);
        let value0 = oldest * wa + middle * wb + newest * wc + first * wd;
        let value1 = middle * wa + newest * wb + first * wc + second * wd;
        let value2 = newest * wa + first * wb + second * wc + third * wd;
        let silu0 = value0 / (one + exp(zero - value0));
        let silu1 = value1 / (one + exp(zero - value1));
        let silu2 = value2 / (one + exp(zero - value2));
        out.partition_mut(shape![1, 1, 128]).store(
            select(active::<MF>(metadata, pid.0 * 3i32), silu0, zero).reshape(shape![1, 1, 128]),
            [0i32, 0i32, 0i32],
        );
        out.partition_mut(shape![1, 1, 128]).store(
            select(active::<MF>(metadata, pid.0 * 3i32 + 1i32), silu1, zero)
                .reshape(shape![1, 1, 128]),
            [0i32, 1i32, 0i32],
        );
        out.partition_mut(shape![1, 1, 128]).store(
            select(active::<MF>(metadata, pid.0 * 3i32 + 2i32), silu2, zero)
                .reshape(shape![1, 1, 128]),
            [0i32, 2i32, 0i32],
        );
    }
}
