//! Parallel causal convolution across a prompt chunk, followed by one state commit.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tile_1, exp, get_tile_block_id, tile_to_scalar,
    };
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow"
    )]
    #[cutile::entry()]
    fn forward(
        out: &mut Tensor<f32, { [1, 128] }>,
        h0: &Tensor<f32, { [-1] }>,
        h1: &Tensor<f32, { [-1] }>,
        h2: &Tensor<f32, { [-1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        w: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let mut first = 0i32;
        if base + offset < 0 {
            first = 0i32 - base - offset;
        }
        let lane = pid.0;
        if lane >= first && lane < count {
            let xp = x.partition(shape![1, 128]);
            let mut oldest = h0.partition(shape![128]).load([pid.1]);
            let mut middle = h1.partition(shape![128]).load([pid.1]);
            let mut newest = h2.partition(shape![128]).load([pid.1]);
            if lane >= first + 3i32 {
                oldest = xp.load([lane - 3i32, pid.1]).reshape(shape![128]);
            } else if lane == first + 2i32 {
                oldest = newest;
            } else if lane == first + 1i32 {
                oldest = middle;
            }
            if lane >= first + 2i32 {
                middle = xp.load([lane - 2i32, pid.1]).reshape(shape![128]);
            } else if lane == first + 1i32 {
                middle = newest;
            }
            if lane >= first + 1i32 {
                newest = xp.load([lane - 1i32, pid.1]).reshape(shape![128]);
            }
            let value = xp.load([lane, pid.1]).reshape(shape![128]);
            let wp = w.partition(shape![128, 1]);
            let wa = wp.load([pid.1, 0i32]).reshape(shape![128]);
            let wb = wp.load([pid.1, 1i32]).reshape(shape![128]);
            let wc = wp.load([pid.1, 2i32]).reshape(shape![128]);
            let wd = wp.load([pid.1, 3i32]).reshape(shape![128]);
            let value = oldest * wa + middle * wb + newest * wc + value * wd;
            let negative: Tile<f32, { [128] }> = 0.0f32.broadcast(shape![128]) - value;
            out.store(
                (value / (1.0f32.broadcast(shape![128]) + exp(negative))).reshape(shape![1, 128]),
            );
        } else {
            out.store(0.0f32.broadcast(shape![1, 128]));
        }
    }
    /// [`forward`] over 16 consecutive rows of one 128-channel tile per block: the four-tap
    /// window slides in registers, so each input row loads once instead of four times and the
    /// grid has a sixteenth of the blocks.
    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow"
    )]
    #[cutile::entry()]
    fn forward_rows(
        out: &mut Tensor<f32, { [16, 128] }>,
        h0: &Tensor<f32, { [-1] }>,
        h1: &Tensor<f32, { [-1] }>,
        h2: &Tensor<f32, { [-1] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        w: &Tensor<f32, { [-1, 4] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let mut first = 0i32;
        if base + offset < 0 {
            first = 0i32 - base - offset;
        }
        let start = pid.0 * 16i32;
        let xp = x.partition(shape![1, 128]);
        let state0: Tile<f32, { [128] }> = h0.partition(shape![128]).load([pid.1]);
        let state1: Tile<f32, { [128] }> = h1.partition(shape![128]).load([pid.1]);
        let state2: Tile<f32, { [128] }> = h2.partition(shape![128]).load([pid.1]);
        let wp = w.partition(shape![128, 1]);
        let wa = wp.load([pid.1, 0i32]).reshape(shape![128]);
        let wb = wp.load([pid.1, 1i32]).reshape(shape![128]);
        let wc = wp.load([pid.1, 2i32]).reshape(shape![128]);
        let wd = wp.load([pid.1, 3i32]).reshape(shape![128]);
        // Rows before `first` come from the carried state: first-1 is the newest.
        let mut oldest = state0;
        if start - 3i32 >= first {
            oldest = xp.load([start - 3i32, pid.1]).reshape(shape![128]);
        } else if start - 3i32 == first - 1i32 {
            oldest = state2;
        } else if start - 3i32 == first - 2i32 {
            oldest = state1;
        }
        let mut middle = state1;
        if start - 2i32 >= first {
            middle = xp.load([start - 2i32, pid.1]).reshape(shape![128]);
        } else if start - 2i32 == first - 1i32 {
            middle = state2;
        } else if start - 2i32 == first - 3i32 {
            middle = state0;
        }
        let mut newest = state2;
        if start - 1i32 >= first {
            newest = xp.load([start - 1i32, pid.1]).reshape(shape![128]);
        } else if start - 1i32 == first - 2i32 {
            newest = state1;
        } else if start - 1i32 == first - 3i32 {
            newest = state0;
        }
        let mut rows = out.partition_mut(shape![1, 128]);
        let zero: Tile<f32, { [128] }> = 0.0f32.broadcast(shape![128]);
        let one: Tile<f32, { [128] }> = 1.0f32.broadcast(shape![128]);
        for row in 0i32..16i32 {
            let lane = start + row;
            let mut current = state2;
            if lane >= first && lane < count {
                current = xp.load([lane, pid.1]).reshape(shape![128]);
            } else if lane == first - 1i32 {
                current = state2;
            } else if lane == first - 2i32 {
                current = state1;
            } else if lane == first - 3i32 {
                current = state0;
            }
            let value = oldest * wa + middle * wb + newest * wc + current * wd;
            let mut result = zero;
            if lane >= first && lane < count {
                result = value / (one + exp(zero - value));
            }
            rows.store(result.reshape(shape![1, 128]), [row, 0i32]);
            oldest = middle;
            middle = newest;
            newest = current;
        }
    }

    #[expect(
        clippy::useless_let_if_seq,
        reason = "cuTile DSL uses statement control flow"
    )]
    #[cutile::entry()]
    fn commit(
        h0: &mut Tensor<f32, { [128] }>,
        h1: &mut Tensor<f32, { [128] }>,
        h2: &mut Tensor<f32, { [128] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let meta = metadata.partition(shape![1]);
        let base: i32 = tile_to_scalar(meta.load([0i32]).reshape(shape![]));
        let count: i32 = tile_to_scalar(meta.load([1i32]).reshape(shape![]));
        let offset: i32 = tile_to_scalar(meta.load([2i32]).reshape(shape![]));
        let mut first = 0i32;
        if base + offset < 0 {
            first = 0i32 - base - offset;
        }
        if count > first {
            let xp = x.partition(shape![1, 128]);
            let mut oldest = h1.load_like(h0);
            let mut middle = h2.load_like(h0);
            if count - first >= 3i32 {
                oldest = xp.load([count - 3i32, pid.0]).reshape(shape![128]);
            } else if count - first == 2i32 {
                oldest = middle;
            }
            if count - first >= 2i32 {
                middle = xp.load([count - 2i32, pid.0]).reshape(shape![128]);
            }
            h0.store(oldest);
            h1.store(middle);
            h2.store(xp.load([count - 1i32, pid.0]).reshape(shape![128]));
        }
    }
}
