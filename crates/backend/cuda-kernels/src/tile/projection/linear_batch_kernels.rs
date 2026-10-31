//! Rows share each weight tile in groups of four: grid dim 0 walks the row groups of the
//! `[rows, N]` output (three verify lanes today, nine pooled speculation rows with MTP),
//! so one replay reads every weight once. F32 accumulation, no activation quantization.
#[cutile::module]
pub mod batched {
    use cutile::core::{
        BroadcastScalar, ElementType, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tile_2, Tile_3, UnpackF4e2m1fnx2Tile, constant, convert_tile,
        f4e2m1fnx2, f8e4m3fn, get_tile_block_id, reduce_sum,
    };

    /// Generic candidate-width packing; the three-row case retains its fused copy.
    #[cutile::entry()]
    fn pack_row(out: &mut Tensor<f32, { [1, 256] }>, input: &Tensor<f32, { [-1] }>, row: i32) {
        let pid = get_tile_block_id();
        if pid.0 == row {
            out.store(
                input
                    .partition(shape![256])
                    .load([pid.1])
                    .reshape(shape![1, 256]),
            );
        }
    }
    #[cutile::entry()]
    fn unpack_row(out: &mut Tensor<f32, { [256] }>, input: &Tensor<f32, { [-1, -1] }>, row: i32) {
        let pid = get_tile_block_id();
        out.store(
            input
                .partition(shape![1, 256])
                .load([row, pid.0])
                .reshape(shape![256]),
        );
    }

    #[cutile::entry()]
    fn pack_inputs(
        out: &mut Tensor<f32, { [1, 256] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        c: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        if pid.0 == 0 {
            out.store(
                a.partition(shape![256])
                    .load([pid.1])
                    .reshape(shape![1, 256]),
            );
        } else if pid.0 == 1 {
            out.store(
                b.partition(shape![256])
                    .load([pid.1])
                    .reshape(shape![1, 256]),
            );
        } else {
            out.store(
                c.partition(shape![256])
                    .load([pid.1])
                    .reshape(shape![1, 256]),
            );
        }
    }

    #[cutile::entry()]
    fn unpack_outputs(
        a: &mut Tensor<f32, { [256] }>,
        b: &mut Tensor<f32, { [256] }>,
        c: &mut Tensor<f32, { [256] }>,
        input: &Tensor<f32, { [3, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let part = input.partition(shape![1, 256]);
        a.store(part.load([0i32, pid.0]).reshape(shape![256]));
        b.store(part.load([1i32, pid.0]).reshape(shape![256]));
        c.store(part.load([2i32, pid.0]).reshape(shape![256]));
    }

    #[cutile::entry()]
    fn dense<E: ElementType, const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [4, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![4, BK]);
        let wp = w.partition(shape![BN, BK]);
        let mut sum: Tile<f32, { [4, BN, BK] }> = constant(0.0f32, shape![4, BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let input = xp
                .load([pid.0, k])
                .reshape(shape![4, 1, BK])
                .broadcast(shape![4, BN, BK]);
            let weight: Tile<f32, { [BN, BK] }> = convert_tile(wp.load([pid.1, k]));
            sum = sum
                + weight
                    .reshape(shape![1, BN, BK])
                    .broadcast(shape![4, BN, BK])
                    * input;
        }
        let reduced: Tile<f32, { [4, BN] }> = reduce_sum(sum, 2i32);
        let reduced = reduced.reshape(shape![4, BN]);
        out.store(reduced);
    }

    #[cutile::entry()]
    fn fp8<const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [4, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![4, BK]);
        let wp = w.partition(shape![BN, BK]);
        let mut sum: Tile<f32, { [4, BN, BK] }> = constant(0.0f32, shape![4, BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let input = xp
                .load([pid.0, k])
                .reshape(shape![4, 1, BK])
                .broadcast(shape![4, BN, BK]);
            let weight: Tile<f32, { [BN, BK] }> = convert_tile(wp.load([pid.1, k]));
            sum = sum
                + weight
                    .reshape(shape![1, BN, BK])
                    .broadcast(shape![4, BN, BK])
                    * input;
        }
        let scales = scale
            .partition(shape![BN])
            .load([pid.1])
            .reshape(shape![1, BN])
            .broadcast(shape![4, BN]);
        let reduced: Tile<f32, { [4, BN] }> = reduce_sum(sum, 2i32);
        let reduced = reduced.reshape(shape![4, BN]);
        out.store(reduced * scales);
    }

    #[cutile::entry()]
    fn nvfp4<const BN: i32, const BK: i32, const K: i32, const BP: i32, const BS: i32>(
        out: &mut Tensor<f32, { [4, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
        inverse_global: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![4, BK]);
        let wp = w.partition(shape![BN, BP]);
        let sp = scale.partition(shape![BN, BS]);
        let mut sum: Tile<f32, { [4, BN, BK] }> = constant(0.0f32, shape![4, BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let input = xp
                .load([pid.0, k])
                .reshape(shape![4, 1, BK])
                .broadcast(shape![4, BN, BK]);
            let packed = wp.load([pid.1, k]);
            let weight: Tile<f32, { [BN, BK] }> = convert_tile(packed.unpack(shape![BN, BK]));
            let scales: Tile<f32, { [BN, BS] }> = convert_tile(sp.load([pid.1, k]));
            let scales = scales
                .reshape(shape![BN, BS, 1])
                .broadcast(shape![BN, BS, 16])
                .reshape(shape![BN, BK]);
            let weight = weight * scales;
            sum = sum
                + weight
                    .reshape(shape![1, BN, BK])
                    .broadcast(shape![4, BN, BK])
                    * input;
        }
        let reduced: Tile<f32, { [4, BN] }> = reduce_sum(sum, 2i32);
        let reduced = reduced.reshape(shape![4, BN]);
        out.store(reduced * inverse_global.broadcast(shape![4, BN]));
    }
}
