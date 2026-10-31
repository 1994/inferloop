//! Exact finite F32 greedy selection; only token IDs and validation flags leave the GPU.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_0,
        Tile_1, Tile_2, andi, bitcast, cat, eq_tile, get_tile_block_id, iota, lt_tile, reduce_max,
        reduce_min, select, xori,
    };

    // Flip the sign bit for nonnegative values and all bits for negative values.
    const SIGN_SHIFT: u32 = 31;
    const MAGNITUDE_MASK: u32 = 0x7fff_ffff;
    const NONFINITE_BITS: u32 = 0x7f80_0000;
    const UNUSED_INDEX: i32 = 2_147_483_647;

    #[cutile::entry()]
    pub fn partials<const V: i32, const B: i32>(
        best_keys: &mut Tensor<u32, { [1, 1] }>,
        indices: &mut Tensor<i32, { [1, 1] }>,
        invalid: &mut Tensor<i32, { [1, 1] }>,
        logits: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let ids: Tile<i32, { [B] }> = iota(shape![B]) + (pid.1 * B).broadcast(shape![B]);
        let values = logits
            .partition(shape![1, B])
            .load([pid.0, pid.1])
            .reshape(shape![B]);
        let bits: Tile<u32, { [B] }> = bitcast(values);
        let sign_bit = 1u32.broadcast(shape![B]) << SIGN_SHIFT.broadcast(shape![B]);
        let all_bits = 0u32.broadcast(shape![B]) - 1u32.broadcast(shape![B]);
        let positive = lt_tile(bits, sign_bit);
        let keys = select(positive, xori(bits, sign_bit), xori(bits, all_bits));
        let present = lt_tile(ids, V.broadcast(shape![B]));
        let masked = select(present, keys, 0u32.broadcast(shape![B]));
        let best: Tile<u32, { [] }> = reduce_max(masked, 0i32);
        let tied = eq_tile(keys, best.reshape(shape![1]).broadcast(shape![B]));
        let candidates = select(
            present,
            select(tied, ids, UNUSED_INDEX.broadcast(shape![B])),
            UNUSED_INDEX.broadcast(shape![B]),
        );
        let token: Tile<i32, { [] }> = reduce_min(candidates, 0i32);
        let abs = andi(bits, MAGNITUDE_MASK.broadcast(shape![B]));
        let finite = lt_tile(abs, NONFINITE_BITS.broadcast(shape![B]));
        let bad = select(
            present,
            select(finite, 0i32.broadcast(shape![B]), 1i32.broadcast(shape![B])),
            0i32.broadcast(shape![B]),
        );
        let flag: Tile<i32, { [] }> = reduce_max(bad, 0i32);
        best_keys.store(best.reshape(shape![1, 1]));
        indices.store(token.reshape(shape![1, 1]));
        invalid.store(flag.reshape(shape![1, 1]));
    }

    #[cutile::entry()]
    pub fn finish<const N: i32, const B: i32, const START: i32, const ROWS: i32>(
        output: &mut Tensor<i32, { [1, 2] }>,
        keys: &Tensor<u32, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1, -1] }>,
        invalid: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        if pid.0 >= START && pid.0 < START + ROWS {
            // Modulo preserves the active-row index and expresses its bounded extent
            // to the launch checker when START=0 and the shared output is wider.
            let row = (pid.0 - START) % ROWS;
            let ids: Tile<i32, { [B] }> = iota(shape![B]);
            let present = lt_tile(ids, N.broadcast(shape![B]));
            let values = keys
                .partition(shape![1, B])
                .load([row, 0i32])
                .reshape(shape![B]);
            let masked = select(present, values, 0u32.broadcast(shape![B]));
            let best: Tile<u32, { [] }> = reduce_max(masked, 0i32);
            let candidates = indices
                .partition(shape![1, B])
                .load([row, 0i32])
                .reshape(shape![B]);
            let tied = eq_tile(values, best.reshape(shape![1]).broadcast(shape![B]));
            let candidates = select(
                present,
                select(tied, candidates, UNUSED_INDEX.broadcast(shape![B])),
                UNUSED_INDEX.broadcast(shape![B]),
            );
            let token: Tile<i32, { [] }> = reduce_min(candidates, 0i32);
            let flags = invalid
                .partition(shape![1, B])
                .load([row, 0i32])
                .reshape(shape![B]);
            let flag: Tile<i32, { [] }> = reduce_max(flags, 0i32);
            let result: Tile<i32, { [1, 2] }> = cat(
                token.reshape(shape![1, 1]),
                flag.reshape(shape![1, 1]),
                1i32,
            );
            output.store(result);
        }
    }
}
