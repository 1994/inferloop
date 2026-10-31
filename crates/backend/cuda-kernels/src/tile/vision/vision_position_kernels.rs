//! Device-side interpolation of the learned vision position table.
#![expect(
    unsafe_code,
    reason = "The position gather validates every table index and owns all device buffers through graph completion"
)]
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, Latency, PointerTile_0, PointerTile_2, Reshape, ReshapePtr, Shape_1,
        Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_1, Tile_2, Token, addptr_tile, bf16,
        convert_tile, fma, ftz, get_tile_block_id, iota, load_ptr_tko, ordering, pointer_to_tile,
        rounding, scope,
    };

    /// Add a four-tap interpolation to a patch projection without a host round trip.
    ///
    /// # Safety
    /// Table indices and all tensor shapes must be validated and their owners retained until the
    /// recorded device work completes.
    #[cutile::entry()]
    unsafe fn add_positions<const H: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        embedded: &Tensor<f32, { [-1, H] }>,
        table: &Tensor<bf16, { [-1, H] }>,
        indices: &Tensor<i32, { [4, -1] }>,
        weights: &Tensor<f32, { [4, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let base: PointerTile<*const bf16, { [] }> = pointer_to_tile(table.as_ptr());
        let base: PointerTile<*const bf16, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*const bf16, { [32, 64] }> = base.broadcast(shape![32, 64]);
        let columns: Tile<i32, { [64] }> = iota(shape![64]) + (pid.1 * 64).broadcast(shape![64]);
        let columns: Tile<i32, { [1, 64] }> = columns.reshape(shape![1, 64]);
        let columns: Tile<i32, { [32, 64] }> = columns.broadcast(shape![32, 64]);
        let mut sum: Tile<f32, { [32, 64] }> = 0.0f32.broadcast(shape![32, 64]);
        for tap in 0i32..4i32 {
            let rows: Tile<i32, { [1, 32] }> = indices.partition(shape![1, 32]).load([tap, pid.0]);
            let rows: Tile<i32, { [32, 1] }> = rows.reshape(shape![32, 1]);
            let rows: Tile<i32, { [32, 64] }> = rows.broadcast(shape![32, 64]);
            let offsets: Tile<i32, { [32, 64] }> = rows * H.broadcast(shape![32, 64]) + columns;
            let addresses: PointerTile<*const bf16, { [32, 64] }> = addptr_tile(base, offsets);
            // SAFETY: the host validates every row and column address; all padded rows target
            // table row zero, and the source tensor remains live through graph completion.
            let (values, _): (Tile<bf16, { [32, 64] }>, Token) = unsafe {
                load_ptr_tko(
                    addresses,
                    ordering::Relaxed,
                    Some(scope::Device),
                    None,
                    None,
                    None,
                    Latency::<0>,
                )
            };
            let weights: Tile<f32, { [1, 32] }> =
                weights.partition(shape![1, 32]).load([tap, pid.0]);
            let weights: Tile<f32, { [32, 1] }> = weights.reshape(shape![32, 1]);
            let weights: Tile<f32, { [32, 64] }> = weights.broadcast(shape![32, 64]);
            let values: Tile<f32, { [32, 64] }> = convert_tile(values);
            sum = fma(weights, values, sum, rounding::NearestEven, ftz::Disabled);
        }
        let embedded: Tile<f32, { [32, 64] }> =
            embedded.partition(shape![32, 64]).load([pid.0, pid.1]);
        out.store(embedded + sum);
    }
}
