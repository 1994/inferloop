//! Scalar launch arguments update resident metadata without temporary device allocations.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, Shape_1, StoreTileAtCurrentBlock, Tensor_1, Tile_1, bitcast, eq_tile,
        iota, select,
    };

    #[cutile::entry()]
    fn write(out: &mut Tensor<i32, { [4] }>, position: f32, token: f32, kv: f32, external: f32) {
        let offsets: Tile<i32, { [4] }> = iota(shape![4]);
        let position: Tile<i32, { [4] }> = bitcast(position.broadcast(shape![4]));
        let token: Tile<i32, { [4] }> = bitcast(token.broadcast(shape![4]));
        let kv: Tile<i32, { [4] }> = bitcast(kv.broadcast(shape![4]));
        let values: Tile<i32, { [4] }> = bitcast(external.broadcast(shape![4]));
        let values = select(eq_tile(offsets, 2i32.broadcast(shape![4])), kv, values);
        let values = select(eq_tile(offsets, 1i32.broadcast(shape![4])), token, values);
        let values = select(
            eq_tile(offsets, 0i32.broadcast(shape![4])),
            position,
            values,
        );
        out.store(values);
    }
    #[cutile::entry()]
    fn write_rotary(
        out: &mut Tensor<i32, { [8] }>,
        position: f32,
        token: f32,
        kv: f32,
        external: f32,
        height: f32,
        width: f32,
    ) {
        let offsets: Tile<i32, { [8] }> = iota(shape![8]);
        let position_values: Tile<i32, { [8] }> = bitcast(position.broadcast(shape![8]));
        let token_values: Tile<i32, { [8] }> = bitcast(token.broadcast(shape![8]));
        let kv_values: Tile<i32, { [8] }> = bitcast(kv.broadcast(shape![8]));
        let external_values: Tile<i32, { [8] }> = bitcast(external.broadcast(shape![8]));
        let height_values: Tile<i32, { [8] }> = bitcast(height.broadcast(shape![8]));
        let width_values: Tile<i32, { [8] }> = bitcast(width.broadcast(shape![8]));
        let result = select(
            eq_tile(offsets, 1i32.broadcast(shape![8])),
            token_values,
            position_values,
        );
        let result = select(
            eq_tile(offsets, 2i32.broadcast(shape![8])),
            kv_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 3i32.broadcast(shape![8])),
            external_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 5i32.broadcast(shape![8])),
            height_values,
            result,
        );
        let result = select(
            eq_tile(offsets, 6i32.broadcast(shape![8])),
            width_values,
            result,
        );
        out.store(result);
    }
}
