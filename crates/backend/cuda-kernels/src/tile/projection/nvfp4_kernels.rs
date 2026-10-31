//! Pack unchanged positive E4M3 bits into the SDK scale layout.
#[cutile::module]
pub mod entries {
    use cutile::core::{
        Array_3, BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_2, Tile_1, Tile_2, Tile_3, convert_tile, f8e4m3fn, get_tile_block_id, iota, permute,
    };
    #[cutile::entry()]
    pub fn pack_pair<const N: i32>(
        out: &mut Tensor<f8e4m3fn, { [1, 512] }>,
        weights: &Tensor<f8e4m3fn, { [-1, -1] }>,
        input: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let raw: Tile<f8e4m3fn, { [128, 4] }> = if pid.0 < N / 128i32 {
            // Modulo preserves this branch's index and exposes its bound to launch validation.
            let values: Tile<f8e4m3fn, { [128, 4] }> = weights
                .partition(shape![128, 4])
                .load([pid.0 % (N / 128i32), pid.1]);
            values
        } else {
            let values: Tile<f8e4m3fn, { [128, 4] }> = input
                .partition(shape![128, 4])
                .load([pid.0 - N / 128i32, pid.1]);
            values
        };
        let tile: Tile<f8e4m3fn, { [4, 32, 4] }> = raw.reshape(shape![4, 32, 4]);
        let packed: Tile<f8e4m3fn, { [32, 4, 4] }> = permute(tile, const_array![1, 0, 2]);
        out.store(packed.reshape(shape![1, 512]));
    }
    /// Deterministic signed probe; the product slot is unused after its codes are prepared.
    #[cutile::entry()]
    pub fn probe_input<const K: i32>(out: &mut Tensor<f32, { [16, 512] }>) {
        let pid = get_tile_block_id();
        let column: Tile<i32, { [512] }> = iota(shape![512]);
        let row: Tile<i32, { [16] }> = iota(shape![16]);
        let r = (row + (pid.0 * 16i32).broadcast(shape![16]))
            .reshape(shape![16, 1])
            .broadcast(shape![16, 512]);
        let c = (column + (pid.1 * 512i32).broadcast(shape![512]))
            .reshape(shape![1, 512])
            .broadcast(shape![16, 512]);
        let index = (r * K.broadcast(shape![16, 512]) + c) * 17i32.broadcast(shape![16, 512])
            % 131i32.broadcast(shape![16, 512]);
        let values: Tile<f32, { [16, 512] }> = convert_tile(index);
        out.store(
            (values - 65.0f32.broadcast(shape![16, 512])) / 37.0f32.broadcast(shape![16, 512]),
        );
    }
}
