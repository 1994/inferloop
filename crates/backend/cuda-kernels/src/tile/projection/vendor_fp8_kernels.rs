//! FP8 channel projections through cuBLASLt with the existing quantization.
//! Scalar matrix scales stay one; the exact per-token/per-channel factors are applied afterwards.
//! Descriptors and zero-workspace algorithms are warmed before graph capture. Native kernels
//! remain the fallback for an absent library, unsupported shape or failed exact tuning gate.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        LoadTileMutAtCurrentBlock, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tile_2, ftz, get_tile_block_id, mulf, rounding,
    };
    #[cutile::entry()]
    fn scale(
        output: &mut Tensor<f32, { [1, 256] }>,
        xs: &Tensor<f32, { [-1, 1] }>,
        ws: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let raw: Tile<f32, { [1, 256] }> = output.load();
        let row = xs
            .partition(shape![1, 1])
            .load([pid.0, 0i32])
            .broadcast(shape![1, 256]);
        let column = ws
            .partition(shape![256])
            .load([pid.1])
            .reshape(shape![1, 256]);
        let first = mulf(raw, row, rounding::NearestEven, ftz::Disabled);
        output.store(mulf(first, column, rounding::NearestEven, ftz::Disabled));
    }
}
