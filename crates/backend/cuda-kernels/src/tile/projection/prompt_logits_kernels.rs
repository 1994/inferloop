//! Prompt graphs project only the last active lane onto the vocabulary.
//!
//! A prompt chunk reads back one row of logits, so its arena keeps one row (see
//! [`super::ActivationArena::new_prompt`]) instead of `lanes x vocabulary`, which is most of a
//! wide prompt arena (2 GB at 2048 lanes for a 248K vocabulary).
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1, Tensor_2,
        get_tile_block_id, tile_to_scalar,
    };

    /// Copy row `count - 1` of a prompt chunk, 256 columns per block.
    #[cutile::entry()]
    fn last_row<const H: i32>(
        out: &mut Tensor<f32, { [1, 256] }>,
        rows: &Tensor<f32, { [-1, H] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let count: i32 =
            tile_to_scalar(metadata.partition(shape![1]).load([1i32]).reshape(shape![]));
        out.store(rows.partition(shape![1, 256]).load([count - 1i32, pid.1]));
    }
}
