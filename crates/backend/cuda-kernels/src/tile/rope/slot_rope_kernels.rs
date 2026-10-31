//! One rotary launch across verification rows, using each row's original position.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_0, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tile_1, convert_tile, cos, get_tile_block_id, sin, tile_to_scalar,
    };

    /// Same expressions as the existing per-lane rotary kernel; only the position binding varies across heads.
    #[cutile::entry()]
    fn rope<const D: i32, const HALF: i32, const HEADS: i32, const MF: i32>(
        out: &mut Tensor<f32, { [1, HALF] }>,
        x: &Tensor<f32, { [-1, D] }>,
        frequencies: &Tensor<f32, { [HALF] }>,
        metadata: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![1, HALF]);
        let value = xp.load([pid.0, pid.1]);
        if pid.1 < 2 {
            let p: Tile<f32, { [1] }> =
                convert_tile(metadata.partition(shape![1]).load([pid.0 / HEADS * MF]));
            let position: f32 = tile_to_scalar(p.reshape(shape![]));
            let angles =
                frequencies.partition(shape![HALF]).load([0i32]) * position.broadcast(shape![HALF]);
            let c = cos(angles).reshape(shape![1, HALF]);
            let s = sin(angles).reshape(shape![1, HALF]);
            let other = xp.load([pid.0, 1i32 - pid.1]);
            if pid.1 == 0 {
                out.store(value * c - other * s);
            } else {
                out.store(value * c + other * s);
            }
        } else {
            out.store(value);
        }
    }
}
