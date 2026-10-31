#[cutile::module]
pub mod fused {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_0, Shape_1, StoreTileAtCurrentBlock,
        Tensor_1, Tile_0, Tile_1, convert_scalar, exp, ftz, reduce_sum, rsqrt,
    };

    #[cutile::entry()]
    fn norm<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        weight: &Tensor<f32, { [-1] }>,
        epsilon: f32,
        offset: f32,
    ) {
        let x = x.load_like(out);
        let sum: Tile<f32, { [] }> = reduce_sum(x * x, 0i32);
        let dim: f32 = convert_scalar(D);
        let variance: Tile<f32, { [] }> =
            sum / dim.broadcast(shape![]) + epsilon.broadcast(shape![]);
        let inverse = rsqrt(variance, ftz::Disabled);
        let scaled: Tile<f32, { [B] }> = x * inverse.reshape(shape![1]).broadcast(shape![B]);
        out.store(scaled * (weight.load_like(out) + offset.broadcast(shape![B])));
    }

    #[cutile::entry()]
    fn silu_mul<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        gate: &Tensor<f32, { [-1] }>,
        up: &Tensor<f32, { [-1] }>,
    ) {
        let x = gate.load_like(out);
        let negative: Tile<f32, { [B] }> = 0.0f32.broadcast(shape![B]) - x;
        let denominator = 1.0f32.broadcast(shape![B]) + exp(negative);
        out.store(x / denominator * up.load_like(out));
    }

    #[cutile::entry()]
    fn residual<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        residual: &Tensor<f32, { [-1] }>,
    ) {
        out.store(x.load_like(out) + residual.load_like(out));
    }
}
