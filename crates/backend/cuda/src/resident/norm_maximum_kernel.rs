//! Preserve the original RMS normalization and the subsequent padded row maximum.
#[cutile::module]
pub(super) mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock,
        Tensor_1, Tensor_2, Tile_1, absf, convert_scalar, ftz, iota, lt_tile, max_tile, reduce_max,
        reduce_sum, rsqrt, select,
    };
    /// Residual add and RMS normalization fused before the following FP8 projection.
    /// The add is evaluated as one F32 elementwise sum, then the same normalization,
    /// weight/offset and row maximum as `norm_max`.
    #[cutile::entry()]
    fn add_norm_max<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        scale: &mut Tensor<f32, { [1, 1] }>,
        x: &Tensor<f32, { [-1, D] }>,
        residual: &Tensor<f32, { [-1, D] }>,
        weight: &Tensor<f32, { [-1] }>,
        epsilon: f32,
        offset: f32,
    ) {
        let x = x.load_like(out) + residual.load_like(out);
        let square: Tile<f32, { [1] }> = reduce_sum(x * x, 1i32);
        let d: f32 = convert_scalar(D);
        let variance: Tile<f32, { [1] }> =
            square / d.broadcast(shape![1]) + epsilon.broadcast(shape![1]);
        let inverse = rsqrt(variance, ftz::Disabled)
            .reshape(shape![1, 1])
            .broadcast(shape![1, B]);
        let weights = weight
            .partition(shape![B])
            .load([0i32])
            .reshape(shape![1, B]);
        let result = x * inverse * (weights + offset.broadcast(shape![1, B]));
        out.store(result);
        let indices: Tile<i32, { [B] }> = iota(shape![B]);
        let valid = lt_tile(indices, D.broadcast(shape![B])).reshape(shape![1, B]);
        let padded = select(valid, result, 0.0f32.broadcast(shape![1, B]));
        let maximum: Tile<f32, { [1] }> = reduce_max(absf(padded), 1i32);
        scale.store(max_tile(
            maximum.reshape(shape![1, 1]) / 448.0f32.broadcast(shape![1, 1]),
            1.0e-12f32.broadcast(shape![1, 1]),
        ));
    }

    #[cutile::entry()]
    fn norm_max<const D: i32, const B: i32>(
        out: &mut Tensor<f32, { [1, B] }>,
        scale: &mut Tensor<f32, { [1, 1] }>,
        x: &Tensor<f32, { [-1, D] }>,
        weight: &Tensor<f32, { [-1] }>,
        epsilon: f32,
        offset: f32,
    ) {
        let x = x.load_like(out);
        let square: Tile<f32, { [1] }> = reduce_sum(x * x, 1i32);
        let d: f32 = convert_scalar(D);
        let variance: Tile<f32, { [1] }> =
            square / d.broadcast(shape![1]) + epsilon.broadcast(shape![1]);
        let inverse = rsqrt(variance, ftz::Disabled)
            .reshape(shape![1, 1])
            .broadcast(shape![1, B]);
        let weights = weight
            .partition(shape![B])
            .load([0i32])
            .reshape(shape![1, B]);
        let result = x * inverse * (weights + offset.broadcast(shape![1, B]));
        out.store(result);
        let indices: Tile<i32, { [B] }> = iota(shape![B]);
        let valid = lt_tile(indices, D.broadcast(shape![B])).reshape(shape![1, B]);
        let padded = select(valid, result, 0.0f32.broadcast(shape![1, B]));
        let maximum: Tile<f32, { [1] }> = reduce_max(absf(padded), 1i32);
        scale.store(max_tile(
            maximum.reshape(shape![1, 1]) / 448.0f32.broadcast(shape![1, 1]),
            1.0e-12f32.broadcast(shape![1, 1]),
        ));
    }
}
