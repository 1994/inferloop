//! Preserve RMS F32 output while writing its block-16 NVFP4 activation codes.
#[cutile::module]
pub mod kernels {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, PackF4e2m1fnx2Tile, Reshape, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tile_1, Tile_2, absf, convert_scalar,
        convert_tile, eq_tile, f4e2m1fn, f4e2m1fnx2, f8e4m3fn, ftz, max_tile, min_tile, reduce_max,
        reduce_sum, rsqrt, select,
    };
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit RMS and packed FP4 output bindings"
    )]
    #[cutile::entry()]
    pub fn norm_fp4<const D: i32, const B: i32, const G: i32, const H: i32>(
        dense: &mut Tensor<f32, { [1, B] }>,
        out: &mut Tensor<f4e2m1fnx2, { [1, H] }>,
        scales: &mut Tensor<f8e4m3fn, { [1, G] }>,
        input: &Tensor<f32, { [-1, D] }>,
        weight: &Tensor<f32, { [-1] }>,
        epsilon: f32,
        offset: f32,
        input_global: f32,
    ) {
        let x = input.load_like(dense);
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
        dense.store(result);
        let groups = result.reshape(shape![1, G, 16]);
        let maximum: Tile<f32, { [1, G] }> = reduce_max(absf(groups), 2i32);
        let maximum: Tile<f32, { [1, G] }> = maximum.reshape(shape![1, G]);
        let factor: Tile<f32, { [1, G] }> = (input_global / 6.0f32).broadcast(shape![1, G]);
        let limit: Tile<f32, { [1, G] }> = 448.0f32.broadcast(shape![1, G]);
        let product: Tile<f32, { [1, G] }> = maximum * factor;
        let scale = min_tile(product, limit);
        let encoded: Tile<f8e4m3fn, { [1, G] }> = convert_tile(scale);
        let rounded: Tile<f32, { [1, G] }> = convert_tile(encoded);
        let zero = eq_tile(rounded, 0.0f32.broadcast(shape![1, G]));
        let denominator = select(zero, 1.0f32.broadcast(shape![1, G]), rounded);
        let normalized = groups
            * (input_global.broadcast(shape![1, G]) / denominator)
                .reshape(shape![1, G, 1])
                .broadcast(shape![1, G, 16]);
        let clipped = min_tile(
            max_tile(normalized, (-6.0f32).broadcast(shape![1, G, 16])),
            6.0f32.broadcast(shape![1, G, 16]),
        );
        let values: Tile<f4e2m1fn, { [1, B] }> = convert_tile(clipped.reshape(shape![1, B]));
        out.store(values.pack(shape![1, H]));
        scales.store(encoded);
    }
}
