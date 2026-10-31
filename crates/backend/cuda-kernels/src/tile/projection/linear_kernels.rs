//! Safe tile kernels compiled to native GPU code by NVIDIA cuTile Rust.
#[cutile::module]
pub mod linear {
    use cutile::core::{
        BroadcastScalar, ElementType, LoadTileLike, Reshape, Shape_1, Shape_2, Shape_3,
        StoreTileAtCurrentBlock, Tensor_1, Tensor_2, Tile_1, Tile_2, UnpackF4e2m1fnx2Tile, bf16,
        constant, convert_tile, f4e2m1fnx2, f8e4m3fn, get_tile_block_id, reduce_sum,
    };

    /// Narrow an F32 activation tile to the BF16 a vendor GEMM consumes. Both operands share a
    /// row-major layout, so this is a straight elementwise conversion over flat tiles.
    #[cutile::entry()]
    fn cast_bf16<const B: i32>(out: &mut Tensor<bf16, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let narrowed: Tile<bf16, { [B] }> = convert_tile(x.load_like(out));
        out.store(narrowed);
    }

    #[cutile::entry()]
    fn dense<E: ElementType, const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [K] }>,
        w: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![BK]);
        let wp = w.partition(shape![BN, BK]);
        let mut sum: Tile<f32, { [BN, BK] }> = constant(0.0, shape![BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let a = xp.load([k]).reshape(shape![1, BK]);
            let b: Tile<f32, { [BN, BK] }> = convert_tile(wp.load([pid.0, k]));
            sum = sum + b * a.broadcast(shape![BN, BK]);
        }
        let reduced: Tile<f32, { [BN] }> = reduce_sum(sum, 1i32);
        out.store(reduced);
    }

    #[cutile::entry()]
    fn fp8<const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [K] }>,
        w: &Tensor<f8e4m3fn, { [-1, K] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![BK]);
        let wp = w.partition(shape![BN, BK]);
        let mut sum: Tile<f32, { [BN, BK] }> = constant(0.0, shape![BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let a = xp.load([k]).reshape(shape![1, BK]);
            let b: Tile<f32, { [BN, BK] }> = convert_tile(wp.load([pid.0, k]));
            sum = sum + b * a.broadcast(shape![BN, BK]);
        }
        let reduced: Tile<f32, { [BN] }> = reduce_sum(sum, 1i32);
        out.store(reduced * scale.load_like(out));
    }

    #[cutile::entry()]
    fn nvfp4<const BN: i32, const BK: i32, const K: i32, const BP: i32, const BS: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [K] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
        inverse_global: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = x.partition(shape![BK]);
        let wp = w.partition(shape![BN, BP]);
        let sp = scale.partition(shape![BN, BS]);
        let mut sum: Tile<f32, { [BN, BK] }> = constant(0.0, shape![BN, BK]);
        for k in 0i32..((K + BK - 1) / BK) {
            let a = xp.load([k]).reshape(shape![1, BK]);
            let packed = wp.load([pid.0, k]);
            let b: Tile<f32, { [BN, BK] }> = convert_tile(packed.unpack(shape![BN, BK]));
            let scales: Tile<f32, { [BN, BS] }> = convert_tile(sp.load([pid.0, k]));
            let scales = scales
                .reshape(shape![BN, BS, 1])
                .broadcast(shape![BN, BS, 16])
                .reshape(shape![BN, BK]);
            sum = sum + b * scales * a.broadcast(shape![BN, BK]);
        }
        let reduced: Tile<f32, { [BN] }> = reduce_sum(sum, 1i32);
        out.store(reduced * inverse_global.broadcast(shape![BN]));
    }
}
