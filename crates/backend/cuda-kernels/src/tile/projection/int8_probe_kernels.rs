//! int8 MMA probe used to validate the cuTile `mmai` path before `Q4_K` MMQ.
#[cutile::module]
pub mod int8_probe {
    use cutile::core::{
        Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_2, constant, mmai, signedness,
    };

    #[cutile::entry()]
    fn probe_mma(
        lhs: &Tensor<i8, { [16, 64] }>,
        rhs: &Tensor<i8, { [64, 16] }>,
        out: &mut Tensor<i32, { [16, 16] }>,
    ) {
        let lhs: Tile<i8, { [16, 64] }> = lhs.partition(shape![16, 64]).load([0, 0]);
        let rhs: Tile<i8, { [64, 16] }> = rhs.partition(shape![64, 16]).load([0, 0]);
        let zero: Tile<i32, { [16, 16] }> = constant(0i32, shape![16, 16]);
        let acc: Tile<i32, { [16, 16] }> =
            mmai(lhs, rhs, zero, signedness::Signed, signedness::Signed);
        out.store(acc);
    }
}

/// Q4_K two-term int8 MMQ: one int8 main term plus one int8 residual term.
#[cutile::module]
pub mod q4k_int8_two {
    use cutile::core::{
        Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_2, constant, convert_tile,
        get_tile_block_id, mmai, signedness,
    };

    #[cutile::entry()]
    fn probe<const GROUPS: i32>(
        out: &mut Tensor<f32, { [16, 16] }>,
        x: &Tensor<i8, { [16, -1] }>,
        residual: &Tensor<i8, { [16, -1] }>,
        codes: &Tensor<i8, { [-1, 16] }>,
        factors: &Tensor<f32, { [-1, 16] }>,
        biases: &Tensor<f32, { [-1, 16] }>,
        scales: &Tensor<f32, { [-1, 16] }>,
        residual_scales: &Tensor<f32, { [-1, 16] }>,
        sums: &Tensor<f32, { [-1, 16] }>,
    ) {
        let xp = x.partition(shape![16, 32]);
        let rp = residual.partition(shape![16, 32]);
        let cp = codes.partition(shape![32, 16]);
        let fp = factors.partition(shape![1, 16]);
        let bp = biases.partition(shape![1, 16]);
        let sp = scales.partition(shape![1, 16]);
        let rsp = residual_scales.partition(shape![1, 16]);
        let up = sums.partition(shape![1, 16]);
        let pid = get_tile_block_id().1;
        let mut sum: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
        for group in 0i32..GROUPS {
            let zero: Tile<i32, { [16, 16] }> = constant(0i32, shape![16, 16]);
            let main: Tile<i32, { [16, 16] }> = mmai(
                xp.load([0, group]),
                cp.load([pid * GROUPS + group, 0]),
                zero,
                signedness::Signed,
                signedness::Signed,
            );
            let fine: Tile<i32, { [16, 16] }> = mmai(
                rp.load([0, group]),
                cp.load([pid * GROUPS + group, 0]),
                zero,
                signedness::Signed,
                signedness::Signed,
            );
            let main: Tile<f32, { [16, 16] }> = convert_tile(main);
            let fine: Tile<f32, { [16, 16] }> = convert_tile(fine);
            let factor: Tile<f32, { [1, 16] }> = fp.load([pid * GROUPS + group, 0]);
            let bias: Tile<f32, { [1, 16] }> = bp.load([pid * GROUPS + group, 0]);
            let scale: Tile<f32, { [1, 16] }> = sp.load([pid * GROUPS + group, 0]);
            let fine_scale: Tile<f32, { [1, 16] }> = rsp.load([pid * GROUPS + group, 0]);
            let x_sum: Tile<f32, { [1, 16] }> = up.load([pid * GROUPS + group, 0]);
            let products = main * scale.transpose().broadcast(shape![16, 16])
                + fine * fine_scale.transpose().broadcast(shape![16, 16]);
            let weighted = products * factor.broadcast(shape![16, 16]);
            let correction =
                bias.broadcast(shape![16, 16]) * x_sum.transpose().broadcast(shape![16, 16]);
            sum = sum + weighted - correction;
        }
        out.store(sum);
    }
}
