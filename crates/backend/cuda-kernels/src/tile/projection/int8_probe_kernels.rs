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
