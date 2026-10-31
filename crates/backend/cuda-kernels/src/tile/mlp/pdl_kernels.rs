//! Audited host/device boundary for one PDL edge: `SwiGLU` -> down projection.
#![expect(
    unsafe_code,
    reason = "PDL launch ordering is a device-side handshake; the graph owns all buffers through completion"
)]
#[cutile::module]
pub mod producer {
    use cutile::core::{
        BroadcastScalar, LoadTileLike, Shape_1, StoreTileAtCurrentBlock, Tensor_1, Tile_1, exp,
        gdc_launch_dependents_tko,
    };
    #[cutile::entry()]
    fn silu_mul(
        out: &mut Tensor<f32, { [256] }>,
        gate: &Tensor<f32, { [-1] }>,
        up: &Tensor<f32, { [-1] }>,
    ) {
        let x = gate.load_like(out);
        let negative: Tile<f32, { [256] }> = 0.0f32.broadcast(shape![256]) - x;
        let denominator = 1.0f32.broadcast(shape![256]) + exp(negative);
        let stored = out.store(x / denominator * up.load_like(out));
        // SAFETY: paired consumers token-order all activation reads after gdc_wait.
        let _signal = unsafe { gdc_launch_dependents_tko(Some(stored)) };
    }
}
