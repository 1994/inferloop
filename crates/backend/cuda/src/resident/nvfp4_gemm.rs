//! Model-declared W4A4 projection with block-16 E4M3 activation scales.
mod quant_reuse;
mod workspace;
use crate::mlp::ProjectionWeight as MlpProjectionWeight;
use cutile::prelude::DeviceError;
use cutile::prelude::Scope;
use cutile::prelude::Tensor as PreludeTensor;
use cutile::prelude::TensorView;
pub use quant_reuse::QuantPairPlan;
#[cfg(test)]
pub use quant_reuse::with_quant_reuse_disabled;
pub use workspace::Workspace;
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, PackF4e2m1fnx2Tile, Reshape, Shape_2, Shape_3, StoreTileAtCurrentBlock,
        Tensor_2, Tensor_3, Tile_2, Tile_3, UnpackF4e2m1fnx2Tile, absf, constant, convert_tile,
        eq_tile, exp, f4e2m1fn, f4e2m1fnx2, f8e4m3fn, get_tile_block_id, max_tile, min_tile,
        mmaf_scaled, reduce_max, select,
    };

    /// Fused activation quantization, kept at a fixed 16-row tile. The production W4A4 path
    /// quantizes once per program and then runs the row-tiled [`packed`] kernel; this fused
    /// variant only backs the fused-versus-separated measurement in `tests`.
    #[cutile::entry()]
    fn matmul<const K: i32>(
        out: &mut Tensor<f32, { [16, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        input_global: f32,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 512]);
        let wp = weight.partition(shape![64, 256]);
        let sp = scales.partition(shape![64, 32]);
        let mut acc: Tile<f32, { [16, 64] }> = constant(0.0f32, shape![16, 64]);
        for k in 0i32..((K + 511) / 512) {
            let x = xp.load([pid.0, k]);
            let groups = x.reshape(shape![16, 32, 16]);
            let maximum: Tile<f32, { [16, 32] }> = reduce_max(absf(groups), 2i32);
            let scale = min_tile(
                maximum * (input_global / 6.0f32).broadcast(shape![16, 32]),
                448.0f32.broadcast(shape![16, 32]),
            );
            let encoded: Tile<f8e4m3fn, { [16, 32] }> = convert_tile(scale);
            let rounded: Tile<f32, { [16, 32] }> = convert_tile(encoded);
            let zero = eq_tile(rounded, 0.0f32.broadcast(shape![16, 32]));
            let denominator = select(zero, 1.0f32.broadcast(shape![16, 32]), rounded);
            let multiplier = input_global.broadcast(shape![16, 32]) / denominator;
            let normalized = groups
                * multiplier
                    .reshape(shape![16, 32, 1])
                    .broadcast(shape![16, 32, 16]);
            let clipped = min_tile(
                max_tile(normalized, (-6.0f32).broadcast(shape![16, 32, 16])),
                6.0f32.broadcast(shape![16, 32, 16]),
            );
            let quantized: Tile<f4e2m1fn, { [16, 512] }> =
                convert_tile(clipped.reshape(shape![16, 512]));
            let w = wp.load([pid.1, k]).unpack(shape![64, 512]).transpose();
            let ws = sp.load([pid.1, k]).transpose();
            acc = mmaf_scaled(quantized, w, acc, encoded, ws);
        }
        out.store(acc * alpha.broadcast(shape![16, 64]));
    }
    /// Quantize each activation group once, shared by all output-column tiles.
    #[cutile::entry()]
    fn quantize<const K: i32>(
        out: &mut Tensor<f4e2m1fnx2, { [1, 256] }>,
        scales: &mut Tensor<f8e4m3fn, { [1, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        input_global: f32,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![1, 512]).load([pid.0, pid.1]);
        let groups = x.reshape(shape![1, 32, 16]);
        let maximum: Tile<f32, { [1, 32] }> = reduce_max(absf(groups), 2i32);
        let scale = min_tile(
            maximum * (input_global / 6.0f32).broadcast(shape![1, 32]),
            448.0f32.broadcast(shape![1, 32]),
        );
        let encoded: Tile<f8e4m3fn, { [1, 32] }> = convert_tile(scale);
        let rounded: Tile<f32, { [1, 32] }> = convert_tile(encoded);
        let zero = eq_tile(rounded, 0.0f32.broadcast(shape![1, 32]));
        let denominator = select(zero, 1.0f32.broadcast(shape![1, 32]), rounded);
        let normalized = groups
            * (input_global.broadcast(shape![1, 32]) / denominator)
                .reshape(shape![1, 32, 1])
                .broadcast(shape![1, 32, 16]);
        let clipped = min_tile(
            max_tile(normalized, (-6.0f32).broadcast(shape![1, 32, 16])),
            6.0f32.broadcast(shape![1, 32, 16]),
        );
        let values: Tile<f4e2m1fn, { [1, 512] }> = convert_tile(clipped.reshape(shape![1, 512]));
        out.store(values.pack(shape![1, 256]));
        scales.store(encoded);
    }

    /// [`quantize`] over sixteen rows of one 512-column window per block, for prompt graphs.
    #[cutile::entry()]
    fn quantize_rows<const K: i32>(
        out: &mut Tensor<f4e2m1fnx2, { [16, 256] }>,
        scales: &mut Tensor<f8e4m3fn, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        input_global: f32,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![16, 512]).load([pid.0, pid.1]);
        let groups = x.reshape(shape![16, 32, 16]);
        let maximum: Tile<f32, { [16, 32] }> = reduce_max(absf(groups), 2i32);
        let scale = min_tile(
            maximum * (input_global / 6.0f32).broadcast(shape![16, 32]),
            448.0f32.broadcast(shape![16, 32]),
        );
        let encoded: Tile<f8e4m3fn, { [16, 32] }> = convert_tile(scale);
        let rounded: Tile<f32, { [16, 32] }> = convert_tile(encoded);
        let zero = eq_tile(rounded, 0.0f32.broadcast(shape![16, 32]));
        let denominator = select(zero, 1.0f32.broadcast(shape![16, 32]), rounded);
        let normalized = groups
            * (input_global.broadcast(shape![16, 32]) / denominator)
                .reshape(shape![16, 32, 1])
                .broadcast(shape![16, 32, 16]);
        let clipped = min_tile(
            max_tile(normalized, (-6.0f32).broadcast(shape![16, 32, 16])),
            6.0f32.broadcast(shape![16, 32, 16]),
        );
        let values: Tile<f4e2m1fn, { [16, 512] }> = convert_tile(clipped.reshape(shape![16, 512]));
        out.store(values.pack(shape![16, 256]));
        scales.store(encoded);
    }

    /// `silu(gate) * up` quantized like [`quantize`], so a gated MLP's down projection reads its
    /// activation codes without materializing the `SiLU` and product rows. The `SiLU` and product
    /// keep the element-wise kernels' arithmetic: `g / (1 + exp(-g))`, then one multiply. Each
    /// block covers sixteen rows of one 512-column window.
    #[cutile::entry()]
    fn swiglu_quantize<const K: i32>(
        out: &mut Tensor<f4e2m1fnx2, { [16, 256] }>,
        scales: &mut Tensor<f8e4m3fn, { [16, 32] }>,
        gate: &Tensor<f32, { [-1, K] }>,
        up: &Tensor<f32, { [-1, K] }>,
        input_global: f32,
    ) {
        let pid = get_tile_block_id();
        let g = gate.partition(shape![16, 512]).load([pid.0, pid.1]);
        let u = up.partition(shape![16, 512]).load([pid.0, pid.1]);
        let negative: Tile<f32, { [16, 512] }> = 0.0f32.broadcast(shape![16, 512]) - g;
        let activated = g / (1.0f32.broadcast(shape![16, 512]) + exp(negative));
        let x = activated * u;
        let groups = x.reshape(shape![16, 32, 16]);
        let maximum: Tile<f32, { [16, 32] }> = reduce_max(absf(groups), 2i32);
        let scale = min_tile(
            maximum * (input_global / 6.0f32).broadcast(shape![16, 32]),
            448.0f32.broadcast(shape![16, 32]),
        );
        let encoded: Tile<f8e4m3fn, { [16, 32] }> = convert_tile(scale);
        let rounded: Tile<f32, { [16, 32] }> = convert_tile(encoded);
        let zero = eq_tile(rounded, 0.0f32.broadcast(shape![16, 32]));
        let denominator = select(zero, 1.0f32.broadcast(shape![16, 32]), rounded);
        let normalized = groups
            * (input_global.broadcast(shape![16, 32]) / denominator)
                .reshape(shape![16, 32, 1])
                .broadcast(shape![16, 32, 16]);
        let clipped = min_tile(
            max_tile(normalized, (-6.0f32).broadcast(shape![16, 32, 16])),
            6.0f32.broadcast(shape![16, 32, 16]),
        );
        let values: Tile<f4e2m1fn, { [16, 512] }> = convert_tile(clipped.reshape(shape![16, 512]));
        out.store(values.pack(shape![16, 256]));
        scales.store(encoded);
    }

    #[cutile::entry()]
    fn packed<const K: i32, const M: i32, const N: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        input_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let xs = input_scales.partition(shape![M, 16]);
        let wp = weight.partition(shape![N, 128]);
        let sp = scales.partition(shape![N, 16]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for k in 0i32..((K + 255) / 256) {
            let x = xp.load([pid.0, k]).unpack(shape![M, 256]);
            let w = wp.load([pid.1, k]).unpack(shape![N, 256]).transpose();
            acc = mmaf_scaled(
                x,
                w,
                acc,
                xs.load([pid.0, k]),
                sp.load([pid.1, k]).transpose(),
            );
        }
        out.store(acc * alpha.broadcast(shape![M, N]));
    }

    /// Prefill main loop with compiler-scheduled operand and scale prefetching.
    /// Keep the native 256-element K step and block-16 scale arithmetic unchanged.
    /// Output geometry is chosen from shape and driver-reported parallelism at capture time.
    #[cutile::entry()]
    fn packed_prefill<const K: i32, const M: i32, const N: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        input_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let xs = input_scales.partition(shape![M, 16]);
        let wp = weight.partition(shape![N, 128]);
        let sp = scales.partition(shape![N, 16]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for k in 0i32..((K + 255) / 256) {
            let x = xp.load_pipelined::<8>([pid.0, k]).unpack(shape![M, 256]);
            let w = wp
                .load_pipelined::<8>([pid.1, k])
                .unpack(shape![N, 256])
                .transpose();
            acc = mmaf_scaled(
                x,
                w,
                acc,
                xs.load_pipelined::<8>([pid.0, k]),
                sp.load_pipelined::<8>([pid.1, k]).transpose(),
            );
        }
        out.store(acc * alpha.broadcast(shape![M, N]));
    }

    /// K-split twin of `packed` for bandwidth-bound narrow GEMMs: CTA `pid.2` accumulates its
    /// contiguous share of the 256-element K steps into partial `pid.2`, and `sum_splits` adds the
    /// partials in split order before applying `alpha`. `S` must divide the K step count.
    #[cutile::entry()]
    fn packed_split<
        const K: i32,
        const M: i32,
        const N: i32,
        const S: i32,
        const R: i32,
        const C: i32,
    >(
        partials: &mut Tensor<f32, { [S, R, C] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        input_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let xs = input_scales.partition(shape![M, 16]);
        let wp = weight.partition(shape![N, 128]);
        let sp = scales.partition(shape![N, 16]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for step in 0i32..((K + 255) / 256 / S) {
            let k = pid.2 * ((K + 255) / 256 / S) + step;
            let x = xp.load_pipelined::<8>([pid.0, k]).unpack(shape![M, 256]);
            let w = wp
                .load_pipelined::<8>([pid.1, k])
                .unpack(shape![N, 256])
                .transpose();
            acc = mmaf_scaled(
                x,
                w,
                acc,
                xs.load_pipelined::<8>([pid.0, k]),
                sp.load_pipelined::<8>([pid.1, k]).transpose(),
            );
        }
        partials
            .partition_mut(shape![1, M, N])
            .store(acc.reshape(shape![1, M, N]), [pid.2, pid.0, pid.1]);
    }

    /// Ordered sum of `packed_split` partials, scaled once like the unsplit epilogue.
    #[cutile::entry()]
    fn sum_splits<const M: i32, const N: i32, const S: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
        partials: &Tensor<f32, { [-1, -1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let pp = partials.partition(shape![1, M, N]);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for split in 0i32..S {
            let part: Tile<f32, { [M, N] }> = pp.load([split, pid.0, pid.1]).reshape(shape![M, N]);
            acc = acc + part;
        }
        out.store(acc * alpha.broadcast(shape![M, N]));
    }

    /// Gate and up projections of a gated MLP in one prefill pass. Each input block feeds both
    /// weights, and `silu(gate) * up` is quantized straight into the down projection's codes, so
    /// neither F32 projection output is written. The epilogue repeats `swiglu_quantize` on the
    /// same F32 values the separate projections store, keeping every code and scale unchanged.
    /// `HALF` and `GROUPS` are `N / 2` packed bytes and `N / 16` scales per tile row.
    #[cutile::entry()]
    fn packed_swiglu<
        const K: i32,
        const M: i32,
        const N: i32,
        const HALF: i32,
        const GROUPS: i32,
    >(
        codes: &mut Tensor<f4e2m1fnx2, { [M, HALF] }>,
        code_scales: &mut Tensor<f8e4m3fn, { [M, GROUPS] }>,
        input: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        input_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        gate: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        gate_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        up: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        up_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        gate_alpha: f32,
        up_alpha: f32,
        down_global: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![M, 128]);
        let xs = input_scales.partition(shape![M, 16]);
        let gp = gate.partition(shape![N, 128]);
        let gs = gate_scales.partition(shape![N, 16]);
        let upp = up.partition(shape![N, 128]);
        let us = up_scales.partition(shape![N, 16]);
        let mut gate_acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        let mut up_acc: Tile<f32, { [M, N] }> = constant(0.0f32, shape![M, N]);
        for k in 0i32..((K + 255) / 256) {
            let x = xp.load_pipelined::<8>([pid.0, k]).unpack(shape![M, 256]);
            let x_scales = xs.load_pipelined::<8>([pid.0, k]);
            let g = gp
                .load_pipelined::<8>([pid.1, k])
                .unpack(shape![N, 256])
                .transpose();
            gate_acc = mmaf_scaled(
                x,
                g,
                gate_acc,
                x_scales,
                gs.load_pipelined::<8>([pid.1, k]).transpose(),
            );
            let u = upp
                .load_pipelined::<8>([pid.1, k])
                .unpack(shape![N, 256])
                .transpose();
            up_acc = mmaf_scaled(
                x,
                u,
                up_acc,
                x_scales,
                us.load_pipelined::<8>([pid.1, k]).transpose(),
            );
        }
        let g: Tile<f32, { [M, N] }> = gate_acc * gate_alpha.broadcast(shape![M, N]);
        let u: Tile<f32, { [M, N] }> = up_acc * up_alpha.broadcast(shape![M, N]);
        let negative: Tile<f32, { [M, N] }> = 0.0f32.broadcast(shape![M, N]) - g;
        let one: Tile<f32, { [M, N] }> = 1.0f32.broadcast(shape![M, N]);
        let activated: Tile<f32, { [M, N] }> = g / (one + exp(negative));
        let x: Tile<f32, { [M, N] }> = activated * u;
        let groups: Tile<f32, { [M, GROUPS, 16] }> = x.reshape(shape![M, GROUPS, 16]);
        let reduced: Tile<f32, { [M, GROUPS] }> = reduce_max(absf(groups), 2i32);
        let maximum: Tile<f32, { [M, GROUPS] }> = reduced.reshape(shape![M, GROUPS]);
        let factor: Tile<f32, { [M, GROUPS] }> =
            (down_global / 6.0f32).broadcast(shape![M, GROUPS]);
        let limit: Tile<f32, { [M, GROUPS] }> = 448.0f32.broadcast(shape![M, GROUPS]);
        let product: Tile<f32, { [M, GROUPS] }> = maximum * factor;
        let scale: Tile<f32, { [M, GROUPS] }> = min_tile(product, limit);
        let encoded: Tile<f8e4m3fn, { [M, GROUPS] }> = convert_tile(scale);
        let rounded: Tile<f32, { [M, GROUPS] }> = convert_tile(encoded);
        let zeros: Tile<f32, { [M, GROUPS] }> = 0.0f32.broadcast(shape![M, GROUPS]);
        let ones: Tile<f32, { [M, GROUPS] }> = 1.0f32.broadcast(shape![M, GROUPS]);
        let zero = eq_tile(rounded, zeros);
        let denominator: Tile<f32, { [M, GROUPS] }> = select(zero, ones, rounded);
        let global: Tile<f32, { [M, GROUPS] }> = down_global.broadcast(shape![M, GROUPS]);
        let multiplier: Tile<f32, { [M, GROUPS] }> = global / denominator;
        let spread: Tile<f32, { [M, GROUPS, 16] }> = multiplier
            .reshape(shape![M, GROUPS, 1])
            .broadcast(shape![M, GROUPS, 16]);
        let normalized: Tile<f32, { [M, GROUPS, 16] }> = groups * spread;
        let low: Tile<f32, { [M, GROUPS, 16] }> = (-6.0f32).broadcast(shape![M, GROUPS, 16]);
        let high: Tile<f32, { [M, GROUPS, 16] }> = 6.0f32.broadcast(shape![M, GROUPS, 16]);
        let clipped: Tile<f32, { [M, GROUPS, 16] }> = min_tile(max_tile(normalized, low), high);
        let flat: Tile<f32, { [M, N] }> = clipped.reshape(shape![M, N]);
        let values: Tile<f4e2m1fn, { [M, N] }> = convert_tile(flat);
        codes.store(values.pack(shape![M, HALF]));
        code_scales.store(encoded);
    }
}

/// Record only when the checkpoint declares an activation scale for packed FP4 weights.
pub(super) fn record(
    scope: &Scope,
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &TensorView<'_, f32>,
    output: &mut PreludeTensor<f32>,
    columns: usize,
) -> Result<bool, DeviceError> {
    workspace.clear_quant_reuse();
    record_input(
        scope,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        None,
    )
}

/// The caller supplies the pointer of the tensor from which this exact view was made.
#[expect(
    clippy::too_many_arguments,
    reason = "The certified view adds its actual device pointer to the unchanged projection contract"
)]
pub(super) fn record_with_input_pointer(
    scope: &Scope,
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &TensorView<'_, f32>,
    output: &mut PreludeTensor<f32>,
    columns: usize,
    input_pointer: u64,
) -> Result<bool, DeviceError> {
    record_input(
        scope,
        workspace,
        weight,
        input_scale,
        input,
        output,
        columns,
        Some(input_pointer),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "One dispatcher preserves both certified and legacy projection contracts"
)]
fn record_input(
    scope: &Scope,
    workspace: &mut Workspace,
    weight: &MlpProjectionWeight,
    input_scale: Option<super::program::ActivationQuantization>,
    input: &TensorView<'_, f32>,
    output: &mut PreludeTensor<f32>,
    columns: usize,
    input_pointer: Option<u64>,
) -> Result<bool, DeviceError> {
    use super::program::ActivationQuantization;
    use crate::mlp::ProjectionWeight;
    match (weight, input_scale) {
        (ProjectionWeight::Fp4(..), Some(ActivationQuantization::Fp4(scale))) => {
            workspace.record_input(scope, weight, scale, input, output, columns, input_pointer)?;
        }
        (
            ProjectionWeight::Fp8(..) | ProjectionWeight::Fp8Block(..),
            Some(ActivationQuantization::Fp8Token),
        ) => {
            workspace.record_fp8_input(scope, weight, input, output, columns, input_pointer)?;
        }
        _ => {
            workspace.clear_quant_reuse();
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "../../tests/unit/resident_nvfp4_gemm.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_nvfp4_pipeline.rs"]
mod pipeline_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_vendor_fp8.rs"]
mod vendor_tests;

#[cfg(test)]
#[path = "../../tests/unit/resident_quant_reuse.rs"]
mod quant_reuse_tests;
