//! Small batched projections using tensor cores with a two-term BF16 input.
//! Retaining the input residual avoids reducing F32 activations to one BF16 value.
pub use infer_cuda_kernels::tile::projection::small_gemm as kernels;
#[cfg(test)]
use std::cell::Cell;

// Unqualified arithmetic candidates exist only in the CUDA test executable.
#[cfg(test)]
#[cutile::module]
pub(crate) mod precise_kernels {
    use cutile::core::{
        ElementType, Shape_2, StoreTileAtCurrentBlock, Tensor_2, Tile_2, bf16, constant,
        convert_tile, get_tile_block_id, mmaf,
    };
    /// The third BF16 component reduces the residual input quantization error.
    /// This experiment is selected consistently by private and pooled capture.
    #[cutile::entry()]
    fn dense<E: ElementType, const K: i32>(
        out: &mut Tensor<f32, { [16, 32] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<E, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![32, 128]);
        let zero: Tile<f32, { [16, 32] }> = constant(0.0f32, shape![16, 32]);
        let mut acc: Tile<f32, { [16, 32] }> = zero;
        let mut correction: Tile<f32, { [16, 32] }> = zero;
        for k in 0i32..((K + 127) / 128) {
            let input = xp.load([pid.0, k]);
            let high: Tile<bf16, { [16, 128] }> = convert_tile(input);
            let rounded: Tile<f32, { [16, 128] }> = convert_tile(high);
            let residual = input - rounded;
            let low: Tile<bf16, { [16, 128] }> = convert_tile(residual);
            let rounded_low: Tile<f32, { [16, 128] }> = convert_tile(low);
            let tail: Tile<bf16, { [16, 128] }> = convert_tile(residual - rounded_low);
            let weight: Tile<bf16, { [32, 128] }> = convert_tile(wp.load([pid.1, k]));
            // Sum each component without injecting it into a large previous accumulator.
            // Keep block summation compensated so cancellation across K stays within the F64 gate.
            let high_product = mmaf(high, weight.transpose(), zero);
            let low_product = mmaf(low, weight.transpose(), zero);
            let tail_product = mmaf(tail, weight.transpose(), zero);
            let block = (high_product + low_product) + tail_product;
            let corrected = block - correction;
            let next = acc + corrected;
            correction = (next - acc) - corrected;
            acc = next;
        }
        out.store(acc);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_small_gemm.rs"]
mod tests;

/// Internal A/B selection; failed candidates cannot be enabled in a production build.
#[cfg(test)]
pub(super) fn precise_dense_enabled() -> bool {
    !pool_f32_enabled() && PRECISE_DENSE.with(Cell::get)
}

#[cfg(test)]
thread_local! {
    static PRECISE_DENSE: Cell<bool> = const { Cell::new(false) };
}
#[cfg(test)]
pub fn with_precise_dense<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRECISE_DENSE.with(|cell| cell.set(self.0));
        }
    }
    let restore = Restore(PRECISE_DENSE.with(|cell| cell.replace(enabled)));
    let result = run();
    drop(restore);
    result
}

/// Internal four-slot experiment; never dispatched by a production build.
#[cfg(test)]
pub(super) fn pool_f32_enabled() -> bool {
    POOL_F32.with(Cell::get)
}
#[cfg(test)]
thread_local! { static POOL_F32: Cell<bool> = const { Cell::new(false) }; }
#[cfg(test)]
pub fn with_pool_f32<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            POOL_F32.with(|cell| cell.set(self.0));
        }
    }
    let restore = Restore(POOL_F32.with(|cell| cell.replace(enabled)));
    let result = run();
    drop(restore);
    result
}

#[cfg(test)]
thread_local! { static POOL_RECORDS: Cell<usize> = const { Cell::new(0) }; }
#[cfg(test)]
pub(super) fn note_pool_record() {
    POOL_RECORDS.with(|count| count.set(count.get().saturating_add(1)));
}
#[cfg(test)]
pub fn pool_f32_capture_records() -> usize {
    POOL_RECORDS.with(Cell::get)
}
