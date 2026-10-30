//! Chunked gated delta rule on tensor cores: the FLA/FlashInfer prompt formulation.
//!
//! Within a 64-token chunk the recurrence `S_t = d_t S_{t-1} + k_t (b_t (v_t - d_t S_{t-1}^T k_t))^T`
//! has the closed form `U = T (b V) - T (b g K) S_0`, `O = (g Q) S_0 + mask(Q K^T G) U` and
//! `S_64 = g_64 S_0 + (g_64 / g) K^T U`, where `g` is the cumulative decay, `G_ij = g_i / g_j` and
//! `T = (I + strict_lower(b_i k_i k_j G_ij))^-1`.
//!
//! [`kernels::chunk_normalize`] writes L2-normalized BF16 queries and keys per key head;
//! [`kernels::chunk_prepare`] evaluates every chunk independently into BF16 factors (BF16
//! products, TF32 for the triangular inverse);
//! [`kernels::chunk_scan`] then carries the F32 state across chunks with four BF16 products per
//! chunk, as the FLA and `FlashInfer` kernels do.
use crate::device::{CudaDevice, device_error};
use cutile::{half::bf16, prelude::*};
use infer_core::Result;
use std::result::Result as StdResult;

/// Tokens per chunk.
pub(super) const CHUNK: usize = 64;
/// Value columns in the general scan tile.
const SCAN_COLUMNS: usize = 32;
/// Wider tiles amortize factor loads for this measured value-head geometry.
const WIDE_SCAN_COLUMNS: usize = 64;
/// Value-head count whose prompt widths benefit from the wider tile.
const WIDE_SCAN_VALUE_HEADS: usize = 48;

pub use infer_cuda_kernels::tile::recurrent_chunk as kernels;

/// Scratch for one prompt width.
pub(super) struct Scratch {
    u: Tensor<bf16>,
    queries: Tensor<bf16>,
    keys: Tensor<bf16>,
    w: Tensor<bf16>,
    p: Tensor<bf16>,
    cumulative: Tensor<f32>,
}

impl Scratch {
    pub(super) fn new(
        device: &CudaDevice,
        (kh, vh, dim): (usize, usize, usize),
        lanes: usize,
    ) -> Result<Self> {
        let halves = |shape: &[usize]| {
            api::zeros::<bf16>(shape)
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        Ok(Self {
            u: halves(&[vh, lanes, dim])?,
            queries: halves(&[kh, lanes, dim])?,
            keys: halves(&[kh, lanes, dim])?,
            w: halves(&[vh, lanes, dim])?,
            p: halves(&[vh, lanes, CHUNK])?,
            cumulative: api::zeros::<f32>(&[vh, lanes])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }

    /// Device bytes of [`Self::new`].
    pub(super) const fn bytes((kh, vh, dim): (usize, usize, usize), lanes: usize) -> usize {
        (2 * kh * lanes * dim + 2 * vh * lanes * dim + vh * lanes * CHUNK) * size_of::<bf16>()
            + vh * lanes * size_of::<f32>()
    }
}

/// Whether the chunked rule covers this geometry.
pub(super) const fn enabled(dim: usize, lanes: usize) -> bool {
    dim == super::ORDERED_DIM && lanes >= CHUNK && lanes.is_multiple_of(CHUNK)
}

/// Record the chunked recurrence; `betas` and `decays` come from the shared gate kernel.
pub(super) fn record(
    scope: &Scope,
    scratch: &mut Scratch,
    qkv: &TensorView<'_, f32>,
    (betas, decays): (&Tensor<f32>, &Tensor<f32>),
    [state, output]: [&mut Tensor<f32>; 2],
    metadata: &Tensor<i32>,
    (kh, vh, dim, lanes): (usize, usize, usize, usize),
) -> StdResult<(), DeviceError> {
    scope.record(
        kernels::chunk_normalize(
            (&mut scratch.queries).partition([1, CHUNK, dim]),
            (&mut scratch.keys).partition([1, CHUNK, dim]),
            qkv,
            metadata,
        )
        .generics(vec![kh.to_string(), dim.to_string()]),
    )?;
    scope.record(
        kernels::chunk_prepare(
            (&mut scratch.u).partition([1, CHUNK, dim]),
            (&mut scratch.w).partition([1, CHUNK, dim]),
            (&mut scratch.p).partition([1, CHUNK, CHUNK]),
            (&mut scratch.cumulative).partition([1, CHUNK]),
            qkv,
            &scratch.queries,
            &scratch.keys,
            betas,
            decays,
            metadata,
        )
        .generics(vec![kh.to_string(), vh.to_string(), dim.to_string()]),
    )?;
    if vh == WIDE_SCAN_VALUE_HEADS {
        scope.record(
            kernels::chunk_scan_wide(
                state.partition([1, dim, WIDE_SCAN_COLUMNS]),
                output.partition([1, lanes, WIDE_SCAN_COLUMNS]),
                &scratch.u,
                &scratch.w,
                &scratch.queries,
                &scratch.keys,
                &scratch.p,
                &scratch.cumulative,
                metadata,
            )
            .generics(vec![kh.to_string(), vh.to_string(), lanes.to_string()]),
        )?;
    } else {
        scope.record(
            kernels::chunk_scan(
                state.partition([1, dim, SCAN_COLUMNS]),
                output.partition([1, lanes, SCAN_COLUMNS]),
                &scratch.u,
                &scratch.w,
                &scratch.queries,
                &scratch.keys,
                &scratch.p,
                &scratch.cumulative,
                metadata,
            )
            .generics(vec![kh.to_string(), vh.to_string(), lanes.to_string()]),
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_recurrent_scan_tiling.rs"]
mod scan_tiling_tests;
