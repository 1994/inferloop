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
use super::super::buffers::buffer;
use super::super::capture::error;
use crate::device::CudaDevice;
use infer_core::Result;
use infer_cuda_kernels::contract::{Buffer, ChunkBuffers, Recording, Recurrent, RecurrentShape};
use infer_cuda_kernels::runtime::Allocation;
use infer_cuda_kernels::runtime::bf16;

/// Tokens per chunk.
pub(super) const CHUNK: usize = 64;
#[cfg(test)]
#[cfg(feature = "cuda")]
use infer_cuda_kernels::tile::recurrent::recurrent_chunk as kernels;

/// Scratch for one prompt width.
pub(super) struct Scratch {
    u: Allocation<bf16>,
    queries: Allocation<bf16>,
    keys: Allocation<bf16>,
    w: Allocation<bf16>,
    p: Allocation<bf16>,
    cumulative: Allocation<f32>,
}

impl Scratch {
    pub(super) fn new(
        device: &CudaDevice,
        (kh, vh, dim): (usize, usize, usize),
        lanes: usize,
    ) -> Result<Self> {
        let runtime = device.runtime();
        let halves = |shape: &[usize]| runtime.zeros::<bf16>(shape.iter().product());
        Ok(Self {
            u: halves(&[vh, lanes, dim])?,
            queries: halves(&[kh, lanes, dim])?,
            keys: halves(&[kh, lanes, dim])?,
            w: halves(&[vh, lanes, dim])?,
            p: halves(&[vh, lanes, CHUNK])?,
            cumulative: runtime.zeros::<f32>(vh * lanes)?,
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

/// F32 operands of [`record`].
pub(super) const CHUNK_OPERANDS: usize = 5;

/// Record the chunked recurrence; `betas` and `decays` come from the shared gate kernel.
pub(super) fn record(
    scope: &Recording<'_>,
    recurrent: &dyn Recurrent,
    scratch: &Scratch,
    shape: RecurrentShape,
    [qkv, betas, decays, state, output]: [Buffer<'_, f32>; CHUNK_OPERANDS],
    metadata: Buffer<'_, i32>,
) -> Result<()> {
    recurrent
        .chunk(
            scope,
            shape,
            CHUNK,
            ChunkBuffers {
                queries: buffer(&scratch.queries)?,
                keys: buffer(&scratch.keys)?,
                u: buffer(&scratch.u)?,
                w: buffer(&scratch.w)?,
                p: buffer(&scratch.p)?,
                cumulative: buffer(&scratch.cumulative)?,
                qkv,
                betas,
                decays,
                state: state.into(),
                output,
                metadata,
            },
        )
        .map_err(error)
}

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../../tests/unit/resident_recurrent_scan_tiling.rs"]
mod scan_tiling_tests;
