//! Exact F32 prefill attention compiled from the pinned Rust SIMT source.
//! The generated PTX is embedded so host builds keep their stable toolchain.
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::{CudaDevice, device_error};
use infer_core::Result;
use infer_cuda_kernels::contract::{DecodeBuffers, KvBuffers, Owner, Recording};
use infer_cuda_kernels::runtime::Function;
use infer_cuda_kernels::runtime::{Argument, KernelLaunch};
use infer_ir::{DataflowGraph, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::sync::Arc;
use std::sync::OnceLock;
mod shared;
mod vector;

const PTX: &str = include_str!("../../kernels/oxide-attention/attention-sm120.ptx");
const HEADS: usize = 24;
const KV_HEADS: usize = 4;
const DIM: usize = 256;
const THREADS: u32 = 256;
const SUPPORTED_LANES: [usize; 5] = [64, 256, 512, 1024, 2048];
const REQUIRED_METADATA_FIELDS: usize = 3;

pub struct Support {
    function: Arc<Function>,
    stream: usize,
    threads: u32,
    rows_per_cta: usize,
    lanes: &'static [usize],
}

/// Geometry of one prefill attention launch: query heads, KV heads, head dimension, lanes and
/// the sequence capacity the table covers.
pub type Geometry = (usize, usize, usize, usize, usize);

/// What a graph replaying an attention launch over `buffers` must keep alive.
pub(super) fn owners(buffers: &DecodeBuffers<'_>) -> Vec<Owner> {
    let [keys, values] = buffers.cache.buffers.owners();
    [
        buffers.output.owner(),
        buffers.query.owner(),
        keys,
        values,
        buffers.metadata.owner(),
        buffers.table.owner(),
    ]
    .into_iter()
    .flatten()
    .cloned()
    .collect()
}

impl Support {
    pub(super) fn eligible(device: &CudaDevice, graph: &DataflowGraph) -> bool {
        #[cfg(test)]
        if DISABLED.get() {
            return false;
        }
        device.target().has_simt_artifacts()
            && graph.nodes.iter().any(|node| {
                matches!(
                    node.op,
                    TensorOp::Attention {
                        query_heads: HEADS,
                        kv_heads: KV_HEADS,
                        head_dim: DIM,
                        ..
                    }
                )
            })
    }

    pub fn load(device: &CudaDevice) -> Result<Option<Self>> {
        if !device.target().has_simt_artifacts() {
            return Ok(None);
        }
        let module = device.tile().load_ptx(PTX).map_err(device_error)?;
        Ok(Some(Self {
            function: Arc::new(module.function("attention_fp8_256").map_err(device_error)?),
            stream: device.tile().stream_id(),
            threads: THREADS,
            rows_per_cta: 1,
            lanes: &SUPPORTED_LANES,
        }))
    }

    pub fn load_vector(device: &CudaDevice) -> Result<Option<Self>> {
        vector::load(device)
    }

    pub fn load_shared(device: &CudaDevice) -> Result<Option<Self>> {
        shared::load(device)
    }

    #[cfg(test)]
    #[cfg(feature = "cuda")]
    pub(crate) fn load_test_ptx(device: &CudaDevice, ptx: &str) -> Result<Self> {
        let module = device.tile().load_ptx(ptx).map_err(device_error)?;
        Ok(Self {
            function: Arc::new(module.function("attention_fp8_256").map_err(device_error)?),
            stream: device.tile().stream_id(),
            threads: THREADS,
            rows_per_cta: 1,
            lanes: &SUPPORTED_LANES,
        })
    }
    /// Whether this kernel's fixed layout fits these operands. A pool whose shared pages outnumber
    /// a slot's capacity does not, so the caller takes its next path.
    #[must_use]
    pub fn accepts(
        &self,
        buffers: &DecodeBuffers<'_>,
        (heads, kv_heads, dim, lanes, capacity): Geometry,
    ) -> bool {
        let cache = &buffers.cache;
        (heads, kv_heads, dim) == (HEADS, KV_HEADS, DIM)
            && self.lanes.contains(&lanes)
            && matches!(cache.buffers, KvBuffers::Fp8 { .. })
            && (cache.groups, cache.dim) == (KV_HEADS, DIM)
            && capacity != 0
            && capacity.is_multiple_of(KV_BLOCK_TOKENS)
            && cache.scales.iter().all(|x| x.is_finite() && *x > 0.0)
            && cache.rows() == Some(capacity)
            && buffers.query.len() == lanes * HEADS * DIM
            && buffers.output.len() == lanes * HEADS * DIM
            && buffers.metadata.len() >= REQUIRED_METADATA_FIELDS
            && !buffers.table.is_empty()
    }

    /// Record the kernel over `buffers`; the graph keeps every owned buffer and the module alive.
    /// # Errors
    /// Rejects operands [`Self::accepts`] refuses, or returns a capture failure.
    #[expect(
        unsafe_code,
        reason = "checked operand addresses are marshalled into the pinned Rust kernel; the graph keeps every owner"
    )]
    pub fn record(
        &self,
        recording: &Recording<'_>,
        buffers: DecodeBuffers<'_>,
        geometry: Geometry,
        window: i32,
    ) -> Result<()> {
        if !self.accepts(&buffers, geometry) {
            return Err(super::capture::error(
                "selected SIMT attention cannot record these operands",
            ));
        }
        let (_, _, _, lanes, capacity) = geometry;
        let [keys, values] = buffers.cache.buffers.addresses();
        let [key_scale, value_scale] = buffers.cache.scales;
        let grid = u32::try_from((lanes * HEADS).div_ceil(self.rows_per_cta))
            .map_err(super::capture::error)?;
        // SAFETY: these are `attention_fp8_256`'s parameters in order. `accepts` checked every
        // extent the kernel reads; the resident allocator validates logical positions and every
        // physical table entry, and the graph keeps the owned buffers alive.
        let launch = unsafe {
            KernelLaunch::new(
                Arc::clone(&self.function),
                self.stream,
                grid,
                self.threads,
                vec![
                    Argument::buffer(buffers.query),
                    Argument::Address(keys),
                    Argument::Address(values),
                    Argument::buffer(buffers.metadata),
                    Argument::buffer(buffers.table),
                    Argument::buffer(buffers.output),
                    Argument::U32(u32::try_from(capacity).map_err(super::capture::error)?),
                    Argument::I32(window),
                    Argument::F32(key_scale),
                    Argument::F32(value_scale),
                ],
            )
        };
        let mut keep = owners(&buffers);
        keep.push(launch.owner());
        recording
            .foreign(launch, keep)
            .map_err(super::capture::error)?;
        #[cfg(test)]
        CAPTURES.set(CAPTURES.get() + 1);
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub fn disabled() -> bool {
    DISABLED.get()
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}
#[cfg(test)]
pub fn without_oxide<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let guard = Reset(DISABLED.replace(true));
    let result = f();
    drop(guard);
    result
}

pub fn artifact_identity() -> &'static str {
    use sha2::{Digest, Sha256};
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let source = include_str!("../../kernels/oxide-attention/src/main.rs");
        let provenance = include_str!("../../kernels/oxide-attention/provenance.json");
        format!(
            "{:x}:{:x}:{:x}",
            Sha256::digest(PTX),
            Sha256::digest(source),
            Sha256::digest(provenance)
        )
    })
}

#[cfg(test)]
#[cfg(feature = "cuda")]
#[path = "../../tests/unit/resident_attention_oxide_resources.rs"]
mod resources_tests;

pub fn shared_identity() -> &'static str {
    shared::artifact_identity()
}

pub fn vector_identity() -> &'static str {
    vector::artifact_identity()
}
