//! AOT FA4 prefill: GPU page gathering, architecture-specific MMA, then F32 readout.
//! The BF16 and SM120-FP16 profiles are distinct from the canonical F32 exact attention profile.
pub mod catalog;
pub mod f16;
mod host;
mod recorded;
mod workspace;
use super::attention_oxide::{Geometry, owners};
use crate::constants::KV_BLOCK_TOKENS;
use crate::device::{CudaDevice, device_error};
use catalog::{DIM, HEADS, KV_HEADS, Variant};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{DecodeBuffers, KvBuffers, Owner, Recording};
use infer_cuda_kernels::runtime::Function;
use infer_cuda_kernels::runtime::{Argument, KernelLaunch};
use std::collections::BTreeMap;
use std::sync::Arc;
pub use workspace::{Workspace, eligible};
#[cfg(test)]
pub use workspace::{eligible_with_role, node_eligible};

const PTX: &str = include_str!("../../../kernels/fa4-attention/bridge.ptx");
const F32_PTX: &str = include_str!("../../../kernels/fa4-attention-f32/bridge.ptx");
const THREADS: u32 = 256;
const RESIDENT_BLOCKS_PER_SM: u32 = 8;
/// Version of the F32 readout's host launch contract: the bridge parameters, their order, the
/// gather and readout grids. A measured choice stays valid across host refactors, so bump this
/// only when a launch of the same artifacts would run differently.
const F32_HOST_CONTRACT: &str = "fa4-attention-f32-launch-v1";

/// Arithmetic of one AOT attention support instance. The draft program stays on BF16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arithmetic {
    Bf16,
    F16,
}
pub struct Support {
    arithmetic: Arithmetic,
    prepare: Arc<Function>,
    prepare_f32: Option<Arc<Function>>,
    finish: Arc<Function>,
    apis: BTreeMap<Variant, Arc<host::Api>>,
    f16_api: Option<Arc<host::Api>>,
    scratch: workspace::Scratch,
    stream: usize,
    grid: u32,
    capacity: usize,
    lanes: usize,
}
fn rejected() -> Error {
    super::capture::error("selected MMA cannot record these operands")
}

impl Support {
    pub fn load(
        device: &CudaDevice,
        capacity: usize,
        lanes: usize,
        arithmetic: Arithmetic,
    ) -> Result<Self> {
        if capacity == 0
            || !capacity.is_multiple_of(KV_BLOCK_TOKENS)
            || i32::try_from(capacity).is_err()
            || i32::try_from(lanes).is_err()
        {
            return Err(Error::invalid("MMA scratch dimensions"));
        }
        if arithmetic == Arithmetic::F16 && device.profile()?.architecture != f16::ARCHITECTURE {
            return Err(Error::invalid(
                "FP16 target attention artifact is only validated for SM120",
            ));
        }
        catalog::workspace_bytes(capacity, lanes)?;
        let (ptx, prepare_name, finish_name) = match arithmetic {
            Arithmetic::Bf16 => (PTX, "prepare_fa4", "finish_fa4"),
            Arithmetic::F16 => (f16::PTX, "prepare_fa4_f16", "finish_fa4_f16"),
        };
        let module = device.tile().load_ptx(ptx).map_err(device_error)?;
        let sms = device.profile()?.multiprocessors;
        Ok(Self {
            arithmetic,
            prepare: Arc::new(module.function(prepare_name).map_err(device_error)?),
            prepare_f32: None,
            finish: Arc::new(module.function(finish_name).map_err(device_error)?),
            apis: BTreeMap::new(),
            f16_api: None,
            scratch: workspace::Scratch::new(device, capacity, lanes, arithmetic)?,
            stream: device.tile().stream_id(),
            grid: sms.saturating_mul(RESIDENT_BLOCKS_PER_SM).max(1),
            capacity,
            lanes,
        })
    }
    pub fn prepare_float_cache(&mut self, device: &CudaDevice) -> Result<()> {
        if self.arithmetic != Arithmetic::Bf16 {
            return Err(Error::invalid(
                "FP16 target attention has no F32-cache bridge",
            ));
        }
        if self.prepare_f32.is_none() {
            let module = device.tile().load_ptx(F32_PTX).map_err(device_error)?;
            self.prepare_f32 = Some(Arc::new(
                module.function("prepare_fa4_f32").map_err(device_error)?,
            ));
        }
        Ok(())
    }
    pub fn prepare_variant(&mut self, device: &CudaDevice, variant: Variant) -> Result<()> {
        let ordinal = i32::try_from(device.ordinal()).map_err(device_error)?;
        if self.arithmetic == Arithmetic::F16 {
            if variant != Variant::Sm120 || device.profile()?.architecture != f16::ARCHITECTURE {
                return Err(Error::invalid(
                    "FP16 attention variant architecture mismatch",
                ));
            }
            self.f16_api = Some(host::Api::load_artifact(f16::VARIANT, f16::SO, ordinal)?);
            return Ok(());
        }
        if !Variant::candidates(&device.profile()?.architecture).contains(&variant) {
            return Err(Error::invalid("MMA variant architecture mismatch"));
        }
        self.apis
            .insert(variant, host::Api::load(variant, ordinal)?);
        Ok(())
    }
    fn operands_valid(
        &self,
        b: &DecodeBuffers<'_>,
        (heads, kv_heads, dim, lanes, capacity): Geometry,
        window: i32,
    ) -> bool {
        catalog::geometry(
            heads,
            kv_heads,
            dim,
            lanes,
            usize::try_from(window).unwrap_or(usize::MAX),
        ) && lanes <= self.lanes
            && capacity == self.capacity
            && b.query.len() == lanes * HEADS * DIM
            && (b.cache.groups, b.cache.dim) == (KV_HEADS, DIM)
            // A paged pool's cache holds more physical rows than one sequence's capacity; the
            // table maps the sequence's positions into them.
            && b.cache.rows().is_some_and(|rows| rows >= capacity)
            && b.output.len() == lanes * HEADS * DIM
            && b.metadata.len() >= catalog::METADATA_FIELDS
            && b.table.len() >= capacity / KV_BLOCK_TOKENS
    }
    #[must_use]
    pub fn accepts(&self, b: &DecodeBuffers<'_>, shape: Geometry, window: i32) -> bool {
        let scales = b.cache.scales;
        let bridge = match (self.arithmetic, b.cache.buffers) {
            (_, KvBuffers::Fp8 { .. }) => true,
            (Arithmetic::Bf16, KvBuffers::F32 { .. }) => {
                self.prepare_f32.is_some() && scales.iter().all(|v| v.to_bits() == 1.0f32.to_bits())
            }
            (Arithmetic::F16, KvBuffers::F32 { .. }) => false,
        };
        bridge
            && scales.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.operands_valid(b, shape, window)
    }
    /// Record conversion, the variant's AOT attention and readout over `b`; the graph keeps
    /// every owned buffer, the scratch, both modules and the AOT library alive.
    /// # Errors
    /// Rejects operands [`Self::accepts`] refuses or an unprepared variant, or returns a capture
    /// failure.
    #[expect(
        unsafe_code,
        reason = "checked operand addresses and the program's scratch are marshalled into the embedded bridges; the graph keeps every owner"
    )]
    pub fn record(
        &self,
        recording: &Recording<'_>,
        b: DecodeBuffers<'_>,
        shape: Geometry,
        window: i32,
        variant: Variant,
    ) -> Result<()> {
        let (_, _, _, lanes, capacity) = shape;
        if !self.accepts(&b, shape, window) {
            return Err(rejected());
        }
        let rows = b.cache.rows().ok_or_else(rejected)?;
        let api = match self.arithmetic {
            Arithmetic::Bf16 => self
                .apis
                .get(&variant)
                .ok_or_else(|| super::capture::error("MMA variant not prepared before capture"))?,
            Arithmetic::F16 => self.f16_api.as_ref().ok_or_else(|| {
                super::capture::error("FP16 MMA artifact not prepared before capture")
            })?,
        };
        let pointers = self.scratch.pointers();
        let [
            q_pointer,
            k_pointer,
            v_pointer,
            o_pointer,
            used_q_pointer,
            _,
        ] = pointers;
        let prepare_function = match (self.arithmetic, b.cache.buffers) {
            (Arithmetic::Bf16, KvBuffers::F32 { .. }) => self
                .prepare_f32
                .as_ref()
                .ok_or_else(|| super::capture::error("F32 cache preparation missing"))?,
            _ => &self.prepare,
        };
        let [keys, values] = b.cache.buffers.addresses();
        let [key_scale, value_scale] = b.cache.scales;
        let u32_of = |value: usize| u32::try_from(value).map_err(super::capture::error);
        // SAFETY: the bridge's parameters in order. `accepts` checked every operand extent and the
        // scratch was sized for this capacity and lane count; the graph keeps both alive. Physical
        // rows, not logical capacity, are the bridge's head stride; lengths bound it.
        let prepare = unsafe {
            KernelLaunch::new(
                Arc::clone(prepare_function),
                self.stream,
                self.grid,
                THREADS,
                vec![
                    Argument::buffer(b.query),
                    Argument::Address(keys),
                    Argument::Address(values),
                    Argument::buffer(b.metadata),
                    Argument::buffer(b.table),
                    Argument::Address(q_pointer),
                    Argument::Address(k_pointer),
                    Argument::Address(v_pointer),
                    Argument::Address(used_q_pointer),
                    Argument::U32(u32_of(rows)?),
                    Argument::U32(u32_of(lanes)?),
                    Argument::F32(key_scale),
                    Argument::F32(value_scale),
                ],
            )
        };
        // SAFETY: the readout's parameters in order, over the same checked output and metadata.
        let finish = unsafe {
            KernelLaunch::new(
                Arc::clone(&self.finish),
                self.stream,
                u32_of((lanes * HEADS * DIM).div_ceil(THREADS as usize))?,
                THREADS,
                vec![
                    Argument::Address(o_pointer),
                    Argument::buffer(b.output),
                    Argument::buffer(b.metadata),
                    Argument::U32(u32_of(lanes)?),
                ],
            )
        };
        if self.arithmetic == Arithmetic::F16 {
            tracing::info!(
                target: "infer::attention",
                attention_profile = "fp16",
                profile = f16::PROFILE,
                variant = f16::VARIANT,
                lanes,
                capacity,
                "MMA F16 attention recorded"
            );
        }
        let mut keep = owners(&b);
        keep.extend(self.scratch.owners());
        keep.extend([prepare.owner(), finish.owner(), Arc::clone(api) as Owner]);
        recording
            .foreign(
                recorded::Recorded {
                    prepare,
                    finish,
                    api: Arc::clone(api),
                    pointers,
                    lanes,
                    capacity,
                },
                keep,
            )
            .map_err(super::capture::error)
    }
}
pub fn artifact_identity() -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(PTX);
    hash.update(include_bytes!("../../../kernels/fa4-attention/bridge.cu"));
    hash.update(include_bytes!(
        "../../../kernels/fa4-attention/manifest.json"
    ));
    hash.update(catalog::PROFILE);
    for arch in ["sm_90", "sm_100", "sm_103", "sm_120"] {
        for variant in Variant::candidates(arch) {
            hash.update(variant.bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

pub fn f32_artifact_identity() -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(artifact_identity());
    hash.update(F32_PTX);
    hash.update(F32_HOST_CONTRACT);
    hash.update(format!("{THREADS}:{RESIDENT_BLOCKS_PER_SM}"));
    hash.update(include_bytes!(
        "../../../kernels/fa4-attention-f32/bridge.cu"
    ));
    format!("{:x}", hash.finalize())
}
