//! AOT FA4 prefill: GPU page gathering, architecture-specific MMA, then F32 readout.
//! The BF16 and SM120-FP16 profiles are distinct from the canonical F32 exact attention profile.
pub mod catalog;
pub mod f16;
mod host;
mod recorded;
mod workspace;
pub use super::attention_oxide::Bindings;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::KV_CACHE_RANK;
use crate::device::{CudaDevice, device_error};
use catalog::{DIM, HEADS, KV_HEADS, Variant};
use cuda_core::{Function, LaunchConfig};
use cutile::{cuda_async::launch::AsyncKernelLaunch, prelude::*};
use infer_core::Result;
use std::collections::BTreeMap;
use std::result::Result as StdResult;
pub use workspace::{Workspace, eligible};
#[cfg(test)]
pub use workspace::{eligible_with_role, node_eligible};

const PTX: &str = include_str!("../../../kernels/fa4-attention/bridge.ptx");
const F32_PTX: &str = include_str!("../../../kernels/fa4-attention-f32/bridge.ptx");
const THREADS: u32 = 256;
const RESIDENT_BLOCKS_PER_SM: u32 = 8;

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
            return Err(infer_core::Error::invalid("MMA scratch dimensions"));
        }
        if arithmetic == Arithmetic::F16 && device.profile()?.architecture != f16::ARCHITECTURE {
            return Err(infer_core::Error::invalid(
                "FP16 target attention artifact is only validated for SM120",
            ));
        }
        catalog::workspace_bytes(capacity, lanes)?;
        let (ptx, prepare_name, finish_name) = match arithmetic {
            Arithmetic::Bf16 => (PTX, "prepare_fa4", "finish_fa4"),
            Arithmetic::F16 => (f16::PTX, "prepare_fa4_f16", "finish_fa4_f16"),
        };
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(ptx)
            .map_err(device_error)?;
        let sms = device.profile()?.multiprocessors;
        Ok(Self {
            arithmetic,
            prepare: Arc::new(module.load_function(prepare_name).map_err(device_error)?),
            prepare_f32: None,
            finish: Arc::new(module.load_function(finish_name).map_err(device_error)?),
            apis: BTreeMap::new(),
            f16_api: None,
            scratch: workspace::Scratch::new(device, capacity, lanes, arithmetic)?,
            stream: device.stream.cu_stream() as usize,
            grid: sms.saturating_mul(RESIDENT_BLOCKS_PER_SM).max(1),
            capacity,
            lanes,
        })
    }
    pub fn prepare_float_cache(&mut self, device: &CudaDevice) -> Result<()> {
        if self.arithmetic != Arithmetic::Bf16 {
            return Err(infer_core::Error::invalid(
                "FP16 target attention has no F32-cache bridge",
            ));
        }
        if self.prepare_f32.is_none() {
            let module = device
                .stream
                .device()
                .load_module_from_ptx_src(F32_PTX)
                .map_err(device_error)?;
            self.prepare_f32 = Some(Arc::new(
                module
                    .load_function("prepare_fa4_f32")
                    .map_err(device_error)?,
            ));
        }
        Ok(())
    }
    pub fn prepare_variant(&mut self, device: &CudaDevice, variant: Variant) -> Result<()> {
        let ordinal = i32::try_from(device.stream.device().ordinal()).map_err(device_error)?;
        if self.arithmetic == Arithmetic::F16 {
            if variant != Variant::Sm120 || device.profile()?.architecture != f16::ARCHITECTURE {
                return Err(infer_core::Error::invalid(
                    "FP16 attention variant architecture mismatch",
                ));
            }
            self.f16_api = Some(host::Api::load_artifact(f16::VARIANT, f16::SO, ordinal)?);
            return Ok(());
        }
        if !Variant::candidates(&device.profile()?.architecture).contains(&variant) {
            return Err(infer_core::Error::invalid(
                "MMA variant architecture mismatch",
            ));
        }
        self.apis
            .insert(variant, host::Api::load(variant, ordinal)?);
        Ok(())
    }
    fn operands_valid<E: DType>(
        &self,
        b: &Bindings<'_, E>,
        (heads, kv_heads, dim, lanes, capacity): (usize, usize, usize, usize, usize),
        window: i32,
    ) -> StdResult<bool, DeviceError> {
        Ok(catalog::geometry(
            heads,
            kv_heads,
            dim,
            lanes,
            usize::try_from(window).unwrap_or(usize::MAX),
        ) && lanes <= self.lanes
            && capacity == self.capacity
            && b.query.is_contiguous()
            && b.query.size() == lanes * HEADS * DIM
            && b.keys.is_contiguous()
            && b.values.is_contiguous()
            // A paged pool's cache holds more physical rows than one sequence's capacity; the
            // table maps the sequence's positions into them.
            && b.keys.shape().len() == KV_CACHE_RANK
            && b.keys.shape()[0] == i32::try_from(KV_HEADS).map_err(super::capture::error)?
            && b.keys.shape()[1] >= i32::try_from(capacity).map_err(super::capture::error)?
            && b.keys.shape()[2] == i32::try_from(DIM).map_err(super::capture::error)?
            && b.values.shape() == b.keys.shape()
            && b.output.is_contiguous()
            && b.output.size() == lanes * HEADS * DIM
            && b.metadata.is_contiguous()
            && b.metadata.size() >= catalog::METADATA_FIELDS
            && b.table.is_contiguous()
            && b.table.size() >= capacity / KV_BLOCK_TOKENS)
    }
    fn require_operands<E: DType>(
        &self,
        b: &Bindings<'_, E>,
        shape: (usize, usize, usize, usize, usize),
        window: i32,
        scales: [f32; 2],
    ) -> StdResult<(), DeviceError> {
        let bridge = match self.arithmetic {
            Arithmetic::Bf16 => {
                (E::DTYPE == cuda_core::f8e4m3fn::DTYPE && size_of::<E>() == 1)
                    || (E::DTYPE == f32::DTYPE
                        && size_of::<E>() == size_of::<f32>()
                        && self.prepare_f32.is_some()
                        && scales.iter().all(|v| v.to_bits() == 1.0f32.to_bits()))
            }
            Arithmetic::F16 => E::DTYPE == cuda_core::f8e4m3fn::DTYPE && size_of::<E>() == 1,
        };
        if bridge
            && scales.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.operands_valid(b, shape, window)?
        {
            return Ok(());
        }
        Err(super::capture::error(
            "selected MMA cannot record these operands",
        ))
    }
    #[expect(
        unsafe_code,
        reason = "checked contiguous graph-owned operands are marshalled into the embedded bridge; recorded execution retains all owners"
    )]
    pub fn record<E: DType>(
        &mut self,
        scope: &Scope,
        b: Bindings<'_, E>,
        shape: (usize, usize, usize, usize, usize),
        window: i32,
        scales: [f32; 2],
        variant: Variant,
    ) -> StdResult<(), DeviceError> {
        let (_, _, _, lanes, capacity) = shape;
        self.require_operands(&b, shape, window, scales)?;
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
        let prepare_function = if self.arithmetic == Arithmetic::F16 {
            &self.prepare
        } else if E::DTYPE == f32::DTYPE {
            self.prepare_f32
                .as_ref()
                .ok_or_else(|| super::capture::error("F32 cache preparation missing"))?
        } else {
            &self.prepare
        };
        let mut prepare = AsyncKernelLaunch::new(Arc::clone(prepare_function));
        for pointer in [
            b.query.device_pointer().cu_deviceptr(),
            b.keys.device_pointer().cu_deviceptr(),
            b.values.device_pointer().cu_deviceptr(),
            b.metadata.device_pointer().cu_deviceptr(),
            b.table.device_pointer().cu_deviceptr(),
            q_pointer,
            k_pointer,
            v_pointer,
            used_q_pointer,
        ] {
            // SAFETY: the checked operands and scratch are retained by Recorded.
            unsafe {
                prepare.push_device_ptr(pointer);
            }
        }
        // Physical rows, not logical capacity: the bridge's head stride; lengths bound it.
        prepare.push_arg(u32::try_from(b.keys.shape()[1]).map_err(super::capture::error)?);
        prepare.push_arg(u32::try_from(lanes).map_err(super::capture::error)?);
        prepare.push_arg(scales[0]);
        prepare.push_arg(scales[1]);
        prepare.set_launch_config(LaunchConfig {
            grid_dim: (self.grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        });
        let mut finish = AsyncKernelLaunch::new(Arc::clone(&self.finish));
        for pointer in [
            o_pointer,
            b.output.device_pointer().cu_deviceptr(),
            b.metadata.device_pointer().cu_deviceptr(),
        ] {
            // SAFETY: these are the same retained, checked outputs and metadata.
            unsafe {
                finish.push_device_ptr(pointer);
            }
        }
        finish.push_arg(u32::try_from(lanes).map_err(super::capture::error)?);
        finish.set_launch_config(LaunchConfig {
            grid_dim: (
                u32::try_from((lanes * HEADS * DIM).div_ceil(THREADS as usize))
                    .map_err(super::capture::error)?,
                1,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        });
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
        scope.record(recorded::Recorded {
            prepare,
            finish,
            api: Arc::clone(api),
            pointers,
            stream: self.stream,
            lanes,
            capacity,
            bindings: b,
            scratch: &mut self.scratch,
        })?;
        Ok(())
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
    hash.update(include_str!("mod.rs"));
    hash.update(include_str!("recorded.rs"));
    hash.update(include_str!("host.rs"));
    hash.update(include_bytes!(
        "../../../kernels/fa4-attention-f32/bridge.cu"
    ));
    format!("{:x}", hash.finalize())
}
