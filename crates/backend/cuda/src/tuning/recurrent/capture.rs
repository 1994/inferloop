//! Capture owns graph before its operand buffers; replay timing includes output transpose.
use super::{
    SHAPE_RANK,
    reference::{Fixture, maximum_error},
};
use infer_cuda_kernels::contract::{DeltaBuffers, DeltaKernel, RecurrentShape};
use infer_cuda_kernels::tile::identity;
use infer_cuda_kernels::tile::runtime::{Allocation, Graph};
use std::env::var_os;

const FINGERPRINT_SPLITS: [usize; 3] = [1, 2, 4];
use crate::device::{CudaDevice, device_error};
use infer_core::{Error, Result};

/// The graph retains every operand; only what a check reads or resets is named here.
pub(super) struct Captured {
    pub graph: Graph,
    state: Allocation<f32>,
    output: Allocation<f32>,
    initial: Allocation<f32>,
}
impl Captured {
    pub fn new(device: &CudaDevice, fixture: &Fixture, block: usize) -> Result<Self> {
        let [kh, vh, dim, lanes] = fixture.shape;
        let runtime = device.runtime();
        let qkv = runtime.upload(fixture.qkv.clone())?;
        let beta = runtime.upload(fixture.beta.clone())?;
        let alpha = runtime.upload(fixture.alpha.clone())?;
        let a_log = runtime.upload(fixture.a_log.clone())?;
        let bias = runtime.upload(fixture.bias.clone())?;
        let initial = runtime.upload(fixture.initial.clone())?;
        let metadata =
            runtime.upload(vec![0, i32::try_from(lanes).map_err(device_error)?, 0, 0])?;
        let state = runtime.zeros::<f32>(vh * dim * dim)?;
        let scratch = runtime.zeros::<f32>(vh * lanes * dim)?;
        let output = runtime.zeros::<f32>(lanes * vh * dim)?;
        let contract = device.kernels();
        let shape = RecurrentShape {
            key_heads: kh,
            value_heads: vh,
            dim,
            lanes,
        };
        let graph = runtime.capture(|recording| {
            contract.recurrent.delta(
                recording,
                DeltaKernel::Wy,
                shape,
                block,
                DeltaBuffers {
                    state: state.operand(),
                    output: scratch.operand(),
                    qkv: qkv.operand(),
                    beta: beta.operand(),
                    alpha: alpha.operand(),
                    decay: a_log.operand(),
                    bias: bias.operand(),
                    metadata: metadata.operand(),
                },
            )?;
            contract
                .recurrent
                .transpose(recording, shape, output.operand(), scratch.operand())
        })?;
        Ok(Self {
            graph,
            state,
            output,
            initial,
        })
    }
    pub fn verify(&self, device: &CudaDevice, fixture: &Fixture) -> Result<f64> {
        let runtime = device.runtime();
        runtime
            .enqueue(|transfers| transfers.copy(self.state.operand(), self.initial.operand()))?;
        device.tile().sync(self.graph.launch())?;
        let state = maximum_error(&runtime.read(&self.state)?, &fixture.state)?;
        let out = maximum_error(&runtime.read(&self.output)?, &fixture.output)?;
        Ok(state.max(out))
    }
}
fn kernel_identities(device: &CudaDevice, shape: [usize; SHAPE_RANK]) -> Result<Vec<String>> {
    identity::recurrent_wy(&device.stream, shape, &FINGERPRINT_SPLITS)
}

pub(super) fn fingerprint(device: &CudaDevice, shape: [usize; SHAPE_RANK]) -> Result<String> {
    use infer_cuda_kernels::tile::jit::{self as compiler, JitOptions};
    use std::{
        collections::BTreeMap,
        sync::{Mutex, OnceLock},
    };
    type IdentityCache = Mutex<BTreeMap<([usize; SHAPE_RANK], String), String>>;
    static CACHE: OnceLock<IdentityCache> = OnceLock::new();
    // Canonical IR is immutable within this binary. A model may contain dozens of
    // identical Delta nodes: lower each compiler/architecture/shape only once.
    // Compiler overrides remain part of the key, so a changed JIT context misses.
    let context = format!(
        "{}:{}:{:?}:{:?}:{:?}:{:?}",
        device.profile()?.architecture,
        compiler::compiler(),
        compiler::bytecode_override(),
        JitOptions::default(),
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS"),
    );
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let key = (shape, context.clone());
    let cached = cache
        .lock()
        .map_err(|_| Error::invariant("WY identity cache poisoned"))?
        .get(&key)
        .cloned();
    if let Some(value) = cached {
        return Ok(value);
    }
    let value = format!(
        "{}:{}",
        context,
        kernel_identities(device, shape)?.join(":")
    );
    cache
        .lock()
        .map_err(|_| Error::invariant("WY identity cache poisoned"))?
        .insert(key, value.clone());
    Ok(value)
}
