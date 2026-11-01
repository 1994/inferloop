//! Capture owns graph before its operand buffers; replay timing includes output transpose.
use super::{
    SHAPE_RANK,
    reference::{Fixture, maximum_error},
};
use infer_cuda_kernels::tile::runtime::Graph;
use std::env::var_os;

const METADATA_LEN: usize = 4;
const FINGERPRINT_SPLITS: [usize; 3] = [1, 2, 4];
use crate::{
    device::{CudaDevice, device_error},
    resident::recurrent_prefill::kernels,
};
use cutile::prelude::*;
use infer_core::{Error, Result};

pub(super) struct Captured {
    pub graph: Graph,
    state: Tensor<f32>,
    output: Tensor<f32>,
    _scratch: Tensor<f32>,
    _qkv: Arc<Tensor<f32>>,
    _beta: Arc<Tensor<f32>>,
    _alpha: Arc<Tensor<f32>>,
    _a_log: Arc<Tensor<f32>>,
    _bias: Arc<Tensor<f32>>,
    initial: Arc<Tensor<f32>>,
    _metadata: Arc<Tensor<i32>>,
}
impl Captured {
    pub fn new(device: &CudaDevice, fixture: &Fixture, block: usize) -> Result<Self> {
        let [kh, vh, dim, lanes] = fixture.shape;
        let qkv = device.upload(fixture.qkv.clone(), &[lanes, 2 * kh + vh, dim])?;
        let beta = device.upload(fixture.beta.clone(), &[lanes, vh])?;
        let alpha = device.upload(fixture.alpha.clone(), &[lanes, vh])?;
        let a_log = device.upload(fixture.a_log.clone(), &[vh])?;
        let bias = device.upload(fixture.bias.clone(), &[vh])?;
        let initial = device.upload(fixture.initial.clone(), &[vh, dim, dim])?;
        let metadata = device.upload(
            vec![0, i32::try_from(lanes).map_err(device_error)?, 0, 0],
            &[METADATA_LEN],
        )?;
        let mut state = api::zeros::<f32>(&[vh, dim, dim])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut scratch = api::zeros::<f32>(&[vh, lanes, dim])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut output = api::zeros::<f32>(&[lanes, vh, dim])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let graph = Graph::scope(&device.stream, |scope| {
            scope.record(
                kernels::delta_wy(
                    (&mut state).partition([1, dim, block]),
                    (&mut scratch).partition([1, lanes, block]),
                    &qkv,
                    &beta,
                    &alpha,
                    &a_log,
                    &bias,
                    &metadata,
                )
                .generics(vec![
                    kh.to_string(),
                    vh.to_string(),
                    dim.to_string(),
                    lanes.to_string(),
                    block.to_string(),
                ]),
            )?;
            scope.record(
                kernels::transpose((&mut output).partition([1, 1, dim]), &scratch)
                    .generics(vec![dim.to_string()]),
            )?;
            Ok(())
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            state,
            output,
            _scratch: scratch,
            _qkv: qkv,
            _beta: beta,
            _alpha: alpha,
            _a_log: a_log,
            _bias: bias,
            initial,
            _metadata: metadata,
        })
    }
    pub fn verify(&mut self, device: &CudaDevice, fixture: &Fixture) -> Result<f64> {
        api::memcpy(&mut self.state, &self.initial)
            .sync_on(&device.stream)
            .map_err(device_error)?;
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let state = maximum_error(&device.read_borrowed(&self.state)?, &fixture.state)?;
        let out = maximum_error(&device.read_borrowed(&self.output)?, &fixture.output)?;
        Ok(state.max(out))
    }
}
fn kernel_identities(device: &CudaDevice, shape: [usize; SHAPE_RANK]) -> Result<Vec<String>> {
    let [kh, vh, dim, lanes] = shape;
    let qkv = api::meta::<f32>(&[lanes, 2 * kh + vh, dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let beta = api::meta::<f32>(&[lanes, vh])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let alpha = api::meta::<f32>(&[lanes, vh])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let a_log = api::meta::<f32>(&[vh])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let bias = api::meta::<f32>(&[vh])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let metadata = api::meta::<i32>(&[METADATA_LEN])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut state = api::meta::<f32>(&[vh, dim, dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut scratch = api::meta::<f32>(&[vh, lanes, dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::meta::<f32>(&[lanes, vh, dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut identities = Vec::new();
    for split in FINGERPRINT_SPLITS {
        let block = dim / split;
        identities.push(
            kernels::delta_wy(
                (&mut state).partition([1, dim, block]),
                (&mut scratch).partition([1, lanes, block]),
                &qkv,
                &beta,
                &alpha,
                &a_log,
                &bias,
                &metadata,
            )
            .generics(vec![
                kh.to_string(),
                vh.to_string(),
                dim.to_string(),
                lanes.to_string(),
                block.to_string(),
            ])
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?,
        );
    }
    identities.push(
        kernels::transpose((&mut output).partition([1, 1, dim]), &scratch)
            .generics(vec![dim.to_string()])
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?,
    );
    if identities.iter().any(String::is_empty) {
        return Err(Error::invalid("incomplete WY kernel identity"));
    }
    Ok(identities)
}

pub(super) fn fingerprint(device: &CudaDevice, shape: [usize; SHAPE_RANK]) -> Result<String> {
    use cutile::cutile_compiler::{cuda_tile_runtime_utils as compiler, hints::CompileOptions};
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
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        CompileOptions::default(),
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
