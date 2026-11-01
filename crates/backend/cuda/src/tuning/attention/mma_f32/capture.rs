//! Original F32 append plus either original SIMT or page gathering/AOT/readout.
use super::{
    super::{Choice, Shape},
    reference::{self, Fixture},
};
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::{
    device::{CudaDevice, device_error},
    resident::{attention_mma, attention_prefill::kernels},
};
use cutile::{
    cutile_compiler::{cuda_tile_runtime_utils as compiler, hints::CompileOptions},
    prelude::*,
};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use std::env::var_os;
use std::result::Result as StdResult;
struct Operands {
    query: Arc<Tensor<f32>>,
    append_keys: Arc<Tensor<f32>>,
    append_values: Arc<Tensor<f32>>,
    metadata: Arc<Tensor<i32>>,
    table: Arc<Tensor<i32>>,
    keys: Tensor<f32>,
    values: Tensor<f32>,
    output: Tensor<f32>,
}
impl Operands {
    fn new(device: &CudaDevice, f: &Fixture) -> Result<Self> {
        let s = f.shape;
        let copy = |v: Vec<f32>| {
            api::copy_host_vec_to_device(&Arc::new(v))
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        Ok(Self {
            query: device.upload(f.query.clone(), &[s.lanes, s.heads * s.dim])?,
            append_keys: device.upload(f.append_keys.clone(), &[s.lanes, s.kv_heads, s.dim])?,
            append_values: device.upload(f.append_values.clone(), &[s.lanes, s.kv_heads, s.dim])?,
            metadata: device.upload(f.metadata.clone(), &[METADATA_FIELDS])?,
            table: device.upload(f.table.clone(), &[f.table.len()])?,
            keys: copy(f.keys.clone())?
                .reshape(&[s.kv_heads, f.capacity, s.dim])
                .map_err(device_error)?,
            values: copy(f.values.clone())?
                .reshape(&[s.kv_heads, f.capacity, s.dim])
                .map_err(device_error)?,
            output: api::zeros::<f32>(&[s.lanes * s.heads, s.dim])
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }
    fn record(
        &mut self,
        scope: &Scope,
        f: &Fixture,
        choice: Choice,
        support: Option<&mut attention_mma::Support>,
    ) -> StdResult<(), DeviceError> {
        let s = f.shape;
        scope.record(
            kernels::append(
                (&mut self.keys).partition([1, f.capacity, s.dim]),
                (&mut self.values).partition([1, f.capacity, s.dim]),
                &self.append_keys,
                &self.append_values,
                &self.metadata,
                &self.table,
                1.0,
                1.0,
            )
            .generics(append_generics(s, f.capacity)),
        )?;
        if let Choice::Mma { variant } = choice {
            support
                .ok_or_else(|| DeviceError::Internal("F32 MMA module missing".into()))?
                .record(
                    scope,
                    attention_mma::Bindings {
                        output: &mut self.output,
                        query: &self.query,
                        keys: &self.keys,
                        values: &self.values,
                        metadata: &self.metadata,
                        table: &self.table,
                    },
                    (s.heads, s.kv_heads, s.dim, s.lanes, f.capacity),
                    0,
                    [1.0, 1.0],
                    variant,
                )?;
        } else {
            scope.record(
                kernels::decode(
                    (&mut self.output).partition([1, s.dim]),
                    &self.query.view(&[s.lanes * s.heads, s.dim])?,
                    &self.keys,
                    &self.values,
                    &self.metadata,
                    &self.table,
                    0,
                    1.0,
                    1.0,
                )
                .generics(compute_generics(s)),
            )?;
        }
        Ok(())
    }
}
fn append_generics(s: Shape, capacity: usize) -> Vec<String> {
    vec![
        f32::DTYPE.as_str().into(),
        s.dim.to_string(),
        capacity.to_string(),
        "0".into(),
        KV_BLOCK_TOKENS.to_string(),
        s.lanes
            .div_ceil(KV_BLOCK_TOKENS)
            .next_power_of_two()
            .to_string(),
    ]
}
fn compute_generics(s: Shape) -> Vec<String> {
    vec![
        f32::DTYPE.as_str().into(),
        s.dim.to_string(),
        (s.heads / s.kv_heads).to_string(),
        s.heads.to_string(),
        KV_BLOCK_TOKENS.to_string(),
    ]
}
pub(super) struct Captured {
    pub graph: Graph,
    operands: Operands,
    _support: Option<attention_mma::Support>,
}
#[derive(Default)]
pub(super) struct Proof {
    pub elements: usize,
    pub rows: usize,
    pub max_error: f64,
    pub relative_l2: f64,
}
impl Captured {
    pub fn new(device: &CudaDevice, f: &Fixture, choice: Choice) -> Result<Self> {
        let mut support = if let Choice::Mma { variant } = choice {
            let mut module = attention_mma::Support::load(
                device,
                f.capacity,
                f.shape.lanes,
                attention_mma::Arithmetic::Bf16,
            )?;
            module.prepare_float_cache(device)?;
            module.prepare_variant(device, variant)?;
            Some(module)
        } else {
            None
        };
        let mut operands = Operands::new(device, f)?;
        let graph = Graph::scope(&device.stream, |scope| {
            operands.record(scope, f, choice, support.as_mut())
        })
        .map_err(device_error)?;
        Ok(Self {
            graph,
            operands,
            _support: support,
        })
    }
    pub fn verify(&self, device: &CudaDevice, f: &Fixture, complete: bool) -> Result<Proof> {
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        for (tensor, expected) in [
            (&self.operands.keys, &f.expected_keys),
            (&self.operands.values, &f.expected_values),
        ] {
            let actual = device.read_borrowed(tensor)?;
            if actual.len() != expected.len()
                || actual
                    .iter()
                    .zip(expected)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(Error::invalid("F32 append cache bits differ"));
            }
        }
        let actual = device.read_borrowed(&self.operands.output)?;
        if actual.iter().any(|a| !a.is_finite()) {
            return Err(Error::invalid("F32 MMA nonfinite output"));
        }
        let s = f.shape;
        let count = usize::try_from(f.metadata[1]).map_err(device_error)?;
        let rows = if complete {
            (0..s.lanes).collect::<Vec<_>>()
        } else {
            vec![0, count.saturating_sub(1), s.lanes - 1]
        };
        let mut proof = Proof::default();
        for row in 0..s.lanes {
            let inactive = row >= count
                || f.metadata[0] + f.metadata[2] + i32::try_from(row).map_err(device_error)? < 0;
            if inactive
                && actual[row * s.heads * s.dim..(row + 1) * s.heads * s.dim]
                    .iter()
                    .any(|a| a.to_bits() != 0)
            {
                return Err(Error::invalid("F32 MMA inactive rows not zero"));
            }
        }
        for row in rows {
            let heads = if complete {
                (0..s.heads).collect::<Vec<_>>()
            } else {
                vec![0, s.heads - 1]
            };
            for head in heads {
                let reference = f.reference(row, head)?;
                let start = (row * s.heads + head) * s.dim;
                let got = &actual[start..start + s.dim];
                let mut error2 = 0.0;
                let mut norm2 = 0.0;
                for (&a, &b) in got.iter().zip(&reference) {
                    let delta = f64::from(a) - b;
                    proof.max_error = proof.max_error.max(delta.abs() / b.abs().max(1.0));
                    error2 = delta.mul_add(delta, error2);
                    norm2 = b.mul_add(b, norm2);
                }
                proof.relative_l2 = proof
                    .relative_l2
                    .max(error2.sqrt() / norm2.sqrt().max(f64::EPSILON));
                proof.rows += 1;
                proof.elements += s.dim;
            }
        }
        if proof.max_error >= reference::MAX_ERROR
            || proof.relative_l2 >= reference::MAX_RELATIVE_L2
        {
            return Err(Error::invalid(format!(
                "F32 MMA independent F64 error {}, relative L2 {}",
                proof.max_error, proof.relative_l2
            )));
        }
        Ok(proof)
    }
}
pub(super) fn fingerprint(device: &CudaDevice, s: Shape) -> Result<String> {
    use sha2::{Digest, Sha256};
    let cap = (s.lanes * 2).next_multiple_of(KV_BLOCK_TOKENS);
    let mut keys = api::meta::<f32>(&[s.kv_heads, cap, s.dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut vals = api::meta::<f32>(&[s.kv_heads, cap, s.dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut out = api::meta::<f32>(&[s.lanes * s.heads, s.dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let q = api::meta::<f32>(&[s.lanes * s.heads, s.dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let kv = api::meta::<f32>(&[s.lanes, s.kv_heads, s.dim])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let meta = api::meta::<i32>(&[METADATA_FIELDS])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let table = api::meta::<i32>(&[cap / KV_BLOCK_TOKENS])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let append = kernels::append(
        (&mut keys).partition([1, cap, s.dim]),
        (&mut vals).partition([1, cap, s.dim]),
        &kv,
        &kv,
        &meta,
        &table,
        1.0,
        1.0,
    )
    .generics(append_generics(s, cap))
    .l2_cache_key_on(&device.stream)
    .map_err(device_error)?;
    let compute = kernels::decode(
        (&mut out).partition([1, s.dim]),
        &q,
        &keys,
        &vals,
        &meta,
        &table,
        0,
        1.0,
        1.0,
    )
    .generics(compute_generics(s))
    .l2_cache_key_on(&device.stream)
    .map_err(device_error)?;
    let payload = serde_json::to_vec(&(
        append,
        compute,
        attention_mma::f32_artifact_identity(),
        include_str!("reference.rs"),
        include_str!("capture.rs"),
        include_str!("../mma_f32.rs"),
        include_str!("../mma.rs"),
        &device.profile()?.architecture,
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        format!("{:?}", CompileOptions::default()),
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS"),
    ))
    .map_err(device_error)?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}

#[cfg(test)]
#[path = "../../../../tests/unit/tuning_attention_f32_append_headers.rs"]
mod append_header_tests;
