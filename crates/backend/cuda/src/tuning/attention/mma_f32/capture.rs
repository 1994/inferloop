//! Original F32 append plus either original SIMT or page gathering/AOT/readout.
use super::{
    super::{Choice, Shape},
    reference::{self, Fixture},
};
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::device::{CudaDevice, device_error};
use crate::resident::attention_mma;
use crate::resident::buffers::kv_cache;
use crate::resident::tensor_buffer as buffer;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    AppendBuffers, DecodeBuffers, DecodePlan, Kernels, PrefillAppendPlan, PrefillDecodePlan,
    Recording,
};
use infer_cuda_kernels::identity::{AttentionShape, attention_candidates};
use infer_cuda_kernels::jit::{self as compiler, JitOptions};
use infer_cuda_kernels::runtime::{Allocation, Graph};
use std::env::var_os;
struct Operands {
    query: Allocation<f32>,
    append_keys: Allocation<f32>,
    append_values: Allocation<f32>,
    metadata: Allocation<i32>,
    table: Allocation<i32>,
    keys: Allocation<f32>,
    values: Allocation<f32>,
    output: Allocation<f32>,
}
impl Operands {
    fn new(device: &CudaDevice, f: &Fixture) -> Result<Self> {
        let s = f.shape;
        let runtime = device.runtime();
        Ok(Self {
            query: device.runtime().upload(f.query.clone())?,
            append_keys: device.runtime().upload(f.append_keys.clone())?,
            append_values: device.runtime().upload(f.append_values.clone())?,
            metadata: device.runtime().upload(f.metadata.clone())?,
            table: device.runtime().upload(f.table.clone())?,
            keys: runtime.upload(f.keys.clone())?,
            values: runtime.upload(f.values.clone())?,
            output: runtime.zeros(s.lanes * s.heads * s.dim)?,
        })
    }
    fn record(
        &self,
        contract: &Kernels,
        scope: &Recording<'_>,
        f: &Fixture,
        choice: Choice,
        support: Option<&attention_mma::Support>,
    ) -> Result<()> {
        let s = f.shape;
        let recording = scope;
        contract
            .attention
            .append_prefill(
                recording,
                PrefillAppendPlan {
                    lanes: s.lanes,
                    rows: f.capacity,
                    block: KV_BLOCK_TOKENS,
                },
                AppendBuffers {
                    cache: kv_cache((&self.keys, &self.values), s.kv_heads, s.dim, [1.0, 1.0])?,
                    keys: buffer(&self.append_keys)?,
                    values: buffer(&self.append_values)?,
                    metadata: buffer(&self.metadata)?,
                    table: buffer(&self.table)?,
                },
            )
            .map_err(|error| device_error(error.to_string()))?;
        if let Choice::Mma { variant } = choice {
            support
                .ok_or_else(|| device_error("F32 MMA module missing"))?
                .record(
                    recording,
                    DecodeBuffers {
                        output: buffer(&self.output)?,
                        query: buffer(&self.query)?,
                        cache: kv_cache((&self.keys, &self.values), s.kv_heads, s.dim, [1.0, 1.0])?,
                        metadata: buffer(&self.metadata)?,
                        table: buffer(&self.table)?,
                    },
                    (s.heads, s.kv_heads, s.dim, s.lanes, f.capacity),
                    0,
                    variant,
                )?;
        } else {
            contract
                .attention
                .decode_prefill(
                    recording,
                    PrefillDecodePlan {
                        decode: DecodePlan {
                            heads: s.heads,
                            kv_heads: s.kv_heads,
                            dim: s.dim,
                            window: 0,
                            block: KV_BLOCK_TOKENS,
                        },
                        lanes: s.lanes,
                    },
                    DecodeBuffers {
                        output: buffer(&self.output)?,
                        query: buffer(&self.query)?,
                        cache: kv_cache((&self.keys, &self.values), s.kv_heads, s.dim, [1.0, 1.0])?,
                        metadata: buffer(&self.metadata)?,
                        table: buffer(&self.table)?,
                    },
                )
                .map_err(|error| device_error(error.to_string()))?;
        }
        Ok(())
    }
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
        let support = if let Choice::Mma { variant } = choice {
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
        let operands = Operands::new(device, f)?;
        let contract = device.kernels();
        let graph = device
            .tile()
            .capture(|scope| operands.record(&contract, scope, f, choice, support.as_ref()))?;
        Ok(Self {
            graph,
            operands,
            _support: support,
        })
    }
    pub fn verify(&self, device: &CudaDevice, f: &Fixture, complete: bool) -> Result<Proof> {
        device.tile().sync(self.graph.launch())?;
        for (tensor, expected) in [
            (&self.operands.keys, &f.expected_keys),
            (&self.operands.values, &f.expected_values),
        ] {
            let actual = device.runtime().read(tensor)?;
            if actual.len() != expected.len()
                || actual
                    .iter()
                    .zip(expected)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(Error::invalid("F32 append cache bits differ"));
            }
        }
        let actual = device.runtime().read(&self.operands.output)?;
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
    let [append, compute]: [String; 2] = attention_candidates::<f32>(
        device.tile(),
        &AttentionShape {
            heads: s.heads,
            kv_heads: s.kv_heads,
            dim: s.dim,
            lanes: s.lanes,
            capacity: (s.lanes * 2).next_multiple_of(KV_BLOCK_TOKENS),
            block: KV_BLOCK_TOKENS,
            window: 0,
            scales: [1.0, 1.0],
            metadata_fields: METADATA_FIELDS,
        },
        None,
    )?
    .try_into()
    .map_err(|_| Error::invariant("F32 attention identity count"))?;
    let payload = serde_json::to_vec(&(
        append,
        compute,
        attention_mma::f32_artifact_identity(),
        &device.profile()?.architecture,
        compiler::compiler(),
        compiler::bytecode_override(),
        format!("{:?}", JitOptions::default()),
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS"),
    ))
    .map_err(device_error)?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}

#[cfg(test)]
#[path = "../../../../tests/unit/tuning_attention_f32_append_headers.rs"]
mod append_header_tests;
