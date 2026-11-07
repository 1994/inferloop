//! Captures retain every operand; timing includes the original append and full attention compute.
use super::{
    Choice, Shape,
    reference::{Fixture, KEY_SCALE, VALUE_SCALE},
};
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::device::{CudaDevice, device_error};
use crate::resident::attention_mma::Arithmetic;
use crate::resident::attention_mma::Support;
use crate::resident::attention_oxide::Support as AttentionOxideSupport;
use crate::resident::attention_oxide::artifact_identity;
use crate::resident::buffers::kv_cache;
use crate::resident::tensor_buffer as buffer;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::{
    AppendBuffers, DecodeBuffers, DecodePlan, Kernels, PrefillAppendPlan, PrefillDecodePlan,
    Recording,
};
use infer_cuda_kernels::identity::{AttentionShape, attention_candidates};
use infer_cuda_kernels::runtime::f8e4m3fn;
use infer_cuda_kernels::runtime::{Allocation, Graph};
use std::env::var_os;
struct Operands {
    query: Allocation<f32>,
    append_keys: Allocation<f32>,
    append_values: Allocation<f32>,
    metadata: Allocation<i32>,
    table: Allocation<i32>,
    keys: Allocation<f8e4m3fn>,
    values: Allocation<f8e4m3fn>,
    output: Allocation<f32>,
}
const fn output_shape(s: Shape, choice: Choice) -> [usize; 2] {
    if choice.query_tile().is_none() {
        [s.lanes * s.heads, s.dim]
    } else {
        [s.lanes, s.heads * s.dim]
    }
}
impl Operands {
    fn new(device: &CudaDevice, f: &Fixture, choice: Choice) -> Result<Self> {
        let s = f.shape;
        Ok(Self {
            query: device.runtime().upload(f.query.clone())?,
            append_keys: device.runtime().upload(f.append_keys.clone())?,
            append_values: device.runtime().upload(f.append_values.clone())?,
            metadata: device.runtime().upload(f.metadata.clone())?,
            table: device.runtime().upload(f.table.clone())?,
            keys: device.runtime().upload(f.keys.clone())?,
            values: device.runtime().upload(f.values.clone())?,
            output: device
                .runtime()
                .zeros(output_shape(s, choice).iter().product())?,
        })
    }
    fn record(
        &self,
        contract: &Kernels,
        scope: &Recording<'_>,
        f: &Fixture,
        choice: Choice,
        oxide: Option<&AttentionOxideSupport>,
        mma: Option<&Support>,
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
                    cache: kv_cache(
                        (&self.keys, &self.values),
                        s.kv_heads,
                        s.dim,
                        [KEY_SCALE, VALUE_SCALE],
                    )?,
                    keys: buffer(&self.append_keys)?,
                    values: buffer(&self.append_values)?,
                    metadata: buffer(&self.metadata)?,
                    table: buffer(&self.table)?,
                },
            )
            .map_err(|error| device_error(error.to_string()))?;
        let window = i32::try_from(f.window).map_err(|e| device_error(e.to_string()))?;
        let geometry = (s.heads, s.kv_heads, s.dim, s.lanes, f.capacity);
        let decode = || -> Result<DecodeBuffers<'_>> {
            Ok(DecodeBuffers {
                output: buffer(&self.output)?,
                query: buffer(&self.query)?,
                cache: kv_cache(
                    (&self.keys, &self.values),
                    s.kv_heads,
                    s.dim,
                    [KEY_SCALE, VALUE_SCALE],
                )?,
                metadata: buffer(&self.metadata)?,
                table: buffer(&self.table)?,
            })
        };
        if let Choice::Mma { variant } = choice {
            mma.ok_or_else(|| device_error("MMA attention missing"))?
                .record(recording, decode()?, geometry, window, variant)?;
        } else if choice.uses_oxide() {
            oxide
                .ok_or_else(|| device_error("Rust attention module missing"))?
                .record(recording, decode()?, geometry, window)?;
        } else {
            let plan = PrefillDecodePlan {
                decode: DecodePlan {
                    heads: s.heads,
                    kv_heads: s.kv_heads,
                    dim: s.dim,
                    window,
                    block: KV_BLOCK_TOKENS,
                },
                lanes: s.lanes,
            };
            let buffers = decode()?;
            choice
                .query_tile()
                .map_or_else(
                    || contract.attention.decode_prefill(recording, plan, buffers),
                    |tile| {
                        contract
                            .attention
                            .decode_prefill_tiled(recording, plan, tile, buffers)
                    },
                )
                .map_err(|error| device_error(error.to_string()))?;
        }
        Ok(())
    }
}
fn identities(device: &CudaDevice, s: Shape, choice: Choice) -> Result<Vec<String>> {
    attention_candidates::<f8e4m3fn>(
        device.tile(),
        &AttentionShape {
            heads: s.heads,
            kv_heads: s.kv_heads,
            dim: s.dim,
            lanes: s.lanes,
            capacity: s.lanes * 2,
            block: KV_BLOCK_TOKENS,
            window: s.window,
            scales: [KEY_SCALE, VALUE_SCALE],
            metadata_fields: METADATA_FIELDS,
        },
        choice.query_tile(),
    )
}
pub(super) struct Captured {
    pub graph: Graph,
    operands: Operands,
    _oxide: Option<AttentionOxideSupport>,
    _mma: Option<Support>,
}
impl Captured {
    pub fn new(device: &CudaDevice, fixture: &Fixture, choice: Choice) -> Result<Self> {
        let oxide = if choice == Choice::SharedKv {
            AttentionOxideSupport::load_shared(device)?
        } else if choice == Choice::VectorSharedKv {
            AttentionOxideSupport::load_vector(device)?
        } else if choice == Choice::VectorSharedKvCombined {
            AttentionOxideSupport::load_combined(device)?
        } else if choice == Choice::RustSimt {
            AttentionOxideSupport::load(device)?
        } else {
            None
        };
        Self::new_with_support(device, fixture, choice, oxide)
    }
    fn new_with_support(
        device: &CudaDevice,
        fixture: &Fixture,
        choice: Choice,
        oxide: Option<AttentionOxideSupport>,
    ) -> Result<Self> {
        let mma = if let Choice::Mma { variant } = choice {
            let mut support = Support::load(
                device,
                fixture.capacity,
                fixture.shape.lanes,
                Arithmetic::Bf16,
            )?;
            support.prepare_variant(device, variant)?;
            Some(support)
        } else {
            None
        };
        let operands = Operands::new(device, fixture, choice)?;
        let contract = device.kernels();
        let graph = device.tile().capture(|scope| {
            operands.record(
                &contract,
                scope,
                fixture,
                choice,
                oxide.as_ref(),
                mma.as_ref(),
            )
        })?;
        Ok(Self {
            graph,
            operands,
            _oxide: oxide,
            _mma: mma,
        })
    }
    #[cfg(test)]
    pub fn candidate(
        device: &CudaDevice,
        fixture: &Fixture,
        support: AttentionOxideSupport,
    ) -> Result<Self> {
        Self::new_with_support(device, fixture, Choice::RustSimt, Some(support))
    }
    #[cfg(test)]
    pub fn graph_variant(
        &self,
        device: &CudaDevice,
        f: &Fixture,
        support: &AttentionOxideSupport,
    ) -> Result<Graph> {
        let contract = device.kernels();
        device.tile().capture(|scope| {
            self.operands
                .record(&contract, scope, f, Choice::RustSimt, Some(support), None)
        })
    }
    #[cfg(test)]
    pub fn verify_variant(&self, device: &CudaDevice, f: &Fixture, graph: &Graph) -> Result<usize> {
        self.verify(device, f)?;
        let expected = device.runtime().read(&self.operands.output)?;
        self.verify_graph(device, f, graph)?;
        let actual = device.runtime().read(&self.operands.output)?;
        if !actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err(Error::invariant("same-address attention bits differ"));
        }
        Ok(actual.len())
    }
    pub fn exact_match(&self, device: &CudaDevice, baseline: &Self) -> Result<usize> {
        device.tile().sync(baseline.graph.launch())?;
        device.tile().sync(self.graph.launch())?;
        let actual = device.runtime().read(&self.operands.output)?;
        let expected = device.runtime().read(&baseline.operands.output)?;
        if actual.len() != expected.len()
            || actual
                .iter()
                .zip(&expected)
                .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
        {
            return Err(Error::invariant(
                "Rust attention original output bits mismatch",
            ));
        }
        Ok(actual.len())
    }
    pub fn verify(&self, device: &CudaDevice, f: &Fixture) -> Result<f64> {
        self.verify_graph(device, f, &self.graph)
    }
    fn verify_graph(&self, device: &CudaDevice, f: &Fixture, graph: &Graph) -> Result<f64> {
        self.verify_graph_limit(device, f, graph, super::MAX_ERROR)
    }
    pub fn verify_mma(&self, device: &CudaDevice, f: &Fixture) -> Result<f64> {
        self.verify_graph_limit(device, f, &self.graph, super::mma::MAX_ERROR)
    }
    pub fn verify_mma_history(&self, device: &CudaDevice, f: &Fixture) -> Result<()> {
        device.tile().sync(self.graph.launch())?;
        let output = device.runtime().read(&self.operands.output)?;
        for (index, &value) in output.iter().enumerate() {
            let row = index / (f.shape.heads * f.shape.dim);
            let inactive = row >= usize::try_from(f.metadata[1]).map_err(device_error)?
                || f.metadata[0] + f.metadata[2] + i32::try_from(row).map_err(device_error)? < 0;
            if !value.is_finite() || inactive && value != 0.0 {
                return Err(Error::invalid("MMA history output or inactive row"));
            }
        }
        Ok(())
    }
    fn verify_graph_limit(
        &self,
        device: &CudaDevice,
        f: &Fixture,
        graph: &Graph,
        limit: f64,
    ) -> Result<f64> {
        device.tile().sync(graph.launch())?;
        for (tensor, expected) in [
            (&self.operands.keys, &f.expected_keys),
            (&self.operands.values, &f.expected_values),
        ] {
            let actual = device.runtime().read(tensor)?;
            if actual.len() != expected.len()
                || actual.iter().zip(expected).any(|(a, b)| a.0 != b.0)
            {
                return Err(Error::invariant("attention appended FP8 cache mismatch"));
            }
        }
        let actual = device.runtime().read(&self.operands.output)?;
        if actual.len() != f.expected.len() {
            return Err(Error::invariant("attention output size"));
        }
        let mut maximum = 0.0_f64;
        for (&a, &b) in actual.iter().zip(&f.expected) {
            if !a.is_finite() || !b.is_finite() {
                return Err(Error::invariant("attention output nonfinite"));
            }
            maximum = maximum.max((f64::from(a) - b).abs() / b.abs().max(1.0));
        }
        if maximum >= limit {
            return Err(Error::invalid(format!(
                "attention full F64 error {maximum}"
            )));
        }
        Ok(maximum)
    }
}
pub(super) fn fingerprint(device: &CudaDevice, shape: Shape) -> Result<String> {
    use infer_cuda_kernels::jit::{self as compiler, JitOptions};
    use std::{
        collections::BTreeMap,
        sync::{Mutex, OnceLock},
    };
    type Cache = Mutex<BTreeMap<(Shape, String), String>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let context = format!(
        "{}:{}:{:?}:{:?}:{:?}:{:?}",
        device.profile()?.architecture,
        compiler::compiler(),
        compiler::bytecode_override(),
        JitOptions::default(),
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS")
    );
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let key = (shape, context.clone());
    let cached = cache
        .lock()
        .map_err(|_| Error::invariant("attention identity poisoned"))?
        .get(&key)
        .cloned();
    if let Some(value) = cached {
        return Ok(value);
    }
    let mut all = Vec::new();
    for choice in [
        Choice::Simt,
        Choice::Tiled {
            query_tile: super::QUERY_TILES[0],
        },
        Choice::Tiled {
            query_tile: super::QUERY_TILES[1],
        },
    ] {
        all.extend(identities(device, shape, choice)?);
    }
    if all.iter().any(String::is_empty) {
        return Err(Error::invalid("incomplete attention kernel identity"));
    }
    let value = format!("{}:{}", context, all.join(":"));
    cache
        .lock()
        .map_err(|_| Error::invariant("attention identity poisoned"))?
        .insert(key, value.clone());
    Ok(value)
}

pub(super) fn native_fingerprint(device: &CudaDevice, shape: Shape) -> Result<String> {
    use infer_cuda_kernels::jit::{self as compiler, JitOptions};
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        sync::{Mutex, OnceLock},
    };
    type Cache = Mutex<BTreeMap<(Shape, String), String>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let context = serde_json::to_string(&(
        device.profile()?.architecture.as_str(),
        compiler::compiler(),
        compiler::bytecode_override(),
        format!("{:?}", JitOptions::default()),
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS"),
        artifact_identity(),
    ))
    .map_err(device_error)?;
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let memo_key = (shape, context);
    if let Some(identity) = cache
        .lock()
        .map_err(|_| Error::invariant("exact attention identity cache poisoned"))?
        .get(&memo_key)
    {
        return Ok(identity.clone());
    }
    let identities = identities(device, shape, Choice::Simt)?;
    if identities.iter().any(String::is_empty) {
        return Err(Error::invalid("incomplete native attention identity"));
    }
    let payload = serde_json::to_vec(&(
        identities,
        device.profile()?.architecture.as_str(),
        compiler::compiler(),
        compiler::bytecode_override(),
        format!("{:?}", JitOptions::default()),
        var_os("CUTILE_DISABLE_CHECK_HOISTING"),
        var_os("CUTILE_FORCE_DEVICE_CHECKS"),
        artifact_identity(),
    ))
    .map_err(device_error)?;
    let identity = format!("{:x}", Sha256::digest(payload));
    cache
        .lock()
        .map_err(|_| Error::invariant("exact attention identity cache poisoned"))?
        .insert(memo_key, identity.clone());
    Ok(identity)
}

#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_append_headers.rs"]
mod append_header_tests;
