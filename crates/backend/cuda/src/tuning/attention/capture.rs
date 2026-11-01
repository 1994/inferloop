//! Captures retain every operand; timing includes the original append and full attention compute.
use super::{
    Choice, Shape,
    reference::{Fixture, KEY_SCALE, VALUE_SCALE},
};
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::resident::attention_mma::Arithmetic;
use crate::resident::attention_mma::Support;
use crate::resident::attention_oxide::Support as AttentionOxideSupport;
use crate::resident::attention_oxide::artifact_identity;
use crate::{
    device::{CudaDevice, device_error},
    resident::attention_prefill::kernels,
};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, Result};
use std::env::var_os;
use std::result::Result as StdResult;
struct Operands {
    query: Arc<Tensor<f32>>,
    append_keys: Arc<Tensor<f32>>,
    append_values: Arc<Tensor<f32>>,
    metadata: Arc<Tensor<i32>>,
    table: Arc<Tensor<i32>>,
    keys: Tensor<f8e4m3fn>,
    values: Tensor<f8e4m3fn>,
    output: Tensor<f32>,
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
            query: device.upload(f.query.clone(), &[s.lanes, s.heads * s.dim])?,
            append_keys: device.upload(f.append_keys.clone(), &[s.lanes, s.kv_heads, s.dim])?,
            append_values: device.upload(f.append_values.clone(), &[s.lanes, s.kv_heads, s.dim])?,
            metadata: device.upload(f.metadata.clone(), &[METADATA_FIELDS])?,
            table: device.upload(f.table.clone(), &[f.table.len()])?,
            keys: api::copy_host_vec_to_device(&Arc::new(f.keys.clone()))
                .sync_on(&device.stream)
                .map_err(device_error)?
                .reshape(&[s.kv_heads, f.capacity, s.dim])
                .map_err(device_error)?,
            values: api::copy_host_vec_to_device(&Arc::new(f.values.clone()))
                .sync_on(&device.stream)
                .map_err(device_error)?
                .reshape(&[s.kv_heads, f.capacity, s.dim])
                .map_err(device_error)?,
            output: api::zeros::<f32>(&output_shape(s, choice))
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }
    fn meta(device: &CudaDevice, s: Shape, choice: Choice) -> Result<Self> {
        let capacity = s.lanes * 2;
        let blocks = capacity / KV_BLOCK_TOKENS;
        Ok(Self {
            query: Arc::new(
                api::meta::<f32>(&[s.lanes, s.heads * s.dim])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
            append_keys: Arc::new(
                api::meta::<f32>(&[s.lanes, s.kv_heads, s.dim])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
            append_values: Arc::new(
                api::meta::<f32>(&[s.lanes, s.kv_heads, s.dim])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
            metadata: Arc::new(
                api::meta::<i32>(&[METADATA_FIELDS])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
            table: Arc::new(
                api::meta::<i32>(&[blocks])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ),
            keys: api::meta::<f8e4m3fn>(&[s.kv_heads, capacity, s.dim])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            values: api::meta::<f8e4m3fn>(&[s.kv_heads, capacity, s.dim])
                .sync_on(&device.stream)
                .map_err(device_error)?,
            output: api::meta::<f32>(&output_shape(s, choice))
                .sync_on(&device.stream)
                .map_err(device_error)?,
        })
    }
    fn append_generics(s: Shape, capacity: usize) -> Vec<String> {
        vec![
            f8e4m3fn::DTYPE.as_str().into(),
            s.dim.to_string(),
            capacity.to_string(),
            "1".into(),
            KV_BLOCK_TOKENS.to_string(),
            s.lanes
                .div_ceil(KV_BLOCK_TOKENS)
                .next_power_of_two()
                .to_string(),
        ]
    }
    fn compute_generics(s: Shape, choice: Choice) -> Vec<String> {
        let mut g = vec![
            f8e4m3fn::DTYPE.as_str().into(),
            s.dim.to_string(),
            (s.heads / s.kv_heads).to_string(),
        ];
        if let Some(q) = choice.query_tile() {
            g.extend([
                q.to_string(),
                KV_BLOCK_TOKENS.to_string(),
                KV_BLOCK_TOKENS.to_string(),
                "1".into(),
            ]);
        } else {
            g.extend([s.heads.to_string(), KV_BLOCK_TOKENS.to_string()]);
        }
        g
    }
    fn record(
        &mut self,
        scope: &Scope,
        f: &Fixture,
        choice: Choice,
        oxide: Option<&AttentionOxideSupport>,
        mma: Option<&mut Support>,
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
                KEY_SCALE,
                VALUE_SCALE,
            )
            .generics(Self::append_generics(s, f.capacity)),
        )?;
        let window = i32::try_from(f.window).map_err(|e| DeviceError::Internal(e.to_string()))?;
        if let Choice::Mma { variant } = choice {
            mma.ok_or_else(|| DeviceError::Internal("MMA attention missing".into()))?
                .record(
                    scope,
                    crate::resident::attention_mma::Bindings {
                        output: &mut self.output,
                        query: &self.query,
                        keys: &self.keys,
                        values: &self.values,
                        metadata: &self.metadata,
                        table: &self.table,
                    },
                    (s.heads, s.kv_heads, s.dim, s.lanes, f.capacity),
                    window,
                    [KEY_SCALE, VALUE_SCALE],
                    variant,
                )?;
        } else if choice.uses_oxide() {
            oxide
                .ok_or_else(|| DeviceError::Internal("Rust attention module missing".into()))?
                .record(
                    scope,
                    crate::resident::attention_oxide::Bindings {
                        output: &mut self.output,
                        query: &self.query,
                        keys: &self.keys,
                        values: &self.values,
                        metadata: &self.metadata,
                        table: &self.table,
                    },
                    (s.heads, s.kv_heads, s.dim, s.lanes, f.capacity),
                    window,
                    [KEY_SCALE, VALUE_SCALE],
                )?;
        } else if let Some(qt) = choice.query_tile() {
            let scale = 1.0
                / f32::from(
                    u16::try_from(s.dim).map_err(|e| DeviceError::Internal(e.to_string()))?,
                )
                .sqrt();
            scope.record(
                kernels::decode_tiled(
                    (&mut self.output).partition([qt, s.dim]),
                    &self.query,
                    &self.keys,
                    &self.values,
                    &self.metadata,
                    &self.table,
                    window,
                    KEY_SCALE,
                    VALUE_SCALE,
                    scale,
                )
                .generics(Self::compute_generics(s, choice)),
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
                    window,
                    KEY_SCALE,
                    VALUE_SCALE,
                )
                .generics(Self::compute_generics(s, choice)),
            )?;
        }
        Ok(())
    }
    fn identities(&mut self, device: &CudaDevice, s: Shape, choice: Choice) -> Result<Vec<String>> {
        let capacity = s.lanes * 2;
        let append = kernels::append(
            (&mut self.keys).partition([1, capacity, s.dim]),
            (&mut self.values).partition([1, capacity, s.dim]),
            &self.append_keys,
            &self.append_values,
            &self.metadata,
            &self.table,
            KEY_SCALE,
            VALUE_SCALE,
        )
        .generics(Self::append_generics(s, capacity))
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?;
        let window = i32::try_from(s.window).map_err(device_error)?;
        let compute = if let Some(qt) = choice.query_tile() {
            let scale = 1.0 / f32::from(u16::try_from(s.dim).map_err(device_error)?).sqrt();
            kernels::decode_tiled(
                (&mut self.output).partition([qt, s.dim]),
                &self.query,
                &self.keys,
                &self.values,
                &self.metadata,
                &self.table,
                window,
                KEY_SCALE,
                VALUE_SCALE,
                scale,
            )
            .generics(Self::compute_generics(s, choice))
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?
        } else {
            kernels::decode(
                (&mut self.output).partition([1, s.dim]),
                &self
                    .query
                    .view(&[s.lanes * s.heads, s.dim])
                    .map_err(device_error)?,
                &self.keys,
                &self.values,
                &self.metadata,
                &self.table,
                window,
                KEY_SCALE,
                VALUE_SCALE,
            )
            .generics(Self::compute_generics(s, choice))
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?
        };
        Ok(vec![append, compute])
    }
}
pub(super) struct Captured {
    pub graph: CudaGraph<()>,
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
        let mut mma = if let Choice::Mma { variant } = choice {
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
        let mut operands = Operands::new(device, fixture, choice)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            operands.record(scope, fixture, choice, oxide.as_ref(), mma.as_mut())
        })
        .map_err(device_error)?;
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
        &mut self,
        device: &CudaDevice,
        f: &Fixture,
        support: &AttentionOxideSupport,
    ) -> Result<CudaGraph<()>> {
        CudaGraph::scope(&device.stream, |scope| {
            self.operands
                .record(scope, f, Choice::RustSimt, Some(support), None)
        })
        .map_err(device_error)
    }
    #[cfg(test)]
    pub fn verify_variant(
        &self,
        device: &CudaDevice,
        f: &Fixture,
        graph: &CudaGraph<()>,
    ) -> Result<usize> {
        self.verify(device, f)?;
        let expected = device.read_borrowed(&self.operands.output)?;
        self.verify_graph(device, f, graph)?;
        let actual = device.read_borrowed(&self.operands.output)?;
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
        baseline
            .graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let actual = device.read_borrowed(&self.operands.output)?;
        let expected = device.read_borrowed(&baseline.operands.output)?;
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
    fn verify_graph(&self, device: &CudaDevice, f: &Fixture, graph: &CudaGraph<()>) -> Result<f64> {
        self.verify_graph_limit(device, f, graph, super::MAX_ERROR)
    }
    pub fn verify_mma(&self, device: &CudaDevice, f: &Fixture) -> Result<f64> {
        self.verify_graph_limit(device, f, &self.graph, super::mma::MAX_ERROR)
    }
    pub fn verify_mma_history(&self, device: &CudaDevice, f: &Fixture) -> Result<()> {
        self.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let output = device.read_borrowed(&self.operands.output)?;
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
        graph: &CudaGraph<()>,
        limit: f64,
    ) -> Result<f64> {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(device_error)?;
        for (tensor, expected) in [
            (&self.operands.keys, &f.expected_keys),
            (&self.operands.values, &f.expected_values),
        ] {
            let actual = device.read_borrowed(tensor)?;
            if actual.len() != expected.len()
                || actual.iter().zip(expected).any(|(a, b)| a.0 != b.0)
            {
                return Err(Error::invariant("attention appended FP8 cache mismatch"));
            }
        }
        let actual = device.read_borrowed(&self.operands.output)?;
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
    use cutile::cutile_compiler::{cuda_tile_runtime_utils as compiler, hints::CompileOptions};
    use std::{
        collections::BTreeMap,
        sync::{Mutex, OnceLock},
    };
    type Cache = Mutex<BTreeMap<(Shape, String), String>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let context = format!(
        "{}:{}:{:?}:{:?}:{:?}:{:?}",
        device.profile()?.architecture,
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        CompileOptions::default(),
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
    let mut identities = Vec::new();
    for choice in [
        Choice::Simt,
        Choice::Tiled {
            query_tile: super::QUERY_TILES[0],
        },
        Choice::Tiled {
            query_tile: super::QUERY_TILES[1],
        },
    ] {
        identities
            .extend(Operands::meta(device, shape, choice)?.identities(device, shape, choice)?);
    }
    if identities.iter().any(String::is_empty) {
        return Err(Error::invalid("incomplete attention kernel identity"));
    }
    let value = format!("{}:{}", context, identities.join(":"));
    cache
        .lock()
        .map_err(|_| Error::invariant("attention identity poisoned"))?
        .insert(key, value.clone());
    Ok(value)
}

pub(super) fn native_fingerprint(device: &CudaDevice, shape: Shape) -> Result<String> {
    use cutile::cutile_compiler::{cuda_tile_runtime_utils as compiler, hints::CompileOptions};
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        sync::{Mutex, OnceLock},
    };
    type Cache = Mutex<BTreeMap<(Shape, String), String>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let context = serde_json::to_string(&(
        device.profile()?.architecture.as_str(),
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        format!("{:?}", CompileOptions::default()),
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
    let identities =
        Operands::meta(device, shape, Choice::Simt)?.identities(device, shape, Choice::Simt)?;
    if identities.iter().any(String::is_empty) {
        return Err(Error::invalid("incomplete native attention identity"));
    }
    let payload = serde_json::to_vec(&(
        identities,
        device.profile()?.architecture.as_str(),
        compiler::tileiras_fingerprint(),
        compiler::bytecode_override(),
        format!("{:?}", CompileOptions::default()),
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
