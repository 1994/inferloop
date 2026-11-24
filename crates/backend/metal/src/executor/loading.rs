use super::{MetalBackend, MetalConfig, MetalDevice, allocate_kv, bytes};
use crate::constants::COMPLETION_POOL_SIZE;
use crate::constants::F32_BYTES;
use crate::constants::F32_BYTES_U64;
use crate::constants::MAX_PREFIX_ENTRIES;
use crate::constants::MAX_TICKET_TASKS;
use crate::constants::MIB;
use infer_core::digest::lower_hex;
use infer_core::{Error, ErrorCode, Result, TensorId};
use infer_ir::{DataflowGraph, ModelIr, TensorStorage};
use infer_model_recipes::decoder::lower;
use infer_models::{
    HostTensor, LoadOptions, ModelPackage, TensorDtype, TensorLoadPlan, WeightLoadPlan,
    WeightStorage, WeightTarget, load_weights,
};
use infer_state::blocks::BlockLease;
use infer_state::{kv::KvCacheConfig, kv::KvPages};
use metal::Buffer;
use objc::rc::autoreleasepool;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, collections::VecDeque, sync::Arc, time::Instant};

/// Largest weight-upload chunk streamed into a staging buffer.
const MAX_UPLOAD_CHUNK_BYTES: usize = 4 * MIB;
/// Capacity of the backend resource pool, sized for admission fan-out.
const RESOURCE_POOL_CAPACITY: usize = 260;

pub(super) struct ResidentWeights {
    pub buffers: BTreeMap<TensorId, Buffer>,
    pub formats: BTreeMap<TensorId, TensorDtype>,
    pub plan: WeightLoadPlan,
    pub identity: String,
}
struct Upload<'a>(&'a MetalDevice);
impl WeightTarget for Upload<'_> {
    type Tensor = Buffer;
    fn allocate(&mut self, tensor: &TensorLoadPlan) -> Result<Buffer> {
        self.0.allocate_bytes(tensor.bytes)
    }
    fn upload(&mut self, tensor: &mut Buffer, offset: u64, input: &[u8]) -> Result<()> {
        MetalDevice::write_bytes_idle(tensor, offset, input)
    }
}

pub(super) fn scratch_layout(graph: &DataflowGraph, rows: usize) -> Result<Vec<usize>> {
    let mut sizes = vec![
        0usize;
        graph
            .lifetimes
            .iter()
            .map(|l| l.slot + 1)
            .max()
            .unwrap_or(0)
    ];
    for life in &graph.lifetimes {
        let count = if Some(life.tensor) == graph.logits {
            1
        } else {
            rows
        };
        let elements = life
            .elements
            .checked_mul(count)
            .ok_or_else(|| Error::invalid("prefill scratch shape overflow"))?;
        super::u32_size(elements)?;
        sizes[life.slot] = sizes[life.slot].max(elements);
    }
    Ok(sizes)
}
fn retained_bytes(graph: &DataflowGraph, weights: u64, config: &MetalConfig) -> Result<u64> {
    scratch_layout(graph, config.prefill_chunk_tokens)?
        .into_iter()
        .try_fold(
            weights
                .checked_add(F32_BYTES_U64)
                .ok_or_else(|| Error::invalid("Metal retained size overflow"))?,
            |n, elements| {
                n.checked_add(bytes(elements)?)
                    .ok_or_else(|| Error::invalid("Metal scratch size overflow"))
            },
        )
}
fn preflight(
    gpu: &MetalDevice,
    graph: &DataflowGraph,
    weights: u64,
    config: &MetalConfig,
) -> Result<()> {
    for spec in &graph.tensors {
        super::u32_size(spec.elements()?)?;
    }
    if config.memory_bytes == 0
        || config.block_size == 0
        || config.trace_capacity == 0
        || config.prefill_chunk_tokens == 0
        || config.upload_staging_bytes < F32_BYTES
    {
        return Err(Error::invalid("invalid Metal load/execution configuration"));
    }
    if config.memory_bytes > gpu.device.recommended_max_working_set_size() {
        return Err(Error::new(
            ErrorCode::Capacity,
            "Metal budget exceeds recommended device working set",
        ));
    }
    let retained = retained_bytes(graph, weights, config)?;
    if retained
        .checked_add(
            config
                .prefix_cache_bytes
                .checked_mul(2)
                .ok_or_else(|| Error::invalid("prefix budget overflow"))?,
        )
        .and_then(|n| n.checked_add(config.probe_bytes))
        .is_none_or(|n| n > config.memory_bytes)
    {
        return Err(Error::new(
            ErrorCode::Capacity,
            "Metal weight/prefill scratch/cache budget exhausted",
        ));
    }
    Ok(())
}
fn identity(model: &ModelIr, payload: &str, block_size: usize) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(include_bytes!("../kernels.metal"));
    digest.update(b"fast-math=false;compute=f32;native-weights;layer-major-v1");
    digest.update(serde_json::to_vec(model).map_err(|e| Error::invalid(e.to_string()))?);
    digest.update(payload.as_bytes());
    Ok(format!(
        "metal-paged-dataflow-v3:{}:page{block_size}",
        lower_hex(&digest.finalize())
    ))
}

impl MetalBackend {
    /// Stream source-precision weights directly into resident device buffers.
    /// # Errors
    /// Returns metadata, budget, shader, payload, or upload errors before publishing a backend.
    pub fn from_package(package: &mut ModelPackage, config: MetalConfig) -> Result<Self> {
        package.validate_weight_bindings()?;
        let model = package.imported.model.clone();
        let graph = package.graph.clone();
        let gpu = autoreleasepool(MetalDevice::open)?;
        preflight(&gpu, &graph, package.manifest.weights_bytes, &config)?;
        let options = LoadOptions {
            storage: WeightStorage::Native,
            resident_budget_bytes: config.memory_bytes,
            staging_budget_bytes: config.upload_staging_bytes,
            chunk_bytes: config.upload_staging_bytes.min(MAX_UPLOAD_CHUNK_BYTES),
        };
        let loaded = load_weights(package, &mut Upload(&gpu), options)?;
        let slots: BTreeMap<_, _> = graph
            .tensors
            .iter()
            .filter_map(|s| {
                if let TensorStorage::Weight { slot } = &s.storage {
                    Some((slot.as_str(), s.id))
                } else {
                    None
                }
            })
            .collect();
        let formats = loaded
            .plan
            .tensors
            .iter()
            .map(|t| (slots[t.slot.as_str()], t.dtype))
            .collect();
        let buffers = loaded
            .tensors
            .into_iter()
            .map(|(slot, buffer)| (slots[slot.as_str()], buffer))
            .collect();
        let identity = identity(&model, &loaded.payload_fingerprint, config.block_size)?;
        Self::assemble(
            model,
            graph,
            gpu,
            ResidentWeights {
                buffers,
                formats,
                plan: loaded.plan,
                identity,
            },
            config,
        )
    }

    /// Construct from caller-owned F32 tensors; package loading uses the bounded streaming path.
    /// # Errors
    /// Returns model, budget, payload, or device errors for invalid weights or resources.
    pub fn new(
        model: ModelIr,
        weights: BTreeMap<String, HostTensor>,
        config: MetalConfig,
    ) -> Result<Self> {
        // Metal binds no vision encoder, so no lane carries media: the text graph has no media
        // feature nodes to execute.
        let graph = lower(&ModelIr {
            media_features: 0,
            ..model.clone()
        })?;
        let gpu = autoreleasepool(MetalDevice::open)?;
        let resident = graph
            .tensors
            .iter()
            .filter(|s| matches!(s.storage, TensorStorage::Weight { .. }))
            .try_fold(0u64, |n, s| {
                n.checked_add(bytes(s.elements()?)?)
                    .ok_or_else(|| Error::invalid("weight size overflow"))
            })?;
        preflight(&gpu, &graph, resident, &config)?;
        let weights = upload_host(&gpu, &model, &graph, weights, &config)?;
        Self::assemble(model, graph, gpu, weights, config)
    }

    pub(super) fn assemble(
        model: ModelIr,
        graph: DataflowGraph,
        gpu: MetalDevice,
        weights: ResidentWeights,
        config: MetalConfig,
    ) -> Result<Self> {
        preflight(&gpu, &graph, weights.plan.resident_bytes, &config)?;
        let sizes = scratch_layout(&graph, config.prefill_chunk_tokens)?;
        let scratch_bytes = sizes.iter().try_fold(0u64, |n, size| {
            n.checked_add(bytes(*size)?)
                .ok_or_else(|| Error::invalid("scratch overflow"))
        })?;
        let scratch = sizes
            .into_iter()
            .map(|n| gpu.zeros(n))
            .collect::<Result<_>>()?;
        let kv = allocate_kv(
            &gpu,
            &graph,
            &config,
            retained_bytes(&graph, weights.plan.resident_bytes, &config)?,
        )?;
        let specs = graph.tensors.iter().map(|t| (t.id, t.clone())).collect();
        let slots = graph.lifetimes.iter().map(|l| (l.tensor, l.slot)).collect();
        let mut last = BTreeMap::new();
        for node in &graph.nodes {
            if let Some(layer) = node.layer {
                last.insert(layer, node.id);
            }
        }
        let kv_manager = KvPages::new(KvCacheConfig {
            namespace: weights.identity.as_bytes().to_vec(),
            block_size: config.block_size,
            blocks: kv.blocks,
            bytes_per_block: kv.block_bytes,
            prefix_bytes: config.prefix_cache_bytes,
            max_prefixes: MAX_PREFIX_ENTRIES,
        })?;
        let dummy = gpu.zeros(1)?;
        let state_recipe = infer_ir::StateRecipe::compile(
            &model,
            &graph,
            config.block_size,
            F32_BYTES,
            config.probe_bytes > 0,
            size_of::<BlockLease>(),
        )?;
        let mut backend = Self {
            owner: Arc::new(()),
            resources: infer_spi::ResourcePool::new(RESOURCE_POOL_CAPACITY)?,
            gpu,
            model,
            graph,
            specs,
            weights: weights.buffers,
            weight_formats: weights.formats,
            load_plan: weights.plan,
            identity: weights.identity,
            slots,
            scratch,
            scratch_bytes,
            dummy,
            config,
            sequences: BTreeMap::new(),
            busy: None,
            inflight_states: vec![],
            inflight_pins: vec![],
            ticket_work: Vec::with_capacity(MAX_TICKET_TASKS),
            completion_pool: (0..COMPLETION_POOL_SIZE)
                .map(|_| Vec::with_capacity(MAX_TICKET_TASKS))
                .collect(),
            tokens_executed: 0,
            prefix_hits: 0,
            kv: kv_manager,
            kv_buffers: kv.buffers,
            kv_block_bytes: kv.block_bytes,
            state_recipe,
            traces: VecDeque::new(),
            trace_dropped: 0,
            probes: VecDeque::new(),
            probe_dropped: 0,
            commands: VecDeque::new(),
            transfers: vec![],
            layer_outputs: last.into_iter().map(|(l, o)| (o, l)).collect(),
            origin: Instant::now(),
            forward_nodes: vec![],
        };
        backend.prepare_forward()?;
        Ok(backend)
    }
    #[must_use]
    pub const fn load_plan(&self) -> &WeightLoadPlan {
        &self.load_plan
    }
}

fn upload_host(
    gpu: &MetalDevice,
    model: &ModelIr,
    graph: &DataflowGraph,
    mut weights: BTreeMap<String, HostTensor>,
    config: &MetalConfig,
) -> Result<ResidentWeights> {
    let mut digest = Sha256::new();
    let mut buffers = BTreeMap::new();
    let mut formats = BTreeMap::new();
    let mut tensors = Vec::new();
    let mut resident_bytes = 0u64;
    let chunk_elements = config.upload_staging_bytes / F32_BYTES;
    let mut staging_bytes = 0;
    for spec in &graph.tensors {
        let TensorStorage::Weight { slot } = &spec.storage else {
            continue;
        };
        let tensor = weights
            .remove(slot)
            .ok_or_else(|| Error::invalid(format!("unbound Metal weight {slot}")))?;
        tensor.validate()?;
        if tensor.shape != spec.shape {
            return Err(Error::invalid("Metal weight shape mismatch"));
        }
        let plan = TensorLoadPlan {
            slot: slot.clone(),
            shape: spec.shape.clone(),
            dtype: TensorDtype::F32,
            bytes: bytes(tensor.data.len())?,
        };
        digest.update(serde_json::to_vec(&plan).map_err(|e| Error::invalid(e.to_string()))?);
        let buffer = gpu.allocate_bytes(plan.bytes)?;
        for (at, chunk) in tensor.data.chunks(chunk_elements).enumerate() {
            let input: Vec<_> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
            digest.update(&input);
            MetalDevice::write_bytes_idle(
                &buffer,
                (at * chunk_elements * F32_BYTES) as u64,
                &input,
            )?;
            staging_bytes = staging_bytes.max(input.len());
        }
        resident_bytes += plan.bytes;
        buffers.insert(spec.id, buffer);
        formats.insert(spec.id, plan.dtype);
        tensors.push(plan);
    }
    if !weights.is_empty() {
        return Err(Error::invalid("unconsumed Metal weights"));
    }
    let identity = identity(model, &lower_hex(&digest.finalize()), config.block_size)?;
    let plan = WeightLoadPlan {
        tensors,
        source_bytes: resident_bytes,
        resident_bytes,
        staging_bytes,
        chunk_bytes: chunk_elements * F32_BYTES,
    };
    Ok(ResidentWeights {
        buffers,
        formats,
        plan,
        identity,
    })
}
