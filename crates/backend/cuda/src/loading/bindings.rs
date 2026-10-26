use super::{Draft, LoadOptions, Projection, floats, phase};
use crate::constants::FUSED_VERIFY_LANES;
use crate::constants::GIB;
use crate::constants::MAX_PREFILL_LANES;
use crate::device::DeviceProfile;
use crate::mlp::ProjectionWeight;
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::tuning::select_tiling;
use crate::tuning::{TilingSource, TuningReport, resolve_table, shipped_table, table_path};
use crate::{
    device::CudaDevice,
    resident::{FusionWeights, ProgramWeights, PromptScratchRegistry},
};
use cutile::half::bf16;
use infer_core::{Error, Result, TensorId};
use infer_ir::{DataflowGraph, TensorOp, TensorStorage};
use infer_models::mrope::axes as mrope_axes;
use infer_models::{QuantizedPackage, TensorDtype};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

/// Read budget for one BF16 embedding tensor.
const MAX_EMBEDDING_READ_BYTES: u64 = 4 * GIB as u64;

pub(super) fn load(
    device: &CudaDevice,
    profile: &DeviceProfile,
    package: &mut QuantizedPackage,
    options: &LoadOptions,
) -> Result<(ProgramWeights, Option<Draft>, TuningReport)> {
    let path = table_path();
    let mut tuning = TuningReport::default();
    let table = match resolve_table(path.as_deref()) {
        Ok(table) => table,
        Err(error) => {
            tuning.fallback.push(format!("tuning cache: {error}"));
            tracing::warn!(target: "infer::tuning", %error, "invalid tuning cache; using conservative fallback");
            None
        }
    };
    let shipped = shipped_table()?;
    let tuning_source = TilingSource {
        local: table.as_ref(),
        shipped: Some(shipped),
        path,
        autotune: options.autotune,
    };
    let mut weights = empty_weights(options);
    let weights_phase = phase::Phase::start(
        "weights",
        format!("binding {} checkpoint weights", package.weights.len()),
    );
    bind_tensors(
        device,
        profile,
        package,
        &tuning_source,
        &mut tuning,
        &mut weights,
    )
    .map_err(|e| weights_phase.fail(&e))?;
    weights_phase.finish();
    add_rope(
        device,
        &package.graph,
        &mut weights,
        &package.imported.model.position,
    )?;
    if options.fp8_kv == Some(true) {
        let kv_phase = phase::Phase::start("kv_scales", "uploading KV scales");
        weights.kv_scales = super::kv_scales::load(package).map_err(|e| kv_phase.fail(&e))?;
        kv_phase.finish();
    }
    let draft = if options.mtp_depth > 0 {
        let draft_phase = phase::Phase::start("draft", "binding the MTP draft head");
        let draft = draft(
            device,
            profile,
            package,
            &weights,
            &tuning_source,
            &mut tuning,
        )
        .map_err(|e| draft_phase.fail(&e))?;
        draft_phase.finish();
        Some(draft)
    } else {
        None
    };
    Ok((weights, draft, tuning))
}

/// Weights with their graph-independent fields set; graph tensors are bound by [`bind_tensors`].
fn empty_weights(options: &LoadOptions) -> ProgramWeights {
    ProgramWeights {
        autotune: options.autotune,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        pool_gemv: Arc::new(Mutex::new(BTreeMap::new())),
        batch_width: options.verification_width.max(
            if options.prefill_width == FUSED_VERIFY_LANES {
                FUSED_VERIFY_LANES
            } else {
                0
            },
        ),
        prefill_width: options.prefill_width,
        narrow_prefill_width: 0,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion: None,
        attention_profile: options.attention_profile,
        fp8_inputs: BTreeSet::new(),
        chunked_recurrent: options.chunked_recurrent,
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
    }
}

/// Bind every graph weight: projections choose a tile, embeddings and constants get uploaded.
fn bind_tensors(
    device: &CudaDevice,
    profile: &DeviceProfile,
    package: &mut QuantizedPackage,
    tuning_source: &TilingSource<'_>,
    tuning: &mut TuningReport,
    weights: &mut ProgramWeights,
) -> Result<()> {
    let (banks, bank_ids) = super::experts::load(device, package)?;
    weights.expert_banks = banks;
    let projections: BTreeSet<_> = package
        .graph
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Linear)
        .map(|n| n.inputs[1])
        .collect();
    let embeddings: BTreeSet<_> = package
        .graph
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Embedding)
        .map(|n| n.inputs[0])
        .collect();
    let total = package
        .graph
        .tensors
        .iter()
        .filter(|tensor| matches!(tensor.storage, TensorStorage::Weight { .. }))
        .count();
    let mut done = 0;
    for tensor in package.graph.tensors.clone() {
        let TensorStorage::Weight { slot } = tensor.storage else {
            continue;
        };
        if bank_ids.contains(&tensor.id) {
            continue;
        }
        let source = package
            .weights
            .get(&slot)
            .ok_or_else(|| Error::invalid("missing graph weight"))?
            .clone();
        if projections.contains(&tensor.id) {
            let projection = Projection::load(device, package, &source)?.resident();
            bind_activation(package, &source, weights, tensor.id)?;
            let tiling = select_tiling(device, &projection, profile, tuning_source, tuning)?;
            weights.tiling.insert(tensor.id, tiling);
            // A tied output projection shares its storage with the input embedding, so the very
            // same weight must also be reachable by the embedding lookup.
            if embeddings.contains(&tensor.id) {
                let ProjectionWeight::Dense(shared) = &projection else {
                    return Err(Error::unsupported(
                        "a tied input embedding requires a dense output projection",
                    ));
                };
                weights.embeddings.insert(tensor.id, Arc::clone(shared));
            }
            weights.projections.insert(tensor.id, projection);
        } else if embeddings.contains(&tensor.id) {
            if source.data.dtype != TensorDtype::BF16 {
                return Err(Error::unsupported("CUDA embedding requires BF16"));
            }
            let bytes = package.read(&source.data, MAX_EMBEDDING_READ_BYTES)?;
            let values = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
                .collect();
            drop(bytes);
            weights
                .embeddings
                .insert(tensor.id, device.upload(values, &source.shape)?);
        } else {
            let values = floats(package, &source.data)?;
            let length = values.len();
            weights
                .constants
                .insert(tensor.id, device.upload(values, &[length])?);
        }
        done += 1;
        tracing::debug!(
            target: "infer::load",
            done,
            total,
            bytes = source.data.bytes,
            item = source.data.name.as_str(),
            "weight bound"
        );
    }
    Ok(())
}

/// Lower the single attention block of the MTP head into a standalone draft graph.
fn draft(
    device: &CudaDevice,
    profile: &DeviceProfile,
    package: &mut QuantizedPackage,
    target: &ProgramWeights,
    source: &TilingSource<'_>,
    report: &mut TuningReport,
) -> Result<Draft> {
    // The draft head's prefix, depth and fusion slots come from the model provider.
    if matches!(
        package.imported.model.feed_forward,
        infer_ir::FeedForward::Moe { .. }
    ) {
        return Err(Error::unsupported(
            "MoE MTP draft binding is not yet implemented",
        ));
    }
    let plan = package
        .imported
        .speculation
        .clone()
        .ok_or_else(|| Error::unsupported("MTP draft head is not declared"))?;
    if plan.layers != 1 {
        return Err(Error::unsupported("MTP requires exactly one MTP layer"));
    }
    let (block, graph) = package.provider.draft_graph(&package.imported.model)?;
    let mut weights = draft_weights(device, profile, package, target, &graph, source, report)?;
    add_rope(device, &graph, &mut weights, &block.position)?;
    Ok(Draft { graph, weights })
}

/// Load one draft projection. BF16 draft projections are re-encoded as FP8 channels with
/// per-token activations, which halves the bytes each proposal step streams; the target still
/// verifies every token.
fn draft_projection(
    device: &CudaDevice,
    package: &mut QuantizedPackage,
    source: &infer_models::DeviceWeight,
    weights: &mut ProgramWeights,
    id: TensorId,
) -> Result<ProjectionWeight> {
    let projection = if source.encoding == infer_models::WeightEncoding::Float {
        weights.fp8_inputs.insert(id);
        let values = floats(package, &source.data)?;
        Projection::quantize_fp8(device, &values, &source.shape)?.resident()
    } else {
        Projection::load(device, package, source)?.resident()
    };
    bind_activation(package, source, weights, id)?;
    Ok(projection)
}

/// Draft KV in FP8 with unit scales, as vLLM caches the uncalibrated MTP layer: decode attention
/// then reads a quarter of the F32 bytes. The draft only proposes, so greedy target outputs keep
/// the target's numerics.
fn draft_kv_scales(graph: &DataflowGraph) -> BTreeMap<TensorId, [f32; 2]> {
    graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, TensorOp::Attention { .. }))
        .filter_map(|node| node.states.first().map(|&state| (state, [1.0, 1.0])))
        .collect()
}

fn draft_linear_inputs(graph: &DataflowGraph) -> BTreeSet<TensorId> {
    graph
        .nodes
        .iter()
        .filter(|node| node.op == TensorOp::Linear)
        .map(|node| node.inputs[1])
        .collect()
}

/// Bind draft weights, sharing the target embedding and vocabulary projection by slot.
fn draft_weights(
    device: &CudaDevice,
    profile: &DeviceProfile,
    package: &mut QuantizedPackage,
    target: &ProgramWeights,
    graph: &DataflowGraph,
    tiling_source: &TilingSource<'_>,
    report: &mut TuningReport,
) -> Result<ProgramWeights> {
    let plan = package
        .imported
        .speculation
        .clone()
        .ok_or_else(|| Error::unsupported("MTP draft head is not declared"))?;
    let linear_inputs = draft_linear_inputs(graph);
    let mut weights = ProgramWeights {
        autotune: target.autotune,
        resident_tuning: Arc::clone(&target.resident_tuning),
        pool_gemv: Arc::new(Mutex::new(BTreeMap::new())),
        batch_width: 0,
        prefill_width: MAX_PREFILL_LANES,
        narrow_prefill_width: 0,
        kv_scales: draft_kv_scales(graph),
        tiling: BTreeMap::new(),
        fusion: Some(draft_fusion(
            device,
            profile,
            package,
            tiling_source,
            &plan,
            report,
        )?),
        // The proposal program keeps its existing BF16 behavior; the target profile only
        // changes target prefill eligibility.
        attention_profile: target.attention_profile,
        fp8_inputs: BTreeSet::new(),
        // The draft proposes for the same sequence, so it follows the target's recurrence policy.
        chunked_recurrent: target.chunked_recurrent,
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
    };
    for tensor in &graph.tensors {
        let TensorStorage::Weight { slot } = &tensor.storage else {
            continue;
        };
        if slot == "embed_tokens.weight" {
            let id = weight_id(&package.graph, slot)?;
            let embedding = target
                .embeddings
                .get(&id)
                .cloned()
                .ok_or_else(|| Error::invalid("missing MTP target embedding"))?;
            weights.embeddings.insert(tensor.id, embedding);
            continue;
        }
        if slot == "lm_head.weight" {
            let id = weight_id(&package.graph, slot)?;
            let projection = target
                .projections
                .get(&id)
                .cloned()
                .ok_or_else(|| Error::invalid("missing MTP target projection"))?;
            let tiling = target
                .tiling
                .get(&id)
                .copied()
                .ok_or_else(|| Error::invalid("missing MTP target tiling"))?;
            weights.projections.insert(tensor.id, projection);
            if target.fp8_inputs.contains(&id) {
                weights.fp8_inputs.insert(tensor.id);
            }
            if let Some(scale) = target.input_scales.get(&id) {
                weights.input_scales.insert(tensor.id, *scale);
            }
            weights.tiling.insert(tensor.id, tiling);
            continue;
        }
        let source = package
            .mtp
            .get(&format!("{}{slot}", plan.prefix))
            .cloned()
            .ok_or_else(|| Error::invalid("missing MTP binding"))?;
        if source.shape != tensor.shape {
            return Err(Error::invalid("MTP weight shapes"));
        }
        if linear_inputs.contains(&tensor.id) {
            let projection = draft_projection(device, package, &source, &mut weights, tensor.id)?;
            weights.tiling.insert(
                tensor.id,
                select_tiling(device, &projection, profile, tiling_source, report)?,
            );
            weights.projections.insert(tensor.id, projection);
        } else {
            let values = floats(package, &source.data)?;
            let length = values.len();
            weights
                .constants
                .insert(tensor.id, device.upload(values, &[length])?);
        }
    }
    Ok(weights)
}

/// Load the MTP fusion projection and its two input normalization vectors.
fn draft_fusion(
    device: &CudaDevice,
    profile: &DeviceProfile,
    package: &mut QuantizedPackage,
    tiling_source: &TilingSource<'_>,
    plan: &infer_spi::SpeculationPlan,
    report: &mut TuningReport,
) -> Result<FusionWeights> {
    let fusion = plan
        .fusion
        .as_ref()
        .ok_or_else(|| Error::unsupported("MTP fusion projection is not declared"))?;
    let fc = package
        .mtp
        .get(&format!("{}{}", plan.prefix, fusion.projection))
        .cloned()
        .ok_or_else(|| Error::invalid("missing MTP fusion weight"))?;
    let embedding_norm = package
        .mtp
        .get(&format!("{}{}", plan.prefix, fusion.norms[0]))
        .cloned()
        .ok_or_else(|| Error::invalid("missing MTP fusion weight"))?;
    let hidden_norm = package
        .mtp
        .get(&format!("{}{}", plan.prefix, fusion.norms[1]))
        .cloned()
        .ok_or_else(|| Error::invalid("missing MTP fusion weight"))?;
    let hidden = package.imported.model.hidden_size;
    if fc.shape != [hidden, hidden * 2]
        || embedding_norm.shape != [hidden]
        || hidden_norm.shape != [hidden]
    {
        return Err(Error::invalid("MTP fusion weight shapes"));
    }
    let fc = Projection::load(device, package, &fc)?.resident();
    // The fusion projection is tuned exactly like the target's own projections.
    let tiling = select_tiling(device, &fc, profile, tiling_source, report)?;
    let mut norms = floats(package, &embedding_norm.data)?;
    norms.extend_from_slice(&floats(package, &hidden_norm.data)?);
    Ok(FusionWeights {
        projection: fc,
        tiling,
        norms: device.upload(norms, &[2, hidden])?,
        epsilon: package.imported.model.norm_epsilon,
        offset: package.imported.model.norm_weight_offset,
    })
}

fn weight_id(graph: &DataflowGraph, slot: &str) -> Result<TensorId> {
    graph
        .tensors
        .iter()
        .find_map(|spec| match &spec.storage {
            TensorStorage::Weight { slot: name } if name == slot => Some(spec.id),
            _ => None,
        })
        .ok_or_else(|| Error::invalid("missing MTP target weight"))
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Bounded rotary dimensions are intentionally rounded to the F32 device frequency contract"
)]
fn add_rope(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &mut ProgramWeights,
    position: &infer_ir::PositionSpec,
) -> Result<()> {
    weights.mrope = !position.multimodal_sections.is_empty();
    for node in &graph.nodes {
        if let TensorOp::Rope {
            rotary_dim, theta, ..
        } = node.op
        {
            let half = rotary_dim / 2;
            let frequencies = (0..half)
                .map(|i| theta.powf(-((2 * i) as f64) / rotary_dim as f64) as f32)
                .collect();
            let axes = mrope_axes(position, half)?;
            weights
                .rope_axes
                .insert(node.id, device.upload(axes, &[half])?);
            weights
                .rope_frequencies
                .insert(node.id, device.upload(frequencies, &[half])?);
        }
    }
    Ok(())
}

/// Bind checkpoint activation encoding separately from weight storage and runtime buffers.
fn bind_activation(
    package: &mut QuantizedPackage,
    source: &infer_models::DeviceWeight,
    weights: &mut ProgramWeights,
    id: TensorId,
) -> Result<()> {
    if source.fp8_token_input {
        weights.fp8_inputs.insert(id);
    }
    if source.encoding == infer_models::WeightEncoding::Nvfp4
        && let Some(scale) = source.input_global_scale.as_ref()
    {
        let values = floats(package, scale)?;
        let [value] = values.as_slice() else {
            return Err(Error::invalid("input global scale must be scalar"));
        };
        if !value.is_finite() || *value <= 0.0 {
            return Err(Error::invalid("invalid activation scale"));
        }
        weights.input_scales.insert(id, *value);
    }
    Ok(())
}
