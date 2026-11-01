use super::*;
use crate::constants::KV_BLOCK_TOKENS;
use crate::resident::PromptScratchRegistry;
use crate::resident::attention_mma::catalog::DIM;
use crate::resident::attention_mma::catalog::HEADS;
use crate::resident::attention_mma::catalog::KV_HEADS;
use crate::resident::attention_mma::catalog::workspace_bytes;
use crate::resident::attention_mma::eligible;
use crate::resident::attention_mma::eligible_with_role;
use crate::resident::attention_mma::node_eligible;
use crate::test_support::fixture_graph;
use crate::tuning::resident_gemm::ResidentTuningSession;
use cutile::half::bf16;
use infer_core::{OpId, TensorId};
use infer_cuda_kernels::tile::kernels as tile_kernels;
use infer_ir::{DType, TensorNode, TensorSpec};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

fn fixture_weights() -> ProgramWeights {
    ProgramWeights {
        kernels: Arc::new(tile_kernels(0)),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        chunked_recurrent: false,
        batch_width: 3,
        prefill_width: 32,
        narrow_prefill_width: 0,
        fusion: None,
        attention_profile: crate::AttentionProfile::Canonical,
        projections: BTreeMap::new(),
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    }
}

#[test]
fn admission_charges_shared_projection_geometry_once() -> Result<()> {
    let mut weights = fixture_weights();
    let mut graph = DataflowGraph::default();
    for (id, size) in [(1, 128), (2, 256), (3, 128), (4, 256)] {
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape: vec![size],
            dtype: DType::F32,
            storage: TensorStorage::Activation,
        });
    }
    for (id, input, output, weight) in [(1, 1, 2, 10), (2, 3, 4, 11)] {
        graph.nodes.push(TensorNode {
            id: OpId::new(id)?,
            layer: None,
            op: TensorOp::Linear,
            inputs: vec![TensorId::new(input)?, TensorId::new(weight)?],
            outputs: vec![TensorId::new(output)?],
            states: vec![],
        });
    }
    // Repeated layers share one [3, K] / [3, N] pair. Prefill has zero-copy views.
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, true))?,
        (128 + 256) * 3 * F32
    );
    weights.input_scales.insert(TensorId::new(10)?, 1.0);
    weights.input_scales.insert(TensorId::new(11)?, 1.0);
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, true))?,
        (128 + 256) * 3 * F32 + (128 / 2 + 128 / 16) * (1 + 3 + 32)
    );
    // Wider verification batches also share one pack/unpack pair per geometry.
    weights.batch_width = 5;
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, true))?,
        (128 + 256) * 5 * F32 + (128 / 2 + 128 / 16) * (1 + 5 + 32)
    );
    // Without a private verification graph its pack/unpack pairs and staging rows vanish.
    assert_eq!(
        projection_workspace(&graph, &weights, ProgramGeometry::new(&weights, false))?,
        (128 / 2 + 128 / 16) * (1 + 32)
    );
    Ok(())
}

#[test]
fn kv_block_bytes_counts_attention_states_in_storage_bytes() -> Result<()> {
    let mut weights = fixture_weights();
    let mut graph = DataflowGraph::default();
    let kv = |id| {
        Ok(TensorSpec {
            id: TensorId::new(id)?,
            // [capacity, kv_heads * head_dim], one spec covering keys and values.
            shape: vec![64, 512],
            dtype: DType::F32,
            storage: TensorStorage::State {
                layer: usize::try_from(id).expect("fixture layer index fits usize"),
                kind: StateKind::AttentionKv,
            },
        })
    };
    graph.tensors.push(kv(1)?);
    graph.tensors.push(kv(2)?);
    // A conv state and an activation are not KV and must not be priced per block.
    graph.tensors.push(TensorSpec {
        id: TensorId::new(3)?,
        shape: vec![128, 4],
        dtype: DType::F32,
        storage: TensorStorage::State {
            layer: 3,
            kind: StateKind::Conv,
        },
    });
    graph.tensors.push(TensorSpec {
        id: TensorId::new(4)?,
        shape: vec![512],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    });
    let block = KV_BLOCK_TOKENS as u64;
    // Unscaled: 2 layers * 32 tokens * 512 columns * (K+V) * f32.
    assert_eq!(kv_block_bytes(&graph, &weights)?, 2 * block * 512 * 2 * F32);
    // An fp8-packed cache stores one byte per element.
    weights.kv_scales.insert(TensorId::new(1)?, [1.0, 1.0]);
    assert_eq!(
        kv_block_bytes(&graph, &weights)?,
        block * 512 * 2 * (F32 + 1)
    );
    Ok(())
}

#[test]
fn pooled_admission_excludes_uncaptured_wide_prompt_and_private_readback() -> Result<()> {
    let mut weights = fixture_weights();
    weights.prefill_width = 256;
    weights.narrow_prefill_width = 64;
    let graph = fixture_graph(48)?;
    let quote = |private| {
        program_price(
            &graph,
            &weights,
            48,
            48,
            OutputReadout::Logits,
            ProgramGeometry::new(&weights, private),
            &fixture_profile(),
        )?
        .at(256)
    };
    // Two live F32 activation rows per graph lane; external rows follow the captured
    // narrow geometry, and prompt readback stages only its final row. Prompt arenas are shared by
    // every sequence and priced once (`prompt_scratch_bytes`), never per sequence.
    let pooled = (2 * 48) * F32 + (1 + 64) * 48 * F32 + 2 * (48 + 48) * F32;
    assert_eq!(quote(false)?, pooled);
    // Private capture adds the wide external rows, verification arena and readback rows.
    assert_eq!(
        quote(true)? - quote(false)?,
        (256 - 64) * 48 * F32 + 3 * (2 * 48) * F32 + 3 * (48 + 48) * F32
    );
    // The shared scratch holds the widest prompt's arena once; the 128-row rung and the narrow
    // graph capture against views of it.
    assert_eq!(
        ProgramGeometry::new(&weights, true).prompt_widths(),
        vec![256, 128, 64]
    );
    assert_eq!(
        prompt_scratch_bytes(
            &graph,
            &weights,
            ProgramGeometry::new(&weights, true),
            &fixture_profile()
        )?,
        256 * (2 * 48) * F32
    );
    Ok(())
}

#[test]
fn pooled_admission_keeps_verification_required_by_prompt_fallback() -> Result<()> {
    let mut weights = fixture_weights();
    weights.prefill_width = 3;
    let graph = fixture_graph(48)?;
    let quote = |private| {
        program_price(
            &graph,
            &weights,
            48,
            48,
            OutputReadout::Logits,
            ProgramGeometry::new(&weights, private),
            &fixture_profile(),
        )?
        .at(256)
    };
    // Without a prompt graph, prefill uses the private verification graph even when the
    // caller requests pooled decode; its activations and readback must remain charged.
    let expected = (1 + 3) * (2 * 48) * F32 + (1 + 3) * 48 * F32 + (1 + 3) * (48 + 48) * F32;
    assert_eq!(quote(false)?, expected);
    assert_eq!(quote(true)?, expected);
    Ok(())
}

#[path = "loading_budget_recurrent.rs"]
mod recurrent;

#[test]
fn affine_admission_rejects_both_product_and_total_overflow() {
    let price = SequencePrice {
        fixed: 1,
        per_token: u64::MAX / 2,
    };
    assert_eq!(price.at(2).unwrap(), u64::MAX);
    assert!(price.at(3).is_err());
    assert!(SequencePrice { fixed: 2, ..price }.at(2).is_err());
    assert!(
        price
            .combine(SequencePrice {
                fixed: 0,
                per_token: u64::MAX
            })
            .is_err()
    );
}

#[test]
fn affine_admission_counts_fp8_and_f32_kv_and_full_history() -> Result<()> {
    let mut weights = fixture_weights();
    let graph = fixture_graph(48)?;
    let mut with_kv = graph.clone();
    for (id, columns) in [(4, 128), (5, 256)] {
        with_kv.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape: vec![64, columns],
            dtype: DType::F32,
            storage: TensorStorage::State {
                layer: usize::try_from(id).expect("small fixture layer"),
                kind: StateKind::AttentionKv,
            },
        });
    }
    weights.kv_scales.insert(TensorId::new(4)?, [1.0, 1.0]);
    for private in [true, false] {
        let base = program_price(
            &graph,
            &weights,
            48,
            48,
            OutputReadout::Logits,
            ProgramGeometry::new(&weights, private),
            &fixture_profile(),
        )?
        .at(1)?;
        for readout in [
            OutputReadout::None,
            OutputReadout::Logits,
            OutputReadout::Full,
        ] {
            let price = program_price(
                &with_kv,
                &weights,
                48,
                48,
                readout,
                ProgramGeometry::new(&weights, private),
                &fixture_profile(),
            )?;
            // Key and value storage: one FP8 and one F32 state. Full history adds
            // a device staging copy and a host copy for each 48-element row.
            let growth = 2 * (128 + 256 * F32)
                + if readout == OutputReadout::Full {
                    48 * READBACK_COPIES * F32
                } else {
                    0
                };
            for power in 0..=15 {
                let capacity = 1_u64 << power;
                assert_eq!(price.at(capacity)?, base + growth * capacity);
            }
        }
    }
    Ok(())
}

fn fixture_profile() -> DeviceProfile {
    DeviceProfile {
        name: "CPU budget fixture".into(),
        architecture: "sm_120".into(),
        multiprocessors: 0,
        total_memory_bytes: 0,
        memory_clock_khz: 0,
        memory_bus_width_bits: 0,
        l2_cache_bytes: 0,
    }
}

/// A valid attention graph for admission arithmetic: three embedding producers feed one
/// FA4-shaped attention node whose KV state is either FP8-packed or F32 by scale presence.
fn attention_graph(lanes: usize) -> Result<(DataflowGraph, TensorId)> {
    let state = TensorId::new(1)?;
    let embedding = TensorId::new(2)?;
    let query = TensorId::new(3)?;
    let key = TensorId::new(4)?;
    let value = TensorId::new(5)?;
    let output = TensorId::new(6)?;
    let width = HEADS * DIM;
    let kv_width = KV_HEADS * DIM;
    let activation = |id, elements| TensorSpec {
        id,
        shape: vec![elements],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    };
    let graph = DataflowGraph {
        tensors: vec![
            TensorSpec {
                id: state,
                shape: vec![64, kv_width],
                dtype: DType::F32,
                storage: TensorStorage::State {
                    layer: 0,
                    kind: StateKind::AttentionKv,
                },
            },
            TensorSpec {
                id: embedding,
                shape: vec![1, 1],
                dtype: DType::Bf16,
                storage: TensorStorage::Weight {
                    slot: "embedding".into(),
                },
            },
            activation(query, lanes * width),
            activation(key, lanes * kv_width),
            activation(value, lanes * kv_width),
            activation(output, lanes * width),
        ],
        nodes: vec![
            TensorNode {
                id: OpId::new(1)?,
                layer: None,
                op: TensorOp::Embedding,
                inputs: vec![embedding],
                outputs: vec![query],
                states: vec![],
            },
            TensorNode {
                id: OpId::new(2)?,
                layer: None,
                op: TensorOp::Embedding,
                inputs: vec![embedding],
                outputs: vec![key],
                states: vec![],
            },
            TensorNode {
                id: OpId::new(3)?,
                layer: None,
                op: TensorOp::Embedding,
                inputs: vec![embedding],
                outputs: vec![value],
                states: vec![],
            },
            TensorNode {
                id: OpId::new(4)?,
                layer: None,
                op: TensorOp::Attention {
                    query_heads: HEADS,
                    kv_heads: KV_HEADS,
                    head_dim: DIM,
                    window: None,
                },
                inputs: vec![query, key, value],
                outputs: vec![output],
                states: vec![state],
            },
        ],
        hidden: Some(output),
        logits: Some(output),
        features: None,
        ..DataflowGraph::default()
    };
    Ok((graph, state))
}

#[test]
fn target_attention_profile_eligibility_matrix_is_explicit() -> Result<()> {
    let (graph, state) = attention_graph(128)?;
    let compiled = cfg!(all(target_os = "linux", target_arch = "x86_64"));
    for architecture in ["sm_90", "sm_100", "sm_103", "sm_120"] {
        for fp8 in [false, true] {
            let mut weights = fixture_weights();
            weights.prefill_width = 128;
            if fp8 {
                weights.kv_scales.insert(state, [1.0, 1.0]);
            }
            for profile in [
                crate::AttentionProfile::Canonical,
                crate::AttentionProfile::Bf16,
                crate::AttentionProfile::Fp16,
            ] {
                weights.attention_profile = profile;
                let target = compiled
                    && fp8
                    && match profile {
                        crate::AttentionProfile::Bf16 => true,
                        crate::AttentionProfile::Fp16 => architecture == "sm_120",
                        crate::AttentionProfile::Canonical => false,
                    };
                assert_eq!(eligible(architecture, &graph, &weights, 128,), target);
                for draft in [false, true] {
                    assert_eq!(
                        eligible_with_role(architecture, &graph, &weights, draft, 128,),
                        draft || target
                    );
                }
            }
        }
    }
    // Unsupported geometry falls back for the draft too, not only for the target.
    let mut weights = fixture_weights();
    weights.prefill_width = 128;
    weights.kv_scales.insert(state, [1.0, 1.0]);
    weights.attention_profile = crate::AttentionProfile::Bf16;
    let mut unsupported = graph;
    if let TensorOp::Attention { query_heads, .. } = &mut unsupported.nodes[3].op {
        *query_heads = 8;
    }
    assert!(!eligible("sm_120", &unsupported, &weights, 128,));
    assert!(!eligible_with_role(
        "sm_120",
        &unsupported,
        &weights,
        true,
        128,
    ));
    Ok(())
}

#[test]
fn mixed_fp8_and_f32_graph_is_decided_per_attention_node() -> Result<()> {
    let (mut graph, fp8_state) = attention_graph(128)?;
    // Keep the FP8 node (index 3) and add a sibling F32 target node. The graph-level helper
    // must remain true because of the FP8 node, while the F32 node itself must never become
    // eligible for the target BF16 profile.
    let float_state = TensorId::new(7)?;
    let mut f32_node = graph.nodes[3].clone();
    f32_node.id = OpId::new(5)?;
    f32_node.states = vec![float_state];
    f32_node.outputs = vec![TensorId::new(8)?];
    graph.nodes.push(f32_node);

    let mut weights = fixture_weights();
    weights.prefill_width = 128;
    weights.attention_profile = crate::AttentionProfile::Bf16;
    weights.kv_scales.insert(fp8_state, [1.0, 1.0]);
    assert!(eligible("sm_120", &graph, &weights, 128));
    assert!(node_eligible(
        "sm_120",
        &graph.nodes[3],
        &weights,
        false,
        128
    ));
    assert!(!node_eligible(
        "sm_120",
        &graph.nodes[4],
        &weights,
        false,
        128
    ));
    // The draft proposal keeps its existing F32-KV behavior in the same graph.
    for node in &graph.nodes[3..] {
        assert!(node_eligible("sm_120", node, &weights, true, 128));
    }
    // Canonical never routes either node to the BF16 target path, even with FP8 KV.
    weights.attention_profile = crate::AttentionProfile::Canonical;
    assert!(!eligible("sm_120", &graph, &weights, 128));
    for node in &graph.nodes[3..] {
        assert!(!node_eligible("sm_120", node, &weights, false, 128));
    }
    Ok(())
}

#[test]
fn target_bf16_profile_admission_quotes_the_same_eligibility_decision() -> Result<()> {
    let (graph, state) = attention_graph(128)?;
    let quote = |fp8: bool, profile| -> Result<u64> {
        let mut weights = fixture_weights();
        weights.prefill_width = 128;
        weights.attention_profile = profile;
        if fp8 {
            weights.kv_scales.insert(state, [1.0, 1.0]);
        }
        program_price(
            &graph,
            &weights,
            48,
            48,
            OutputReadout::Logits,
            ProgramGeometry::new(&weights, false),
            &fixture_profile(),
        )?
        .at(256)
    };
    let canonical_fp8 = quote(true, crate::AttentionProfile::Canonical)?;
    let bf16_fp8 = quote(true, crate::AttentionProfile::Bf16)?;
    let fp16_fp8 = quote(true, crate::AttentionProfile::Fp16)?;
    let workspace = workspace_bytes(0, 128)? as u64;
    let token = (KV_HEADS * DIM * 2 * size_of::<bf16>()) as u64;
    assert_eq!(bf16_fp8 - canonical_fp8, workspace + token * 256);
    // FP16 uses the same 16-bit staging size and the same target-only admission rule.
    assert_eq!(fp16_fp8, bf16_fp8);
    // The explicit profiles do not move F32-KV target admission.
    for profile in [crate::AttentionProfile::Bf16, crate::AttentionProfile::Fp16] {
        assert_eq!(
            quote(false, profile)?,
            quote(false, crate::AttentionProfile::Canonical)?
        );
    }
    Ok(())
}

/// Pooled worker quotes follow the capture width, without changing private sequence geometry.
#[test]
fn worker_quote_uses_actual_prompt_rows_and_keeps_unscaled_kv() -> Result<()> {
    let mut weights = fixture_weights();
    weights.prefill_width = 2048;
    weights.narrow_prefill_width = 32;
    let graph = fixture_graph(48)?;
    let profile = fixture_profile();
    let geometry = ProgramGeometry::worker(&weights, 256);
    assert_eq!(geometry.prompts, [256, 32]);
    assert_eq!(ProgramGeometry::new(&weights, true).prompts, [2048, 32]);
    assert_eq!(
        external_bytes(&weights, 48, geometry, &profile)?,
        257 * 48 * F32
    );
    let quote = |graph: &DataflowGraph, shared| {
        worker_storage_bytes(graph, &weights, (8192, 48, 48), 256, shared, &profile)
    };
    let plain = quote(&graph, false)?;
    assert_eq!(quote(&graph, true)?, plain);
    let mut with_kv = graph;
    let id = TensorId::new(9)?;
    with_kv.tensors.push(TensorSpec {
        id,
        shape: vec![64, 128],
        dtype: DType::F32,
        storage: TensorStorage::State {
            layer: 0,
            kind: StateKind::AttentionKv,
        },
    });
    // No FP8 scale: a shared paged FP8 pool does not own this F32 KV allocation.
    assert_eq!(quote(&with_kv, true)?, quote(&with_kv, false)?);
    assert_eq!(quote(&with_kv, true)? - plain, 8192 * 128 * 2 * F32);
    weights.kv_scales.insert(id, [1.0, 1.0]);
    let private = worker_storage_bytes(&with_kv, &weights, (8192, 48, 48), 256, false, &profile)?;
    let shared = worker_storage_bytes(&with_kv, &weights, (8192, 48, 48), 256, true, &profile)?;
    assert_eq!(private - shared, 8192 * 128 * 2);
    Ok(())
}

#[test]
fn pending_prompt_scratch_prices_only_uncached_captured_widths() -> Result<()> {
    let mut weights = fixture_weights();
    weights.prefill_width = 2048;
    weights.narrow_prefill_width = 64;
    let graph = fixture_graph(48)?;
    let profile = fixture_profile();
    for private in [false, true] {
        let geometry = ProgramGeometry::new(&weights, private);
        assert_eq!(
            missing_prompt_scratch_bytes(&graph, &weights, geometry, &profile)?,
            prompt_scratch_bytes(&graph, &weights, geometry, &profile)?
        );
    }
    assert_eq!(
        missing_prompt_scratch_bytes(
            &graph,
            &weights,
            ProgramGeometry {
                verification: 0,
                prompts: [0, 0]
            },
            &profile
        )?,
        0
    );
    Ok(())
}

/// Physical-page budgeting must preserve useful per-slot capacity and round to whole pages.
#[test]
fn funded_kv_rounds_down_without_changing_logical_capacity() -> Result<()> {
    let width = 16;
    let capacity = 8192;
    let maximum = width * 6144;
    let fixed = 5_200_689_152;
    let block_bytes = 1_048_576;
    let envelope = 7_569_557_647;
    let tokens = funded_kv_tokens(
        minimum_kv_tokens(width, capacity, 0)?,
        maximum,
        block_bytes,
        fixed,
        envelope,
    )
    .ok_or_else(|| Error::invariant("funded fixture"))?;
    assert_eq!(tokens, 72_288);
    assert_eq!(tokens % KV_BLOCK_TOKENS, 0);
    assert!(fixed + (tokens / KV_BLOCK_TOKENS) as u64 * block_bytes <= envelope);
    assert!(fixed + (tokens / KV_BLOCK_TOKENS + 1) as u64 * block_bytes > envelope);
    assert!(tokens >= capacity);
    // A known request length must fit every selected lane, including page rounding.
    assert_eq!(
        funded_kv_tokens(
            minimum_kv_tokens(width, capacity, capacity / 2 + 1)?,
            maximum,
            block_bytes,
            fixed,
            fixed + (width * capacity / 2 / KV_BLOCK_TOKENS) as u64 * block_bytes
        ),
        None
    );
    assert_eq!(
        funded_kv_tokens(capacity, maximum, 0, fixed, envelope),
        None
    );
    assert_eq!(
        funded_kv_tokens(capacity, maximum, block_bytes, fixed, fixed),
        None
    );
    Ok(())
}

/// Short requests can share physical pages without a half-capacity allotment per slot.
#[test]
fn funded_kv_admits_short_requests_without_per_slot_half_capacity() -> Result<()> {
    let width = 4;
    let capacity = 8192;
    let maximum = width * capacity;
    let block_bytes = 1024;
    let fixed = 100_000;
    // One full-context request, or four 512-token requests with ample spare pages.
    let physical_tokens = capacity;
    let envelope = fixed + (physical_tokens / KV_BLOCK_TOKENS) as u64 * block_bytes;
    assert_eq!(
        funded_kv_tokens(
            minimum_kv_tokens(width, capacity, 0)?,
            maximum,
            block_bytes,
            fixed,
            envelope
        ),
        Some(physical_tokens)
    );
    Ok(())
}

#[test]
fn real_request_kv_rounding_and_bounds_are_checked() -> Result<()> {
    assert_eq!(minimum_kv_tokens(4, 128, 1)?, 128);
    assert_eq!(minimum_kv_tokens(4, 128, 32)?, 128);
    assert_eq!(minimum_kv_tokens(4, 128, 33)?, 256);
    assert_eq!(minimum_kv_tokens(4, 128, 128)?, 512);
    assert!(minimum_kv_tokens(4, 128, 129).is_err());
    assert!(minimum_kv_tokens(usize::MAX, 128, 33).is_err());
    assert_eq!(funded_kv_tokens(256, 512, 10, 100, 179), None);
    assert_eq!(funded_kv_tokens(256, 512, 10, 100, 180), Some(256));
    Ok(())
}

/// Every captured target arena must be funded even though the graphs alias one state pool.
#[test]
fn pooled_quote_covers_all_narrow_verification_arenas() -> Result<()> {
    let graph = fixture_graph(48)?;
    let weights = fixture_weights();
    let bytes = pool_storage_bytes(&graph, &weights, (8, 256, 2048), 8, 48, &fixture_profile())?;
    let arenas = [1, 2, 4, 8]
        .into_iter()
        .map(|slots| ActivationArena::required_bytes(&graph, slots * 8))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>() as u64;
    assert!(
        bytes >= arenas,
        "quote {bytes} omits captured arenas {arenas}"
    );
    Ok(())
}
