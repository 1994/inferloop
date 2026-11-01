//! Numerical checks for graph replay, recurrent state and grouped attention.
mod batch;
mod fusion;
mod projection_weights;
mod quantization;
mod rounded_projection;

use cutile::half::bf16;
use cutile::jit_cache::enable_default;
use infer_backend_cuda::mlp::ProjectionWeight;
use infer_backend_cuda::tuning::resident_gemm::ResidentTuningSession;
use infer_backend_cuda::{
    device::CudaDevice,
    resident::{DeviceProgram, ProgramWeights, PromptScratchRegistry},
};
use infer_core::{Error, OpId, Result, TensorId};
use infer_ir::{DType, DataflowGraph, StateKind, TensorNode, TensorOp, TensorSpec, TensorStorage};
use infer_state::physical::{PagedRows, PhysicalTensor};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

#[expect(
    clippy::cast_precision_loss,
    reason = "Small deterministic test dimensions fit exactly in f32"
)]
fn values(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.37).sin() * 0.2)
        .collect()
}

fn check(
    device: &CudaDevice,
    op: TensorOp,
    inputs: Vec<Vec<f32>>,
    size: usize,
    state: Option<(StateKind, Vec<usize>, PhysicalTensor)>,
) -> Result<()> {
    check_projection(device, op, inputs, size, state, None)
}

/// Activation tensor spec used by the fixture's graph.
fn activation(id: TensorId, size: usize) -> TensorSpec {
    TensorSpec {
        id,
        shape: vec![size],
        dtype: DType::F32,
        storage: TensorStorage::Activation,
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "Test cases own their input fixtures for clarity"
)]
fn check_projection(
    device: &CudaDevice,
    op: TensorOp,
    inputs: Vec<Vec<f32>>,
    size: usize,
    state: Option<(StateKind, Vec<usize>, PhysicalTensor)>,
    projection: Option<ProjectionWeight>,
) -> Result<()> {
    let mut graph = DataflowGraph::default();
    let mut weights = ProgramWeights {
        kernels: device.kernels(),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        chunked_recurrent: false,
        batch_width: 3,
        prefill_width: 3,
        narrow_prefill_width: 0,
        attention_profile: infer_backend_cuda::AttentionProfile::Canonical,
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fusion: None,
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        projections: BTreeMap::new(),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::new(),
        embeddings: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
    };
    set_attention_scale(&op, &mut weights)?;
    let mut ids = Vec::new();
    for (i, data) in inputs.iter().enumerate() {
        let id = TensorId::new(i as u64 + 1)?;
        ids.push(id);
        graph.tensors.push(TensorSpec {
            id,
            shape: vec![data.len()],
            dtype: DType::F32,
            storage: TensorStorage::Weight {
                slot: format!("input{i}"),
            },
        });
        weights
            .constants
            .insert(id, device.upload(data.clone(), &[data.len()])?);
    }
    if let Some(projection) = projection {
        weights.constants.remove(&TensorId::new(2)?);
        weights.projections.insert(TensorId::new(2)?, projection);
        graph.tensors[1].shape = vec![size, inputs[0].len()];
    }
    graph.tensors[0].storage = TensorStorage::Activation;
    weights.constants.remove(&TensorId::ONE);
    add_external(device, &mut graph, &mut weights, inputs[0].len())?;
    if matches!(op, TensorOp::Attention { .. }) {
        dynamic_kv(&mut graph, &mut weights)?;
    }
    let hidden = TensorId::new(100)?;
    let logits = TensorId::new(101)?;
    graph
        .tensors
        .extend([activation(hidden, size), activation(logits, size)]);
    let mut state_ids = vec![];
    let initial = if let Some((kind, shape, value)) = state {
        let id = TensorId::new(102)?;
        graph.tensors.push(TensorSpec {
            id,
            shape,
            dtype: DType::F32,
            storage: TensorStorage::State { layer: 0, kind },
        });
        state_ids.push(id);
        Some(value)
    } else {
        None
    };
    graph.nodes.push(TensorNode {
        id: OpId::ONE,
        layer: None,
        op: op.clone(),
        inputs: ids,
        outputs: vec![hidden],
        states: state_ids,
    });
    graph.nodes.push(TensorNode {
        id: OpId::new(2)?,
        layer: None,
        op: TensorOp::Add,
        inputs: vec![hidden, hidden],
        outputs: vec![logits],
        states: vec![],
    });
    graph.hidden = Some(hidden);
    graph.logits = Some(logits);
    add_rope(device, &op, &mut weights)?;
    let mut program = DeviceProgram::new(device, &graph, &weights, 64, inputs[0].len(), 256)?;
    batch::prompt32(device, &graph, &mut weights, &op, &inputs, initial.as_ref())?;
    drop(weights);
    replay(&mut program, &op, &inputs, initial.as_ref())?;
    batch::run(&mut program, &op, &inputs, initial.as_ref())?;
    println!("PASS {op:?}: varied inputs, state reset and applicable KV rollback");
    Ok(())
}

fn dynamic_kv(graph: &mut DataflowGraph, weights: &mut ProgramWeights) -> Result<()> {
    let key = TensorId::new(2)?;
    let value = TensorId::new(3)?;
    weights.constants.remove(&key);
    weights.constants.remove(&value);
    graph.tensors[1].storage = TensorStorage::Activation;
    graph.tensors[2].storage = TensorStorage::Activation;
    graph.nodes.push(TensorNode {
        id: OpId::new(4)?,
        layer: None,
        op: TensorOp::Split {
            widths: vec![32, 32],
            heads: 1,
        },
        inputs: vec![TensorId::ONE],
        outputs: vec![key, value],
        states: vec![],
    });
    Ok(())
}

fn add_external(
    device: &CudaDevice,
    graph: &mut DataflowGraph,
    weights: &mut ProgramWeights,
    width: usize,
) -> Result<()> {
    let id = TensorId::new(103)?;
    graph.tensors.push(TensorSpec {
        id,
        shape: vec![256, width],
        dtype: DType::Bf16,
        storage: TensorStorage::Weight {
            slot: "external-embedding".into(),
        },
    });
    let table = (0u16..256)
        .flat_map(|token| {
            values(width, f32::from(token))
                .into_iter()
                .map(bf16::from_f32)
        })
        .collect();
    weights
        .embeddings
        .insert(id, device.upload(table, &[256, width])?);
    graph.nodes.push(TensorNode {
        id: OpId::new(3)?,
        layer: None,
        op: TensorOp::Embedding,
        inputs: vec![id],
        outputs: vec![TensorId::ONE],
        states: vec![],
    });
    Ok(())
}

fn replay(
    program: &mut DeviceProgram,
    op: &TensorOp,
    inputs: &[Vec<f32>],
    initial: Option<&PhysicalTensor>,
) -> Result<()> {
    for _ in 0..2 {
        let mut reference = initial.cloned();
        let mut checkpoint = None;
        let steps = if matches!(op, TensorOp::Attention { .. }) {
            40
        } else {
            8
        };
        for position in 0..steps {
            if position == 17 {
                checkpoint.clone_from(&reference);
            }
            replay_one(program, op, inputs, &mut reference, position)?;
        }
        if matches!(op, TensorOp::Attention { .. }) {
            program.rewind_attention(17)?;
            for position in 17..steps {
                replay_one(program, op, inputs, &mut checkpoint, position)?;
            }
        }
    }
    Ok(())
}

fn replay_one(
    program: &mut DeviceProgram,
    op: &TensorOp,
    inputs: &[Vec<f32>],
    state: &mut Option<PhysicalTensor>,
    position: usize,
) -> Result<()> {
    let external = values(
        inputs[0].len(),
        f32::from(u16::try_from(position).map_err(|e| Error::invalid(e.to_string()))?),
    );
    let mut refs: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
    refs[0] = &external;
    if matches!(op, TensorOp::Attention { .. }) {
        refs[1] = &external[..32];
        refs[2] = &external[32..];
    }
    let (quantized_keys, quantized_values);
    let scaled_attention = matches!(
        op,
        TensorOp::Attention {
            window: Some(35),
            ..
        }
    );
    if scaled_attention {
        quantized_keys = quantization::round(&external[..32], 0.001);
        quantized_values = quantization::round(&external[32..], 0.002);
        refs[1] = &quantized_keys;
        refs[2] = &quantized_values;
    }
    let expected =
        infer_reference_ops::reference_operation(op, &refs, state.as_mut(), 0, position)?;
    let (actual, logits) = program.step(
        0,
        position,
        position,
        Some(&external),
        position.is_multiple_of(2),
    )?;
    for (i, (&a, &b)) in actual.iter().zip(&expected[0]).enumerate() {
        if !a.is_finite()
            || (a - b).abs() > 3e-4 * b.abs().max(1.0)
            || (!logits.is_empty() && a.mul_add(-2.0, logits[i]).abs() > 1e-7)
        {
            return Err(Error::invariant(format!(
                "{op:?} position={position} index={i}: GPU={a}, CPU={b}"
            )));
        }
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Bounded rotary dimensions, intentionally rounded F32 device frequencies"
)]
fn add_rope(device: &CudaDevice, op: &TensorOp, weights: &mut ProgramWeights) -> Result<()> {
    if let TensorOp::Rope {
        rotary_dim, theta, ..
    } = *op
    {
        let frequencies = (0..rotary_dim / 2)
            .map(|i| theta.powf(-((2 * i) as f64) / rotary_dim as f64) as f32)
            .collect();
        weights
            .rope_frequencies
            .insert(OpId::ONE, device.upload(frequencies, &[rotary_dim / 2])?);
    }
    Ok(())
}

pub fn run() -> Result<()> {
    enable_default().map_err(|e| Error::invalid(e.to_string()))?;
    let device = CudaDevice::new(0)?;
    fusion::run(&device)?;
    projections(&device)?;
    check(
        &device,
        TensorOp::Conv {
            channels: 384,
            kernel: 4,
        },
        vec![values(384, 0.0), values(1536, 1.0)],
        384,
        Some((
            StateKind::Conv,
            vec![384, 4],
            PhysicalTensor::Conv {
                channels: 384,
                kernel: 4,
                history: vec![0.0; 1152],
            },
        )),
    )?;
    check(
        &device,
        TensorOp::Delta {
            key_heads: 2,
            value_heads: 4,
            key_dim: 16,
            value_dim: 16,
        },
        vec![
            values(128, 0.0),
            values(4, 1.0),
            values(4, 2.0),
            values(4, 3.0),
            values(4, 4.0),
        ],
        64,
        Some((
            StateKind::LinearAttention,
            vec![4, 16, 16],
            PhysicalTensor::Delta {
                heads: 4,
                key_dim: 16,
                value_dim: 16,
                recurrent: vec![0.0; 1024],
            },
        )),
    )?;
    for window in [None, Some(3), Some(35)] {
        check(
            &device,
            TensorOp::Attention {
                query_heads: 4,
                kv_heads: 2,
                head_dim: 16,
                window,
            },
            vec![values(64, 0.0), values(32, 1.0), values(32, 2.0)],
            64,
            Some((
                StateKind::AttentionKv,
                vec![64, 32],
                PhysicalTensor::Kv {
                    keys: PagedRows::new(32, 4, 128)?,
                    values: PagedRows::new(32, 4, 128)?,
                },
            )),
        )?;
    }
    check_pointwise(&device)
}

fn check_pointwise(device: &CudaDevice) -> Result<()> {
    check(
        device,
        TensorOp::Norm {
            epsilon: 1e-6,
            offset: 1.0,
            head_dim: 48,
        },
        vec![values(144, 0.0), values(48, 1.0)],
        144,
        None,
    )?;
    check(
        device,
        TensorOp::GatedNorm {
            epsilon: 1e-6,
            head_dim: 16,
        },
        vec![values(64, 0.0), values(64, 1.0), values(16, 2.0)],
        64,
        None,
    )?;
    check(
        device,
        TensorOp::Rope {
            heads: 4,
            head_dim: 32,
            rotary_dim: 16,
            theta: 10000.0,
        },
        vec![values(128, 0.0)],
        128,
        None,
    )?;
    for op in [TensorOp::Silu, TensorOp::Sigmoid] {
        check(device, op, vec![values(560, 0.0)], 560, None)?;
    }
    Ok(())
}

fn projections(device: &CudaDevice) -> Result<()> {
    for (rows, columns) in [(65, 48), (97, 80)] {
        for encoding in 0..4 {
            let (weight, reference) = projection_weights::make(device, encoding, rows, columns)?;
            check_projection(
                device,
                TensorOp::Linear,
                vec![values(columns, 0.0), reference],
                rows,
                None,
                Some(weight),
            )?;
        }
    }
    Ok(())
}

fn set_attention_scale(op: &TensorOp, weights: &mut ProgramWeights) -> Result<()> {
    let scaled_attention = matches!(
        op,
        TensorOp::Attention {
            window: Some(35),
            ..
        }
    );
    if scaled_attention {
        weights
            .kv_scales
            .insert(TensorId::new(102)?, [0.001, 0.002]);
    }
    Ok(())
}
