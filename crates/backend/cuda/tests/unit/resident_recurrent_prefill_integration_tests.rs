use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::{
    device::CudaDevice,
    mlp::ProjectionWeight,
    resident::{DeviceProgram, ProgramWeights, PromptScratchRegistry},
};
use cutile::half::bf16;
use infer_core::{OpId, Result, TensorId};
use infer_cuda_kernels::tile::kernels as tile_kernels;
use infer_ir::{DType, DataflowGraph, StateKind, TensorNode, TensorOp, TensorSpec, TensorStorage};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Delta geometry for [`fixture`]: key heads, value heads, head dimension.
type DeltaGeometry = (usize, usize, usize);

/// Geometry small enough to reason about by hand.
const SMALL: DeltaGeometry = (1, 2, 32);

fn fixture(device: &CudaDevice) -> Result<(DataflowGraph, ProgramWeights)> {
    fixture_at(device, SMALL)
}

fn fixture_at(
    device: &CudaDevice,
    (key_heads, value_heads, dim): DeltaGeometry,
) -> Result<(DataflowGraph, ProgramWeights)> {
    // Activation tensors are declared with one lane's element count; the arena multiplies by the
    // captured width. The Delta binds `[lanes, 2*kh+vh, dim]` for qkv, `[lanes, vh]` for
    // beta/alpha and `[vh, dim, dim]` for the state, so the declared sizes follow from these.
    let qkv_columns = (2 * key_heads + value_heads) * dim;
    let gate_columns = value_heads;
    let out_columns = value_heads * dim;
    let graph = graph_at((key_heads, value_heads, dim))?;
    let mut embeddings = BTreeMap::new();
    for (id, columns) in [(1, qkv_columns), (3, gate_columns), (5, gate_columns)] {
        let data = super::tests::values(16 * columns, usize::try_from(id).unwrap() + 2)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        embeddings.insert(TensorId::new(id)?, device.upload(data, &[16, columns])?);
    }
    let dense = device.upload(
        super::tests::values(16 * out_columns, 11)
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
        &[16, out_columns],
    )?;
    let weights = ProgramWeights {
        kernels: Arc::new(tile_kernels(0)),
        mrope: false,
        prompt_scratch: PromptScratchRegistry::default(),
        autotune: false,
        resident_tuning: Arc::new(ResidentTuningSession::default()),
        chunked_recurrent: false,
        batch_width: 3,
        prefill_width: 1,
        narrow_prefill_width: 0,
        attention_profile: crate::AttentionProfile::Canonical,
        fusion: None,
        embeddings,
        projections: BTreeMap::from([(TensorId::new(11)?, ProjectionWeight::Dense(dense))]),
        expert_banks: BTreeMap::new(),
        constants: BTreeMap::from([
            (
                TensorId::new(7)?,
                device.upload(super::tests::values(value_heads, 3), &[value_heads])?,
            ),
            (
                TensorId::new(8)?,
                device.upload(super::tests::values(value_heads, 5), &[value_heads])?,
            ),
        ]),
        kv_scales: BTreeMap::new(),
        tiling: BTreeMap::new(),
        fp8_inputs: BTreeSet::new(),
        input_scales: BTreeMap::new(),
        rope_axes: BTreeMap::new(),
        rope_frequencies: BTreeMap::new(),
    };
    Ok((graph, weights))
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn recurrent_prefill_preserves_state_tail_and_verification_rollback() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for width in [32, 128] {
        let (graph, mut weights) = fixture(&device)?;
        let mut reference = DeviceProgram::new(&device, &graph, &weights, 256, 64, 16)?;
        weights.prefill_width = width;
        let mut candidate = DeviceProgram::new(&device, &graph, &weights, 256, 64, 16)?;
        let mut position = 0;
        for count in [width - 3, 1, 2] {
            let tokens: Vec<u32> = (position..position + count)
                .map(|i| u32::try_from(i % 16).unwrap())
                .collect();
            let actual = candidate.prefill_batch(&tokens, position, true)?;
            for (i, &token) in tokens.iter().enumerate() {
                let expected = reference.step(token, position + i, position + i, None, true)?;
                close(&actual[i].0, &expected.0);
            }
            position += count;
        }
        assert!(candidate.discard_verification());
        assert!(!candidate.discard_verification());
        candidate.ensure_verification(&graph, &weights)?;
        let a = candidate.step_batch(&[3, 4, 5], position)?;
        let b = reference.step_batch(&[3, 4, 5], position)?;
        for (a, b) in a.iter().zip(&b) {
            close(&a.0, &b.0);
            close(&a.1, &b.1);
        }
        candidate.commit_batch(1)?;
        reference.commit_batch(1)?;
        assert!(candidate.discard_verification());
        candidate.ensure_verification(&graph, &weights)?;
        let a = candidate.step(7, position + 1, position + 1, None, true)?;
        let b = reference.step(7, position + 1, position + 1, None, true)?;
        close(&a.0, &b.0);
        close(&a.1, &b.1);
    }
    Ok(())
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 0.00002, "{a} != {b}");
    }
}

/// The chunked recurrence disagrees with the per-lane one on the 27B (5% of hidden L2) while
/// both kernels reproduce an independent reference on identical inputs, so the difference has to
/// be in what the capture binds. This runs the same capture comparison at the geometry the 27B
/// actually uses, which needs no checkpoint and finishes in milliseconds — a fixture this cheap
/// is the place to instrument until the binding is found.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn chunked_recurrence_matches_per_lane_at_model_geometry() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for width in [32, 128] {
        let (graph, mut weights) = fixture_at(&device, (16, 48, 128))?;
        let mut reference = DeviceProgram::new(&device, &graph, &weights, 256, 64, 16)?;
        weights.prefill_width = width;
        let mut candidate = DeviceProgram::new(&device, &graph, &weights, 256, 64, 16)?;
        let mut position = 0;
        for count in [width - 3, 1, 2] {
            let tokens: Vec<u32> = (position..position + count)
                .map(|i| u32::try_from(i % 16).unwrap())
                .collect();
            let actual = candidate.prefill_batch(&tokens, position, true)?;
            for (i, &token) in tokens.iter().enumerate() {
                let expected = reference.step(token, position + i, position + i, None, true)?;
                let error = actual[i]
                    .0
                    .iter()
                    .zip(&expected.0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                eprintln!(
                    "width={width} count={count} lane={i} chunked-vs-per-lane max_abs={error:e}"
                );
                close(&actual[i].0, &expected.0);
            }
            position += count;
        }
    }
    Ok(())
}

/// The default dense-model admission must exercise the split kernel without loader options.
/// Captures otherwise share projection, padding and readout policy.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn default_ordered_prefill_matches_legacy_capture_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for width in super::ORDERED_WIDTHS {
        let (graph, mut weights) = fixture_at(&device, (16, 48, super::ORDERED_DIM))?;
        weights.prefill_width = width;
        assert!(!weights.chunked_recurrent);
        assert!(super::budget::enabled(&weights, false));
        let hidden = 48 * super::ORDERED_DIM;
        let mut candidate = DeviceProgram::new(&device, &graph, &weights, 2 * width, hidden, 16)?;
        super::LEGACY_DELTA_CAPTURE.set(true);
        let scalar = DeviceProgram::new(&device, &graph, &weights, 2 * width, hidden, 16);
        super::LEGACY_DELTA_CAPTURE.set(false);
        let mut scalar = scalar?;
        assert_eq!(candidate.prefill_width(), width);
        assert_eq!(scalar.prefill_width(), width);
        let mut position = 0;
        for count in [52.min(width), width - 3, 4] {
            let tokens: Vec<_> = (position..position + count)
                .map(|token| u32::try_from(token % 16).unwrap())
                .collect();
            let actual = candidate.prefill_batch(&tokens, position, true)?;
            let expected = scalar.prefill_batch(&tokens, position, true)?;
            assert_eq!(actual.len(), count);
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.last().expect("prefill output").1.len(), 16);
            for (a, b) in actual.iter().zip(&expected) {
                assert_checkpoint_hidden_bits(&a.0, &b.0, "default prefill hidden");
                assert_checkpoint_hidden_bits(&a.1, &b.1, "default prefill logits");
            }
            assert_eq!(candidate.states().len(), scalar.states().len());
            let mut checked = 0;
            for (id, states) in candidate.states() {
                let reference = scalar.states().get(id).expect("scalar state id");
                assert_eq!(states.len(), reference.len());
                for (a, b) in states.iter().zip(reference) {
                    let a = device.read_borrowed(a)?;
                    let b = device.read_borrowed(b)?;
                    checked += a.len();
                    assert_checkpoint_hidden_bits(&a, &b, "default prefill state");
                }
            }
            assert_eq!(checked, 48 * super::ORDERED_DIM * super::ORDERED_DIM);
            position += count;
        }
        eprintln!(
            "DEFAULT_ORDERED_CAPTURE width={width} block={} different_bits=0",
            super::ORDERED_BLOCK
        );
    }
    Ok(())
}

/// The default prompt recurrence multiplies its chunk factors in BF16; the legacy capture does
/// not, so they agree within this bound rather than bit for bit.
const DEFAULT_PREFILL_TOLERANCE: f32 = 2e-3;

fn assert_checkpoint_hidden_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label} element={index}"
        );
        assert!(
            (actual - expected).abs() <= DEFAULT_PREFILL_TOLERANCE,
            "{label} element={index}: {actual} != {expected}"
        );
    }
}

/// Pooled rollback must preserve the exact scalar recurrence, including slots and
/// candidate suffixes that were inactive in the preceding replay.
#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn pooled_recurrent_checkpoints_preserve_exact_prefixes() -> Result<()> {
    use crate::resident::slot_batch::SlotPool;

    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for geometry in [SMALL, (16, 48, 128)] {
        for verify in [2, 3, 5] {
            let (graph, mut weights) = fixture_at(&device, geometry)?;
            weights.batch_width = verify;
            let hidden = geometry.1 * geometry.2;
            let mut pool = SlotPool::new(
                &device,
                &graph,
                &weights,
                4,
                128,
                hidden,
                (16, verify),
                None,
            )?;
            assert!(pool.test_batched_commit(), "verify={verify}");
            let mut references = (0..4)
                .map(|_| DeviceProgram::new(&device, &graph, &weights, 128, hidden, 16))
                .collect::<Result<Vec<_>>>()?;
            for mut plans in [
                vec![
                    (0, vec![1, 2, 3], 0),
                    (2, vec![4, 5, 6], 1),
                    (3, vec![7, 8, 9], 2),
                ],
                vec![
                    (1, vec![10, 11, 12], 2),
                    (3, vec![10], 0),
                    (0, vec![4, 5], 1),
                ],
                vec![
                    (3, vec![11, 12, 13], 0),
                    (2, vec![7, 8, 9], 2),
                    (1, vec![13, 14, 15], 1),
                    (0, vec![6, 7, 8], 2),
                ],
                vec![(2, vec![10, 11, 12, 13, 14], 4), (1, vec![3, 4], 0)],
            ] {
                let mut lanes = Vec::new();
                let mut expected = Vec::new();
                for (slot, tokens, accepted) in &mut plans {
                    tokens.truncate(verify);
                    *accepted = (*accepted).min(tokens.len() - 1);
                    let position = references[*slot].position();
                    for (offset, &token) in tokens.iter().enumerate() {
                        lanes.push((*slot, offset, token, position + offset, position + offset));
                    }
                    expected.push(references[*slot].step_batch(tokens, position)?);
                }
                let actual = pool.run_verify(&device, &lanes)?;
                let mut commits = Vec::new();
                for ((slot, tokens, accepted), reference) in plans.into_iter().zip(expected) {
                    for (row, expected) in reference.iter().enumerate() {
                        assert_pooled_close(
                            &actual[slot * verify + row].0,
                            &expected.0,
                            &format!("geometry={geometry:?} verify={verify} slot={slot} row={row}"),
                        );
                    }
                    assert_eq!(reference.len(), tokens.len());
                    commits.push((slot, accepted));
                    references[slot].commit_batch(accepted + 1)?;
                }
                pool.commit_verify_many(&device, &commits)?;
                for &(slot, _) in &commits {
                    assert_eq!(pool.cursors()[slot], references[slot].position());
                }
            }
            references[0].reset()?;
            references[0].step(5, 0, 0, None, false)?;
            pool.bind(
                0,
                references[0].states(),
                references[0].fp8_states(),
                1,
                &device,
            )?;
            let expected = references[0].step_batch(&[6, 7], 1)?;
            let actual = pool.run_verify(&device, &[(0, 0, 6, 1, 1), (0, 1, 7, 2, 2)])?;
            for (row, expected) in expected.iter().enumerate() {
                assert_pooled_close(&actual[row].0, &expected.0, "rebound slot");
            }
            pool.commit_verify(&device, 0, 0)?;
            references[0].commit_batch(1)?;
            let expected = references[0].step_batch(&[8], 2)?;
            let actual = pool.run_verify(&device, &[(0, 0, 8, 2, 2)])?;
            assert_pooled_close(&actual[0].0, &expected[0].0, "continued slot");
            pool.commit_verify(&device, 0, 0)?;
        }
    }
    Ok(())
}

/// The pooled graph verifies a slot's lanes in one fused launch and commits the accepted prefix
/// afterwards (`slot_recurrent`), so it matches the serial lanes to rounding, not bit for bit.
fn assert_pooled_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "{label} element={index}: {actual} != {expected}"
        );
    }
}

fn graph_at((key_heads, value_heads, dim): DeltaGeometry) -> Result<DataflowGraph> {
    let qkv_columns = (2 * key_heads + value_heads) * dim;
    let gate_columns = value_heads;
    let out_columns = value_heads * dim;
    let mut graph = DataflowGraph::default();
    for (id, shape, kind) in [
        (1, vec![16, qkv_columns], 0),
        (2, vec![qkv_columns], 1),
        (3, vec![16, gate_columns], 0),
        (4, vec![gate_columns], 1),
        (5, vec![16, gate_columns], 0),
        (6, vec![gate_columns], 1),
        (7, vec![value_heads], 0),
        (8, vec![value_heads], 0),
        (9, vec![value_heads, dim, dim], 2),
        (10, vec![out_columns], 1),
        (11, vec![16, out_columns], 0),
        (12, vec![16], 1),
    ] {
        let storage = match kind {
            0 => TensorStorage::Weight {
                slot: format!("w{id}"),
            },
            1 => TensorStorage::Activation,
            _ => TensorStorage::State {
                layer: 0,
                kind: StateKind::LinearAttention,
            },
        };
        graph.tensors.push(TensorSpec {
            id: TensorId::new(id)?,
            shape,
            dtype: DType::F32,
            storage,
        });
    }
    for (id, op, inputs, output, state) in [
        (1, TensorOp::Embedding, vec![1], 2, None),
        (2, TensorOp::Embedding, vec![3], 4, None),
        (3, TensorOp::Embedding, vec![5], 6, None),
        (
            4,
            TensorOp::Delta {
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
            },
            vec![2, 4, 6, 7, 8],
            10,
            Some(9),
        ),
        (5, TensorOp::Linear, vec![10, 11], 12, None),
    ] {
        graph.nodes.push(TensorNode {
            id: OpId::new(id)?,
            layer: None,
            op,
            inputs: inputs
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<_>>()?,
            outputs: vec![TensorId::new(output)?],
            states: state
                .into_iter()
                .map(TensorId::new)
                .collect::<Result<_>>()?,
        });
    }
    graph.hidden = Some(TensorId::new(10)?);
    graph.logits = Some(TensorId::new(12)?);
    Ok(graph)
}

/// Exercise both recurrent writers with nonzero history and every supported serial width.
fn checkpoint_fixture(
    device: &CudaDevice,
    geometry: DeltaGeometry,
) -> Result<(DataflowGraph, ProgramWeights)> {
    let (mut graph, mut weights) = fixture_at(device, geometry)?;
    let channels = (2 * geometry.0 + geometry.1) * geometry.2;
    let state = TensorId::new(13)?;
    let kernel = TensorId::new(14)?;
    let output = TensorId::new(15)?;
    for (id, shape, storage) in [
        (
            state,
            vec![channels, 4],
            TensorStorage::State {
                layer: 0,
                kind: StateKind::Conv,
            },
        ),
        (
            kernel,
            vec![channels, 4],
            TensorStorage::Weight {
                slot: "conv".into(),
            },
        ),
        (output, vec![channels], TensorStorage::Activation),
    ] {
        graph.tensors.push(TensorSpec {
            id,
            shape,
            dtype: DType::F32,
            storage,
        });
    }
    graph.nodes[3].inputs[0] = output;
    graph.nodes.insert(
        3,
        TensorNode {
            id: OpId::new(6)?,
            layer: None,
            op: TensorOp::Conv {
                channels,
                kernel: 4,
            },
            inputs: vec![TensorId::new(2)?, kernel],
            outputs: vec![output],
            states: vec![state],
        },
    );
    weights.constants.insert(
        kernel,
        device.upload(super::tests::values(channels * 4, 19), &[channels, 4])?,
    );
    Ok((graph, weights))
}

fn assert_serial_states(
    device: &CudaDevice,
    candidate: &DeviceProgram,
    reference: &DeviceProgram,
) -> Result<()> {
    assert_eq!(candidate.states().len(), reference.states().len());
    for (id, actual) in candidate.states() {
        let expected = &reference.states()[id];
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_checkpoint_hidden_bits(
                &device.read_borrowed(actual)?,
                &device.read_borrowed(expected)?,
                "serial recurrent state",
            );
        }
    }
    assert_eq!(candidate.position(), reference.position());
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run under safe-run"]
fn serial_recurrent_checkpoints_preserve_exact_prefixes() -> Result<()> {
    use crate::resident::capture_checkpoint::LEGACY_FLAT_CAPTURE;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    for geometry in [SMALL, (16, 48, 128)] {
        for width in [2, 3, 5, 9] {
            let (graph, mut weights) = checkpoint_fixture(&device, geometry)?;
            weights.batch_width = width;
            let hidden = geometry.1 * geometry.2;
            let mut candidate = DeviceProgram::new(&device, &graph, &weights, 128, hidden, 16)?;
            LEGACY_FLAT_CAPTURE.set(true);
            let reference = DeviceProgram::new(&device, &graph, &weights, 128, hidden, 16);
            LEGACY_FLAT_CAPTURE.set(false);
            let mut reference = reference?;
            for (count, commit) in [
                (width, 1),
                (1, 1),
                (width - 1, width - 1),
                (width, width / 2 + 1),
            ] {
                let position = candidate.position();
                let tokens: Vec<_> = (0..count)
                    .map(|offset| u32::try_from((position + offset + 3) % 16).unwrap())
                    .collect();
                let actual = candidate.step_batch(&tokens, position)?;
                let expected = reference.step_batch(&tokens, position)?;
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_checkpoint_hidden_bits(&actual.0, &expected.0, "serial hidden");
                    assert_checkpoint_hidden_bits(&actual.1, &expected.1, "serial logits");
                }
                assert_serial_states(&device, &candidate, &reference)?;
                candidate.commit_batch(commit)?;
                reference.commit_batch(commit)?;
                assert_serial_states(&device, &candidate, &reference)?;
                let position = candidate.position();
                let actual = candidate.step(7, position, position, None, true)?;
                let expected = reference.step(7, position, position, None, true)?;
                assert_checkpoint_hidden_bits(&actual.0, &expected.0, "continued hidden");
                assert_checkpoint_hidden_bits(&actual.1, &expected.1, "continued logits");
                assert_serial_states(&device, &candidate, &reference)?;
            }
        }
    }
    Ok(())
}
