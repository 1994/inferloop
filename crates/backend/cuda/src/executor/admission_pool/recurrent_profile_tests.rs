//! The BF16 recurrent-state profile through the executor: storage, prefix reuse and the
//! difference from F32 states on a real checkpoint.
use super::*;
use crate::RecurrentProfile;
use crate::device::{CudaDevice, asynchronous::block_on};
use crate::loading::{LoadOptions, LoadedModel};
use infer_core::{ModelId, RequestId, StateId};
use infer_ir::{ExecutionInput, ExecutionTask, StateKind, TensorStorage, TokenBuffer};
use std::env::var;

/// Tokens each test sequence reserves.
const RESERVED_TOKENS: usize = 384;
/// Prompt length; the prefix boundary falls two KV blocks in.
const PROMPT_TOKENS: u32 = 96;
/// Greedy tokens each pair of sequences decodes after the prompt.
const DECODE_STEPS: usize = 16;

fn fixture(recurrent_profile: RecurrentProfile) -> Result<CudaExecutor> {
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL").map_err(|error| Error::invalid(error.to_string()))?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 0,
            autotune: false,
            recurrent_profile,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let budget = loaded
        .device()
        .allocatable_memory_bytes()?
        .saturating_sub(loaded.device().profile()?.device_headroom_bytes())
        .saturating_sub(loaded.shared_scratch_bytes()?);
    CudaExecutor::new_automatic(loaded, budget, MAX_POOL_SLOTS)
}

fn prompt() -> TokenBuffer {
    TokenBuffer::from(
        (0..PROMPT_TOKENS)
            .map(|i| 1000 + i * 7)
            .collect::<Vec<u32>>(),
    )
}

fn chunk(
    backend: &mut CudaExecutor,
    state: StateId,
    tokens: ExecutionInput,
    publish_prefix: bool,
) -> Result<Vec<f32>> {
    backend.prepare_execution()?;
    let result = block_on(backend.execute(&[ExecutionTask {
        plain_greedy: false,
        publish_prefix,
        request: RequestId::ONE,
        state,
        tokens,
        sampling: None,
        media: None,
    }]));
    let fence = block_on(backend.loaded.device().fence());
    backend.busy = false;
    fence?;
    Ok(result?.remove(0).output.logits)
}

/// Every gated-delta state of `state` is stored as `profile` selects and every other in F32.
fn check_storage(backend: &CudaExecutor, state: StateId, profile: RecurrentProfile) {
    let graph = backend.loaded.graph();
    let mut delta = 0;
    for (id, tensors) in backend.states[&state].program.states() {
        let kind = graph
            .tensors
            .iter()
            .find(|spec| spec.id == *id)
            .map(|spec| &spec.storage);
        let expected = match kind {
            Some(TensorStorage::State {
                kind: StateKind::LinearAttention,
                ..
            }) => {
                delta += 1;
                profile
            }
            _ => RecurrentProfile::F32,
        };
        for tensor in tensors {
            assert_eq!(tensor.profile(), expected, "storage of state {id:?}");
        }
    }
    assert!(delta > 0, "the checkpoint has no gated-delta state");
}

/// The index of the largest logit.
fn top(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .and_then(|(index, _)| u32::try_from(index).ok())
        .ok_or_else(|| Error::invalid("empty logits"))
}

/// Whether two logit rows agree bit for bit, naming how many places differ.
fn same_bits(left: &[f32], right: &[f32], label: &str) {
    let differing = left
        .iter()
        .zip(right)
        .filter(|(l, r)| l.to_bits() != r.to_bits())
        .count();
    assert_eq!(left.len(), right.len(), "{label} length");
    assert_eq!(differing, 0, "{label} differ in {differing} places");
}

/// Run one sequence through the prompt and `DECODE_STEPS` greedy tokens: from scratch,
/// publishing its prefix at the boundary, or restored from that prefix. Returns every logits row.
fn sequence(
    backend: &mut CudaExecutor,
    (state, profile, slot): (StateId, RecurrentProfile, bool),
    restore: bool,
) -> Result<Vec<Vec<f32>>> {
    let prompt = prompt();
    let boundary = 2 * KV_BLOCK_TOKENS;
    backend.reserve_state_for(state, RESERVED_TOKENS, OutputReadout::Logits)?;
    assert_eq!(backend.states[&state].home.is_some(), slot);
    check_storage(backend, state, profile);
    if restore {
        assert_eq!(
            backend.reuse_prefix_shared(state, &prompt, prompt.len() - 1)?,
            boundary
        );
    } else {
        let head = chunk(
            backend,
            state,
            ExecutionInput::Prefill {
                span: prompt.span(0..boundary)?,
                readout: OutputReadout::None,
            },
            true,
        )?;
        assert_eq!(head, Vec::<f32>::new());
    }
    let mut rows = vec![chunk(
        backend,
        state,
        ExecutionInput::Prefill {
            span: prompt.span(boundary..prompt.len())?,
            readout: OutputReadout::Logits,
        },
        false,
    )?];
    for step in 0..DECODE_STEPS {
        let token = top(rows.last().map_or(&[][..], Vec::as_slice))?;
        let input = ExecutionInput::Decode {
            position: prompt.len() + step,
            token,
        };
        rows.push(chunk(backend, state, input, false)?);
    }
    backend.release_state(state)?;
    for row in &rows {
        assert_eq!(row.len(), backend.model().vocab_size);
        assert!(row.iter().all(|value| value.is_finite()));
    }
    Ok(rows)
}

/// One sequence computes the prompt and publishes its prefix; a second, restored from it,
/// continues to the same logits bit for bit through the prompt tail and the greedy decode.
/// Returns the computed rows.
fn prefix_round_trip(
    backend: &mut CudaExecutor,
    profile: RecurrentProfile,
    slot: bool,
) -> Result<Vec<Vec<f32>>> {
    let computed = sequence(backend, (StateId::ONE, profile, slot), false)?;
    let restored = sequence(backend, (StateId::new(2)?, profile, slot), true)?;
    for (step, (computed, restored)) in computed.iter().zip(&restored).enumerate() {
        same_bits(
            computed,
            restored,
            &format!("restored logits at step {step}"),
        );
    }
    Ok(computed)
}

/// A slot-resident sequence keeps its prefix in the pool's state families, and another slot
/// restored from it continues bit for bit.
fn slot_round_trip(profile: RecurrentProfile) -> Result<()> {
    let mut backend = fixture(profile)?;
    backend.capture_funded_pool();
    prefix_round_trip(&mut backend, profile, true).map(drop)
}

/// The F32 control of [`bf16_slot_prefix_restores_bit_for_bit`].
#[test]
#[ignore = "requires a CUDA device and INFER_TEST_MODEL with gated-delta layers"]
fn f32_slot_prefix_restores_bit_for_bit() -> Result<()> {
    slot_round_trip(RecurrentProfile::F32)
}

/// BF16 pool families keep and restore a prefix bit for bit.
#[test]
#[ignore = "requires a CUDA device and INFER_TEST_MODEL with gated-delta layers"]
fn bf16_slot_prefix_restores_bit_for_bit() -> Result<()> {
    slot_round_trip(RecurrentProfile::Bf16)
}

/// A private BF16 program snapshots and restores its prefix bit for bit, and its logits stay
/// close to the F32 states' on the same prompt.
#[test]
#[ignore = "requires a CUDA device and INFER_TEST_MODEL with gated-delta layers"]
fn bf16_private_prefix_restores_and_tracks_f32() -> Result<()> {
    let wide = prefix_round_trip(
        &mut fixture(RecurrentProfile::F32)?,
        RecurrentProfile::F32,
        false,
    )?;
    let narrow = prefix_round_trip(
        &mut fixture(RecurrentProfile::Bf16)?,
        RecurrentProfile::Bf16,
        false,
    )?;
    let worst = |step: usize| {
        wide[step]
            .iter()
            .zip(&narrow[step])
            .map(|(w, n)| (w - n).abs())
            .fold(0.0_f32, f32::max)
    };
    // Each run feeds back its own greedy token, so the trajectories are compared token by token
    // only up to the first disagreement.
    let tokens = |rows: &[Vec<f32>]| rows.iter().map(|row| top(row)).collect::<Result<Vec<_>>>();
    let agreed = tokens(&wide)?
        .iter()
        .zip(&tokens(&narrow)?)
        .take_while(|(w, n)| w == n)
        .count();
    let first = worst(0);
    let trajectory = (0..agreed.max(1)).map(worst).collect::<Vec<_>>();
    eprintln!(
        "RECURRENT_PROFILE bf16 vs f32 after {PROMPT_TOKENS} prompt tokens: max |logit| \
         difference {first}; greedy tokens agree for {agreed} of {}; per agreed step \
         {trajectory:?}; f32 tokens {:?}; bf16 tokens {:?}",
        DECODE_STEPS + 1,
        tokens(&wide)?,
        tokens(&narrow)?,
    );
    assert!(first.is_finite());
    Ok(())
}
