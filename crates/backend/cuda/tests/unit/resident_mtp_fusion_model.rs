//! Real MTP2 gate: independent default FC/GDN target and draft versus original captures.
//! Both target timelines compare complete hidden, vocabulary and actual F32/FP8 state bits.
use super::prefix_state::{DraftSnapshot, ProgramSnapshot};
use crate::tuning::table_path as tuning_table_path;
use crate::{
    device::{CudaDevice, device_error},
    loading::{LoadOptions, LoadedModel},
    resident::{DeviceProgram, fusion_batch, narrow_dense_captures, with_dense_reference},
};
use cutile::prelude::{DType, Tensor};
use infer_core::{Error, ModelId, Result, TensorId};
use infer_ir::{DataflowGraph, OutputReadout, StateKind, TensorStorage};
use sha2::{Digest, Sha256};
use std::env::var;
use std::fs::read;
use std::iter::once;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

const CAPACITY: usize = 32768;
const HIDDEN: usize = 5120;
const VOCABULARY: usize = 248_320;
// G1 keeps the 128x10240 fusion input and adds the 128x5120 gathered embedding rows.
const SCRATCH_BYTES: usize = 128 * (10240 + 5120) * size_of::<f32>();
const PROMPTS: [usize; 5] = [127, 128, 129, 255, 511];
type StateDigest = BTreeMap<TensorId, Vec<[u8; 32]>>;

#[derive(Default)]
struct Coverage {
    hidden: usize,
    logits: usize,
    dense_state: usize,
    packed_state: usize,
    target_dense_finite: usize,
    target_hidden: usize,
    target_logits: usize,
    target_packed_finite: usize,
}

struct Timelines {
    target: DeviceProgram,
    target_reference: DeviceProgram,
    candidate: DeviceProgram,
    reference: DeviceProgram,
    previous_target_hidden: Vec<f32>,
    packed_captures: usize,
    narrow_captures: usize,
    coverage: Coverage,
}

impl Timelines {
    fn new(loaded: &LoadedModel) -> Result<Self> {
        let price = loaded.sequence_budget(CAPACITY, OutputReadout::Logits, true)?;
        let total_price = price
            .checked_mul(2)
            .ok_or_else(|| Error::invariant("two independent timeline price overflow"))?;
        loaded.device().reclaim_barrier()?;
        let (driver_free, total) = loaded.device().memory_info()?;
        let free = driver_free
            .saturating_add(loaded.device().pool_reclaimable_bytes()?)
            .min(total);
        let headroom = loaded.profile().device_headroom_bytes();
        assert!(
            total_price <= free.saturating_sub(headroom),
            "two target/draft pairs exceed original device headroom"
        );
        let before_narrow = narrow_dense_captures();
        let target = loaded.sequence(CAPACITY)?;
        let narrow_captures = narrow_dense_captures() - before_narrow;
        assert!(narrow_captures > 0, "default GDN N16 capture not used");
        let before_reference = narrow_dense_captures();
        let target_reference = with_dense_reference(|| loaded.sequence(CAPACITY))?;
        assert_eq!(
            narrow_dense_captures(),
            before_reference,
            "original target reference captured candidate N16"
        );
        let (candidate, reference, packed_captures) = capture_drafts(loaded)?;
        for program in [&target, &target_reference] {
            assert_eq!(program.prefill_width(), 64);
            assert_eq!(program.batch_width(), 3);
            assert!(!program.fp8_states().is_empty(), "target FP8 KV coverage");
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "MTP_FC_MODEL_ALLOCATION": true, "capacity": CAPACITY,
                "single_target_and_draft_sequence_price": price,
                "two_independent_target_and_draft_price_upper_bound": total_price,
                "free_before_programs": free, "driver_free_before_programs": driver_free,
                "device_headroom_bytes": headroom, "independent_target_count": 2,
                "candidate_scratch_bytes": SCRATCH_BYTES, "reference_scratch_bytes": 0,
                "autotune": false, "target_arithmetic_independent_reference": true,
                "candidate_narrow_dense_captures": narrow_captures,
                "reference_narrow_dense_captures": 0,
            })
        );
        Ok(Self {
            target,
            target_reference,
            candidate,
            reference,
            previous_target_hidden: Vec::new(),
            packed_captures,
            narrow_captures,
            coverage: Coverage::default(),
        })
    }
}

fn capture_drafts(loaded: &LoadedModel) -> Result<(DeviceProgram, DeviceProgram, usize)> {
    let before = fusion_batch::captures();
    let candidate = loaded
        .draft(CAPACITY)?
        .ok_or_else(|| Error::invariant("real MTP2 draft missing"))?;
    let packed_captures = fusion_batch::captures() - before;
    assert!(packed_captures > 0, "default FC128 capture not used");
    assert_eq!(candidate.test_fusion_batch_scratch_bytes(), SCRATCH_BYTES);
    let before_reference = fusion_batch::captures();
    let reference =
        with_dense_reference(|| fusion_batch::with_row_reference(|| loaded.draft(CAPACITY)))?
            .ok_or_else(|| Error::invariant("row reference draft missing"))?;
    assert_eq!(fusion_batch::captures(), before_reference);
    for program in [&candidate, &reference] {
        assert_eq!(program.prefill_width(), 128);
        assert!(!program.states().is_empty(), "draft F32 state coverage");
        assert!(program.fp8_states().is_empty(), "real draft KV is F32");
    }
    Ok((candidate, reference, packed_captures))
}

fn finite(values: &[f32], length: usize, label: &str) {
    assert_eq!(values.len(), length, "{label} length");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} nonfinite"
    );
}

fn bits(actual: &[f32], expected: &[f32], length: usize, label: &str) {
    finite(actual, length, label);
    finite(expected, length, label);
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label} bit drift"
    );
}

fn full_output(actual: &(Vec<f32>, Vec<f32>), expected: &(Vec<f32>, Vec<f32>)) {
    bits(&actual.0, &expected.0, HIDDEN, "draft hidden");
    bits(&actual.1, &expected.1, VOCABULARY, "draft full vocabulary");
}

fn greedy(logits: &[f32]) -> Result<u32> {
    finite(logits, VOCABULARY, "greedy full vocabulary");
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(device_error)
}

fn layout(program: &DeviceProgram, graph: &DataflowGraph) {
    let expected: BTreeSet<_> = graph
        .tensors
        .iter()
        .filter(|tensor| matches!(tensor.storage, TensorStorage::State { .. }))
        .map(|tensor| tensor.id)
        .collect();
    let actual: BTreeSet<_> = program
        .states()
        .keys()
        .chain(program.fp8_states().keys())
        .copied()
        .collect();
    assert!(!expected.is_empty(), "real model graph has no state IDs");
    assert!(!actual.is_empty(), "real program has no state IDs");
    assert_eq!(actual, expected, "all actual model state IDs");
    for tensors in program.states().values() {
        assert!(!tensors.is_empty(), "F32 state tensor set is empty");
        for tensor in tensors {
            assert!(tensor.size() > 0, "F32 state tensor is empty");
            assert!(tensor.shape().iter().all(|&dimension| dimension > 0));
        }
    }
    for (keys, values) in program.fp8_states().values() {
        assert!(
            keys.size() > 0 && values.size() > 0,
            "FP8 state tensor is empty"
        );
        assert_eq!(keys.shape(), values.shape(), "FP8 key/value shape");
    }
    assert!(
        program
            .states()
            .keys()
            .all(|id| !program.fp8_states().contains_key(id))
    );
}

fn compare_programs(
    loaded: &LoadedModel,
    actual: &DeviceProgram,
    expected: &DeviceProgram,
    graph: &DataflowGraph,
) -> Result<(usize, usize)> {
    layout(actual, graph);
    layout(expected, graph);
    assert_eq!(actual.position(), expected.position());
    assert_eq!(
        actual.states().keys().collect::<Vec<_>>(),
        expected.states().keys().collect::<Vec<_>>()
    );
    let mut dense_count = 0;
    let mut packed_count = 0;
    for (id, tensors) in actual.states() {
        let reference = &expected.states()[id];
        assert_eq!(tensors.len(), reference.len());
        for (a, b) in tensors.iter().zip(reference) {
            assert_eq!(a.shape(), b.shape(), "complete F32 state shape");
            assert_eq!(a.size(), b.size(), "complete F32 state size");
            let elements = a.size();
            assert!(elements > 0, "complete F32 state is empty");
            let a = loaded.device().read_borrowed(a)?;
            let b = loaded.device().read_borrowed(b)?;
            bits(&a, &b, elements, "complete independent F32 state");
            dense_count += elements;
        }
    }
    assert_eq!(
        actual.fp8_states().keys().collect::<Vec<_>>(),
        expected.fp8_states().keys().collect::<Vec<_>>()
    );
    for (id, (keys, values)) in actual.fp8_states() {
        let (ref_keys, ref_values) = &expected.fp8_states()[id];
        for (a, b) in [(keys, ref_keys), (values, ref_values)] {
            assert_eq!(a.shape(), b.shape());
            assert_eq!(a.size(), b.size());
            let a = loaded.device().read_borrowed(a)?;
            let b = loaded.device().read_borrowed(b)?;
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().chain(&b).all(|value| value.0 & 127 != 127),
                "complete independent FP8 state nonfinite"
            );
            assert!(
                a.iter().zip(&b).all(|(a, b)| a.0 == b.0),
                "complete independent FP8 state bit drift"
            );
            packed_count += a.len();
        }
    }
    Ok((dense_count, packed_count))
}

fn compare_drafts(loaded: &LoadedModel, timelines: &mut Timelines) -> Result<()> {
    let graph = loaded
        .draft_graph()
        .ok_or_else(|| Error::invariant("draft graph"))?;
    let (dense, packed) =
        compare_programs(loaded, &timelines.candidate, &timelines.reference, graph)?;
    timelines.coverage.dense_state += dense;
    timelines.coverage.packed_state += packed;
    Ok(())
}

fn compare_targets(loaded: &LoadedModel, timelines: &mut Timelines) -> Result<()> {
    let (dense, packed) = compare_programs(
        loaded,
        &timelines.target,
        &timelines.target_reference,
        loaded.graph(),
    )?;
    assert!(
        dense > 0 && packed > 0,
        "real target full F32 and FP8 state bits"
    );
    timelines.coverage.target_dense_finite += dense;
    timelines.coverage.target_packed_finite += packed;
    Ok(())
}

/// Exactly the executor's predecessor timeline: position zero falls back to its own target row.
fn externals(hidden: &[Vec<f32>], previous: &[f32]) -> Vec<f32> {
    hidden
        .iter()
        .enumerate()
        .flat_map(|(index, current)| {
            let prior = if index == 0 {
                previous
            } else {
                &hidden[index - 1]
            };
            if prior.len() == current.len() {
                prior
            } else {
                current
            }
        })
        .copied()
        .collect()
}

fn target_prompt(
    timelines: &mut Timelines,
    tokens: &[u32],
    position: usize,
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    let mut hidden = Vec::with_capacity(tokens.len());
    let mut logits = Vec::new();
    for (chunk, input) in tokens.chunks(64).enumerate() {
        let rows = timelines
            .target
            .prefill_batch(input, position + chunk * 64, true)?;
        let reference_rows =
            timelines
                .target_reference
                .prefill_batch(input, position + chunk * 64, true)?;
        assert_eq!(rows.len(), input.len());
        assert_eq!(reference_rows.len(), input.len());
        for (index, ((row_hidden, row_logits), reference)) in
            rows.into_iter().zip(reference_rows).enumerate()
        {
            bits(
                &row_hidden,
                &reference.0,
                HIDDEN,
                "independent target prompt hidden",
            );
            timelines.coverage.target_hidden += HIDDEN;
            if index + 1 == input.len() {
                bits(
                    &row_logits,
                    &reference.1,
                    VOCABULARY,
                    "independent target prompt full vocabulary",
                );
                timelines.coverage.target_logits += VOCABULARY;
            } else {
                assert!(
                    reference.1.is_empty(),
                    "reference nonfinal prompt vocabulary"
                );
            }
            finite(&row_hidden, HIDDEN, "target prompt hidden");
            if index + 1 == input.len() {
                finite(&row_logits, VOCABULARY, "target prompt last vocabulary");
                logits = row_logits;
            } else {
                assert!(
                    row_logits.is_empty(),
                    "only last prompt lane computes vocabulary"
                );
            }
            hidden.push(row_hidden);
        }
    }
    finite(&logits, VOCABULARY, "target prompt full vocabulary");
    Ok((hidden, logits))
}

fn append_prompt(
    loaded: &LoadedModel,
    timelines: &mut Timelines,
    tokens: &[u32],
    position: usize,
) -> Result<Vec<f32>> {
    let (hidden, logits) = target_prompt(timelines, tokens, position)?;
    for (chunk, input) in tokens.chunks(128).enumerate() {
        let start = chunk * 128;
        let rows = &hidden[start..start + input.len()];
        let external = externals(rows, &timelines.previous_target_hidden);
        let base = position + start;
        timelines.candidate.prime_batch(input, base, &external)?;
        let candidate = timelines
            .candidate
            .test_retained_prompt_hidden(base, input.len())?;
        timelines.reference.prime_batch(input, base, &external)?;
        let reference = timelines
            .reference
            .test_retained_prompt_hidden(base, input.len())?;
        assert_eq!(candidate.len(), input.len());
        assert_eq!(reference.len(), input.len());
        for (a, b) in candidate.iter().zip(&reference) {
            bits(a, b, HIDDEN, "same replay retained draft prompt hidden");
            timelines.coverage.hidden += a.len();
        }
        timelines.previous_target_hidden.clone_from(
            rows.last()
                .ok_or_else(|| Error::invariant("last target prompt hidden"))?,
        );
        assert_eq!(
            timelines.candidate.position(),
            (base + input.len()).saturating_sub(1)
        );
        compare_drafts(loaded, timelines)?;
    }
    assert_eq!(timelines.target.position(), position + tokens.len());
    compare_targets(loaded, timelines)?;
    Ok(logits)
}

fn reset_prompt(
    loaded: &LoadedModel,
    timelines: &mut Timelines,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    // Explicitly reset all independent timelines; position-zero replay also resets internally.
    timelines.target.reset()?;
    timelines.target_reference.reset()?;
    timelines.candidate.reset()?;
    timelines.reference.reset()?;
    timelines.previous_target_hidden.clear();
    append_prompt(loaded, timelines, tokens, 0)
}

fn proposal_chain(timelines: &mut Timelines, submitted: u32) -> Result<Vec<u32>> {
    let position = timelines.target.position();
    let mut chain = vec![submitted];
    let mut hidden = timelines.previous_target_hidden.clone();
    for offset in 0..2 {
        let a = timelines.candidate.step(
            chain[offset],
            position + offset,
            position + offset - 1,
            Some(&hidden),
            true,
        )?;
        let b = timelines.reference.step(
            chain[offset],
            position + offset,
            position + offset - 1,
            Some(&hidden),
            true,
        )?;
        full_output(&a, &b);
        assert_eq!(greedy(&a.1)?, greedy(&b.1)?);
        chain.push(greedy(&a.1)?);
        timelines.coverage.hidden += a.0.len();
        timelines.coverage.logits += a.1.len();
        hidden = a.0;
    }
    Ok(chain)
}

fn verify_commit(
    loaded: &LoadedModel,
    timelines: &mut Timelines,
    logits: &[f32],
    commit: usize,
) -> Result<Vec<f32>> {
    assert!((1..=3).contains(&commit));
    let position = timelines.target.position();
    assert_eq!(timelines.candidate.position(), position - 1);
    let chain = proposal_chain(timelines, greedy(logits)?)?;
    compare_drafts(loaded, timelines)?;
    let rows = timelines.target.step_batch(&chain, position)?;
    let reference_rows = timelines.target_reference.step_batch(&chain, position)?;
    assert_eq!(rows.len(), 3);
    assert_eq!(reference_rows.len(), 3);
    for (actual, reference) in rows.iter().zip(&reference_rows) {
        full_output(actual, reference);
        timelines.coverage.target_hidden += HIDDEN;
        timelines.coverage.target_logits += VOCABULARY;
    }
    // Check the complete pre-commit recurrent snapshots and all KV bytes as well.
    compare_targets(loaded, timelines)?;
    timelines.target.commit_batch(commit)?;
    timelines.target_reference.commit_batch(commit)?;
    // Match Speculation::replay: keep submitted-token KV, then replace accepted draft-conditioned
    // suffix entries with target-conditioned rows, even when both proposals are accepted.
    timelines.candidate.rewind_attention(position)?;
    timelines.reference.rewind_attention(position)?;
    for offset in 0..commit - 1 {
        let a = timelines.candidate.step(
            chain[offset + 1],
            position + offset + 1,
            position + offset,
            Some(&rows[offset].0),
            false,
        )?;
        let b = timelines.reference.step(
            chain[offset + 1],
            position + offset + 1,
            position + offset,
            Some(&rows[offset].0),
            false,
        )?;
        bits(
            &a.0,
            &b.0,
            HIDDEN,
            "accepted target-conditioned draft hidden",
        );
        timelines.coverage.hidden += a.0.len();
        assert!(
            a.1.is_empty() && b.1.is_empty(),
            "catchup does not compute vocabulary"
        );
    }
    timelines
        .previous_target_hidden
        .clone_from(&rows[commit - 1].0);
    assert_eq!(timelines.target.position(), position + commit);
    assert_eq!(
        timelines.candidate.position(),
        timelines.target.position() - 1
    );
    compare_drafts(loaded, timelines)?;
    compare_targets(loaded, timelines)?;
    Ok(rows[commit - 1].1.clone())
}

fn live_ranges<T: DType>(
    tensor: &Tensor<T>,
    graph: &DataflowGraph,
    id: TensorId,
    covered: usize,
) -> Result<Vec<Range<usize>>> {
    let spec = graph
        .tensors
        .iter()
        .find(|spec| spec.id == id)
        .ok_or_else(|| Error::invariant("state graph ID"))?;
    if !matches!(
        spec.storage,
        TensorStorage::State {
            kind: StateKind::AttentionKv,
            ..
        }
    ) {
        return Ok(once(0..tensor.size()).collect());
    }
    let shape = tensor.shape();
    assert_eq!(shape.len(), 3, "head-major KV shape");
    let heads = usize::try_from(shape[0]).map_err(device_error)?;
    let capacity = usize::try_from(shape[1]).map_err(device_error)?;
    let columns = usize::try_from(shape[2]).map_err(device_error)?;
    assert!(covered <= capacity);
    Ok((0..heads)
        .map(|head| {
            let start = head * capacity * columns;
            start..start + covered * columns
        })
        .collect())
}

/// For prefix-replay equality only: hash every recurrent value and every valid head-major KV row.
/// Stale KV beyond the logical cursor is deliberately not a prefix snapshot.
fn state_digest(
    loaded: &LoadedModel,
    program: &DeviceProgram,
    graph: &DataflowGraph,
) -> Result<StateDigest> {
    layout(program, graph);
    let mut result = BTreeMap::new();
    for (id, tensors) in program.states() {
        let mut hashes = Vec::new();
        for tensor in tensors {
            let values = loaded.device().read_borrowed(tensor)?;
            finite(&values, tensor.size(), "prefix F32 state");
            let mut hash = Sha256::new();
            for range in live_ranges(tensor, graph, *id, program.position())? {
                for value in &values[range] {
                    hash.update(value.to_bits().to_le_bytes());
                }
            }
            hashes.push(hash.finalize().into());
        }
        result.insert(*id, hashes);
    }
    for (id, (keys, values)) in program.fp8_states() {
        let mut hashes = Vec::new();
        for tensor in [keys, values] {
            let values = loaded.device().read_borrowed(tensor)?;
            assert!(
                values.iter().all(|value| value.0 & 127 != 127),
                "prefix FP8 state finite"
            );
            let mut hash = Sha256::new();
            for range in live_ranges(tensor, graph, *id, program.position())? {
                for value in &values[range] {
                    hash.update([value.0]);
                }
            }
            hashes.push(hash.finalize().into());
        }
        assert!(result.insert(*id, hashes).is_none());
    }
    Ok(result)
}

struct FreshPrefix {
    target: StateDigest,
    draft: StateDigest,
    logits: Vec<f32>,
    hidden: Vec<f32>,
}

impl FreshPrefix {
    fn capture(loaded: &LoadedModel, timelines: &Timelines, logits: Vec<f32>) -> Result<Self> {
        Ok(Self {
            target: state_digest(loaded, &timelines.target, loaded.graph())?,
            draft: state_digest(
                loaded,
                &timelines.candidate,
                loaded
                    .draft_graph()
                    .ok_or_else(|| Error::invariant("draft graph"))?,
            )?,
            logits,
            hidden: timelines.previous_target_hidden.clone(),
        })
    }
    fn compare(
        &self,
        loaded: &LoadedModel,
        timelines: &mut Timelines,
        logits: &[f32],
    ) -> Result<()> {
        bits(
            logits,
            &self.logits,
            VOCABULARY,
            "fresh versus resumed target vocabulary",
        );
        bits(
            &timelines.previous_target_hidden,
            &self.hidden,
            HIDDEN,
            "joint prefix last hidden",
        );
        assert_eq!(
            self.target,
            state_digest(loaded, &timelines.target, loaded.graph())?
        );
        assert_eq!(
            self.draft,
            state_digest(
                loaded,
                &timelines.candidate,
                loaded
                    .draft_graph()
                    .ok_or_else(|| Error::invariant("draft graph"))?
            )?
        );
        compare_targets(loaded, timelines)?;
        compare_drafts(loaded, timelines)
    }
}

struct JointPrefix {
    target: ProgramSnapshot,
    target_reference: ProgramSnapshot,
    candidate: DraftSnapshot,
    reference: DraftSnapshot,
    covered: usize,
}

impl JointPrefix {
    fn capture(loaded: &LoadedModel, timelines: &Timelines, covered: usize) -> Result<Self> {
        assert_eq!(timelines.target.position(), covered);
        assert_eq!(timelines.candidate.position(), covered - 1);
        let graph = loaded
            .draft_graph()
            .ok_or_else(|| Error::invariant("draft graph"))?;
        Ok(Self {
            target: ProgramSnapshot::capture(
                loaded.device(),
                &timelines.target,
                loaded.graph(),
                covered,
            )?,
            target_reference: ProgramSnapshot::capture(
                loaded.device(),
                &timelines.target_reference,
                loaded.graph(),
                covered,
            )?,
            candidate: DraftSnapshot {
                program: ProgramSnapshot::capture(
                    loaded.device(),
                    &timelines.candidate,
                    graph,
                    covered - 1,
                )?,
                hidden: timelines.previous_target_hidden.clone(),
            },
            reference: DraftSnapshot {
                program: ProgramSnapshot::capture(
                    loaded.device(),
                    &timelines.reference,
                    graph,
                    covered - 1,
                )?,
                hidden: timelines.previous_target_hidden.clone(),
            },
            covered,
        })
    }
    fn restore(&self, loaded: &LoadedModel, timelines: &mut Timelines) -> Result<()> {
        self.target
            .restore(loaded.device(), &mut timelines.target, self.covered)?;
        self.target_reference.restore(
            loaded.device(),
            &mut timelines.target_reference,
            self.covered,
        )?;
        compare_targets(loaded, timelines)?;
        self.candidate.program.restore(
            loaded.device(),
            &mut timelines.candidate,
            self.covered - 1,
        )?;
        self.reference.program.restore(
            loaded.device(),
            &mut timelines.reference,
            self.covered - 1,
        )?;
        bits(
            &self.candidate.hidden,
            &self.reference.hidden,
            HIDDEN,
            "snapshot draft last hidden",
        );
        timelines
            .previous_target_hidden
            .clone_from(&self.candidate.hidden);
        compare_drafts(loaded, timelines)
    }
}

fn prefix_resume(
    loaded: &LoadedModel,
    timelines: &mut Timelines,
    tokens: &[u32],
    fresh: &FreshPrefix,
) -> Result<()> {
    reset_prompt(loaded, timelines, &tokens[..128])?;
    let snapshot = JointPrefix::capture(loaded, timelines, 128)?;
    let logits = append_prompt(loaded, timelines, &tokens[128..255], 128)?;
    fresh.compare(loaded, timelines, &logits)?;
    append_prompt(loaded, timelines, &tokens[255..511], 255)?;
    snapshot.restore(loaded, timelines)?;
    let logits = append_prompt(loaded, timelines, &tokens[128..255], 128)?;
    fresh.compare(loaded, timelines, &logits)?;
    verify_commit(loaded, timelines, &logits, 2)?;
    Ok(())
}

fn fixture() -> Result<(String, Vec<u32>)> {
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let token_file = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(token_file).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    if tokens.len() != 511 {
        return Err(Error::invalid("frozen 511-token MTP2 prompt required"));
    }
    Ok((path, tokens))
}

#[test]
#[ignore = "real MTP2 checkpoint, CUDA, frozen table/tokens and protected serial runner required"]
fn default_fc128_preserves_real_mtp2_model_bits() -> Result<()> {
    let (path, tokens) = fixture()?;
    let table_path =
        tuning_table_path().ok_or_else(|| Error::invalid("frozen tuning table required"))?;
    let table = read(&table_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            mtp_depth: 2,
            autotune: false,
            chunked_recurrent: false,
            ..LoadOptions::default()
        },
    )?;
    assert_eq!(loaded.mtp_depth(), 2);
    assert_eq!(loaded.model().hidden_size, HIDDEN);
    assert_eq!(loaded.model().vocab_size, VOCABULARY);
    let mut timelines = Timelines::new(&loaded)?;
    let mut fresh = None;
    for length in PROMPTS {
        let mut logits = reset_prompt(&loaded, &mut timelines, &tokens[..length])?;
        if length == 255 {
            fresh = Some(FreshPrefix::capture(&loaded, &timelines, logits.clone())?);
        }
        for commit in [1, 2, 3] {
            logits = verify_commit(&loaded, &mut timelines, &logits, commit)?;
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "MTP_FC_MODEL_CASE": true, "MTP": 2, "prompt": length,
                "target_width": 64, "draft_width": 128, "capacity": CAPACITY,
                "same_replay_prompt_hidden_bits": true, "subsequent_full_vocabulary_bits": true,
                "full_actual_draft_state_bits": true, "target_state_finite": true,
                "full_actual_target_state_bits": true,
                "target_full_hidden_and_vocabulary_bits": true,
                "target_independent_arithmetic_reference": true, "verify_commits": [1,2,3],
                "candidate_scratch_bytes": timelines.candidate.test_fusion_batch_scratch_bytes(),
                "reference_scratch_bytes": timelines.reference.test_fusion_batch_scratch_bytes(),
            })
        );
    }
    prefix_resume(
        &loaded,
        &mut timelines,
        &tokens,
        &fresh.ok_or_else(|| Error::invariant("fresh 255 baseline"))?,
    )?;
    assert_eq!(
        table,
        read(table_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    let checked = &timelines.coverage;
    assert!(checked.hidden > 0 && checked.logits > 0);
    assert!(
        checked.dense_state > 0,
        "draft F32 full-state bits were not checked"
    );
    assert_eq!(checked.packed_state, 0, "this real draft has no FP8 states");
    assert!(checked.target_dense_finite > 0 && checked.target_packed_finite > 0);
    assert!(checked.target_hidden > 0 && checked.target_logits > 0);
    eprintln!(
        "MTP_FUSION_MODEL_CHECK {}",
        serde_json::json!({
            "prompts": PROMPTS, "candidate_packed_captures": timelines.packed_captures,
            "reference_packed_captures": 0, "bit_failures": 0,
            "hidden_elements": checked.hidden, "logits_elements": checked.logits,
            "f32_state_elements": checked.dense_state,
            "fp8_state_elements": checked.packed_state,
            "draft_kv_storage": "f32", "target_kv_storage": "fp8",
            "target_f32_finite_elements": checked.target_dense_finite,
            "target_hidden_elements": checked.target_hidden,
            "target_logits_elements": checked.target_logits,
            "target_f32_state_bit_elements": checked.target_dense_finite,
            "target_fp8_state_bit_elements": checked.target_packed_finite,
            "independent_target_count": 2,
            "candidate_narrow_dense_captures": timelines.narrow_captures,
            "reference_narrow_dense_captures": 0,
            "target_fp8_finite_elements": checked.target_packed_finite,
            "partial_commit_widths": [1, 2, 3], "actual_draft_depth": loaded.mtp_depth(),
            "capacity": CAPACITY, "target_width": 64, "draft_width": 128,
            "candidate_scratch_bytes": timelines.candidate.test_fusion_batch_scratch_bytes(),
            "reference_scratch_bytes": timelines.reference.test_fusion_batch_scratch_bytes(),
            "joint_prefix_restored": true, "joint_prefix_covered": 128, "draft_prefix_covered": 127,
            "fresh_vs_resumed_valid_state_sha256": true,
            "target_independent_arithmetic_reference": true,
            "element_counts": "checked visits including repeated replays",
            "quality_accepted": false, "performance_accepted": false,
        })
    );
    Ok(())
}

#[test]
fn predecessor_rows_match_position_zero_and_resumed_priming() {
    let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    assert_eq!(externals(&rows, &[]), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        externals(&rows, &[7.0, 8.0]),
        vec![7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
    );
}
