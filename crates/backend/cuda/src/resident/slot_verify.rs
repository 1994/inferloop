//! Cross-sequence target verification with per-slot recurrent rollback.
use super::batch::gdn_ba_pairs;
use super::prefill_projection::MlpPlan;
use super::prefill_projection::{PairContext, record_pair};
use super::{
    ActivationArena,
    batch::{BatchBuilder, BatchReadout, Dispatch32, States, dispatch32, take_result},
    capture::{Capture, CaptureMode, error},
    fp8_cache::Fp8Caches,
    profile,
    slot_commit::CapturedCommit,
};
use crate::constants::CB_DECODE_SLOTS;
use crate::constants::F32_BYTES;
use crate::constants::INACTIVE_LANE_STATE_POSITION;
use crate::constants::MAX_VERIFICATION_WIDTH;
use crate::constants::METADATA_FIELDS;
use crate::constants::MIB;
use crate::device::DeviceProfile;
use crate::device::PinnedUpload;
use crate::device::Readbacks;
use crate::device::{CudaDevice, ReadbackSource, device_error};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{StateKind, TensorNode, TensorOp, TensorStorage};
use std::result::Result as StdResult;
use std::slice::from_ref;

/// Pooled speculation graph: `slots` state sets × `verify` contiguous candidate lanes each,
/// flattened into one batched arena so every weight still loads once per replay. Lane
/// `slot * verify + position` verifies position `cursors[slot] + position` of its slot's own
/// state set — the vLLM-style flattening of (sequence × draft depth) into the batch axis.
///
/// Conv/delta snapshots are recorded after every lane but each slot's last, so a partial
/// acceptance rolls one slot back to its longest accepted prefix without touching the other
/// slots; KV rollback is the cursor rewind, exactly like the single-sequence verify graph.
pub(super) struct SlotVerifyGraph {
    graph: CudaGraph<()>,
    /// `restore[slot][accepted]` reinstates the conv/delta snapshot taken after that many
    /// candidate lanes of the slot committed; a fully accepted chain never restores. With
    /// [`Self::fused`] it instead applies the `accepted + 1` committed lanes to the untouched
    /// slot state, and runs on every commit.
    restore: Vec<Vec<CudaGraph<()>>>,
    batched_commit: Option<CapturedCommit>,
    /// Recurrent nodes verify without writing state (`slot_recurrent`), so commits re-apply
    /// the accepted prefix instead of restoring snapshots.
    fused: bool,
    readbacks: Readbacks,
    token_readout: Option<super::token_readout::TokenReadout>,
    metadata_owner: Tensor<i32>,
    metadata_upload: Vec<i32>,
    metadata_staging: PinnedUpload<i32>,
    hidden: Arc<Tensor<f32>>,
    logits: Arc<Tensor<f32>>,
    /// Slot state sets the graph interleaves.
    pub slots: usize,
    /// Candidate lanes each slot runs per replay (submitted token plus draft proposals).
    pub verify: usize,
    /// Host-side state position each slot's next replay continues at.
    cursors: Vec<usize>,
    pending: Vec<usize>,
    /// Optional in-graph boundary events (`INFER_CUDA_PREFILL_PROFILE`); declared after
    /// the graph so its executable is destroyed before the events it records.
    profile: Option<profile::SlotProfile>,
    /// Commit inputs the verify graph copies out and the commit graphs read.
    _recurrent: RecurrentCapture,
    // The shared batched arena anchors the graph's activations; declared last so the graph
    // and the restore graphs drop before it.
    _arena: ActivationArena,
}

/// Captured verification and commits retain all recurrent scratch and staged inputs.
struct RecurrentCapture {
    staging: super::slot_recurrent::Staging,
    parameters: super::slot_states::RecurrentParameters,
}
impl RecurrentCapture {
    fn new(
        builder: &BatchBuilder<'_>,
        arena: &ActivationArena,
        slots: usize,
        fused: bool,
    ) -> Result<Self> {
        let staging = if fused {
            super::slot_recurrent::allocate_staging(
                builder.device,
                &builder.graph.nodes,
                arena,
                slots,
            )?
        } else {
            super::slot_recurrent::Staging::new()
        };
        let parameters = if fused {
            super::slot_states::prepare_parameters(
                builder.device,
                &builder.graph.nodes,
                builder.weights,
            )?
        } else {
            super::slot_states::RecurrentParameters::new()
        };
        Ok(Self {
            staging,
            parameters,
        })
    }
}

/// One active lane of a slot verify replay: slot, candidate position inside the slot's
/// chain, token, `RoPE` position and state position.
pub(super) type SlotVerifyLane = (usize, usize, u32, usize, usize);

/// One slot's verification lanes of the pooled arena with their metadata.
fn slot_lanes(
    slot: usize,
    metadata: &[Tensor<i32>],
) -> StdResult<super::slot_recurrent::SlotLanes<'_>, DeviceError> {
    let base = slot * super::slot_recurrent::LANES;
    let [first, second, third] = [base, base + 1, base + 2];
    let lane = |index: usize| {
        metadata
            .get(index)
            .ok_or_else(|| error("slot verify lane metadata"))
    };
    Ok(super::slot_recurrent::SlotLanes {
        lanes: [first, second, third],
        metadata: [lane(first)?, lane(second)?, lane(third)?],
    })
}

fn validate_commits(pending: &[usize], commits: &[(usize, usize)]) -> Result<()> {
    let mut seen = vec![false; pending.len()];
    for &(slot, accepted) in commits {
        let Some(&count) = pending.get(slot) else {
            return Err(Error::invalid("slot verify commit"));
        };
        if accepted >= count || seen[slot] {
            return Err(Error::invalid("slot verify commit count or duplicate"));
        }
        seen[slot] = true;
    }
    Ok(())
}

/// Whether every recurrent writer uses snapshot-free pooled verification.
pub(super) fn fused_recurrent(graph: &infer_ir::DataflowGraph, verify: usize) -> bool {
    verify == super::slot_recurrent::LANES
        && graph.nodes.iter().all(|node| {
            !node.states.iter().any(|id| {
                graph.tensors.iter().any(|spec| {
                    spec.id == *id
                        && matches!(
                            spec.storage,
                            TensorStorage::State {
                                kind: StateKind::Conv | StateKind::LinearAttention,
                                ..
                            }
                        )
                })
            }) || super::slot_recurrent::covers(node)
        })
}

impl BatchBuilder<'_> {
    /// Per slot, one graph per committed lane count that applies the accepted prefix of the
    /// snapshot-free recurrent verification to the slot's state.
    fn capture_commits(
        &self,
        lane_states: &mut [States],
        arena: &ActivationArena,
        metadata: &[Tensor<i32>],
        (verify, staging): (usize, &super::slot_recurrent::Staging),
    ) -> Result<Vec<Vec<CudaGraph<()>>>> {
        let mut commits = Vec::with_capacity(lane_states.len());
        let value_split =
            super::slot_recurrent::value_split(self.device.profile()?.multiprocessors);
        for (slot, states) in lane_states.iter_mut().enumerate() {
            let lanes = slot_lanes(slot, metadata).map_err(device_error)?;
            let mut graphs = Vec::with_capacity(verify);
            for count in 1..=verify {
                graphs.push(
                    CudaGraph::scope(&self.device.stream, |scope| {
                        for (index, node) in self.graph.nodes.iter().enumerate() {
                            if super::slot_recurrent::covers(node) {
                                let staged = staging
                                    .get(&(slot, index))
                                    .ok_or_else(|| error("slot commit staging"))?;
                                super::slot_recurrent::record_commit(
                                    scope,
                                    node,
                                    arena,
                                    states,
                                    &lanes,
                                    self.weights,
                                    value_split,
                                    (staged, count),
                                )?;
                            }
                        }
                        Ok(())
                    })
                    .map_err(device_error)?,
                );
            }
            commits.push(graphs);
        }
        Ok(commits)
    }

    /// Per slot, one graph per snapshot that reinstates it in the slot's state.
    fn capture_restores(
        &self,
        lane_states: &mut [States],
        checkpoints: &[Vec<States>],
    ) -> Result<Vec<Vec<CudaGraph<()>>>> {
        let mut restore = Vec::with_capacity(checkpoints.len());
        for (states, per_slot) in lane_states.iter_mut().zip(checkpoints) {
            let mut graphs = Vec::with_capacity(per_slot.len());
            for snapshot in per_slot {
                graphs.push(
                    CudaGraph::scope(&self.device.stream, |scope| {
                        for (id, tensors) in snapshot {
                            for (output, input) in states
                                .get_mut(id)
                                .ok_or_else(|| error("slot verify restore state"))?
                                .iter_mut()
                                .zip(tensors)
                            {
                                scope.record(api::memcpy(output, input))?;
                            }
                        }
                        Ok(())
                    })
                    .map_err(device_error)?,
                );
            }
            restore.push(graphs);
        }
        Ok(restore)
    }

    fn slot_checkpoints(&self, lane_states: &[States], verify: usize) -> Result<Vec<Vec<States>>> {
        let budget = usize::try_from(self.device.profile()?.checkpoint_budget_bytes())
            .map_err(device_error)?;
        let mut checkpoints: Vec<Vec<States>> = Vec::with_capacity(lane_states.len());
        let mut bytes = 0usize;
        for slot_states in lane_states {
            let mut per_slot = Vec::with_capacity(verify - 1);
            for _ in 0..verify - 1 {
                let mut snapshot = States::new();
                for spec in &self.graph.tensors {
                    if !matches!(
                        spec.storage,
                        TensorStorage::State {
                            kind: StateKind::Conv | StateKind::LinearAttention,
                            ..
                        }
                    ) {
                        continue;
                    }
                    let mut copies = Vec::new();
                    for source in slot_states
                        .get(&spec.id)
                        .ok_or_else(|| Error::invariant("slot verify checkpoint source"))?
                    {
                        bytes = source
                            .size()
                            .checked_mul(F32_BYTES)
                            .and_then(|n| bytes.checked_add(n))
                            .ok_or_else(|| Error::invalid("slot verify checkpoint overflow"))?;
                        if bytes > budget {
                            return Err(Error::new(
                                infer_core::ErrorCode::Capacity,
                                format!(
                                    "slot verify checkpoints exceed budget: need {} MiB, \
                                     budget {} MiB at {} slots x {} lanes",
                                    bytes / MIB,
                                    budget / MIB,
                                    lane_states.len(),
                                    verify
                                ),
                            ));
                        }
                        let shape = source
                            .shape()
                            .iter()
                            .map(|v| usize::try_from(*v).map_err(device_error))
                            .collect::<Result<Vec<_>>>()?;
                        copies.push(
                            api::zeros::<f32>(&shape)
                                .sync_on(&self.device.stream)
                                .map_err(device_error)?,
                        );
                    }
                    snapshot.insert(spec.id, copies);
                }
                per_slot.push(snapshot);
            }
            checkpoints.push(per_slot);
        }
        Ok(checkpoints)
    }

    /// Bind one recurrent node to each slot's independent state and commit staging.
    fn record_slot_recurrent(
        &self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        (states, metadata): (&[States], &[Tensor<i32>]),
        (staging, index, value_split, batch): (
            &mut super::slot_recurrent::Staging,
            usize,
            usize,
            &super::slot_states::VerifyBatch<'_>,
        ),
    ) -> StdResult<(), DeviceError> {
        staging.record_node(scope, node, arena, index)?;
        if super::slot_convolution::record(scope, node, arena, batch, self.weights)?
            || super::slot_states::record_batch(scope, node, arena, batch, value_split)?
        {
            return Ok(());
        }
        for (slot, states) in states.iter().enumerate() {
            let lanes = slot_lanes(slot, metadata)?;
            super::slot_recurrent::record_verify(
                scope,
                node,
                arena,
                states,
                &lanes,
                self.weights,
                value_split,
            )?;
        }
        Ok(())
    }

    fn record_slot_verify_rows(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        (lane_states, lane_fp8): (&mut [States], &mut [Fp8Caches]),
        (verify, metadata): (usize, &[Tensor<i32>]),
        checkpoints: &mut [Vec<States>],
    ) -> StdResult<(), DeviceError> {
        for (lane, lane_metadata) in metadata.iter().enumerate() {
            let slot = lane / verify;
            let position = lane % verify;
            let mut capture = Capture {
                scope,
                arena,
                weights: self.weights,
                nvfp4: self.nvfp4,
                attention: self.attention,
                states: &mut lane_states[slot],
                fp8_states: &mut lane_fp8[slot],
                metadata: lane_metadata,
                // The candidate lanes of one slot share that slot's cache.
                table: &self.tables[slot],
                external: &self.lane_external[lane],
                capacity: self.capacity,
                fusion: &mut self.lane_fusion[lane],
                mode: CaptureMode::Row(lane),
            };
            // The state's writer saves its computed value before the next
            // candidate can modify it. Inactive lanes skip both stores.
            // The last candidate has no rollback snapshot.
            if position + 1 < verify
                && let Some(id) = node.states.first()
                && let Some(snapshot) = checkpoints
                    .get_mut(slot)
                    .and_then(|per_slot| per_slot.get_mut(position))
                    .and_then(|snapshot| snapshot.get_mut(id))
            {
                capture.record_row_checkpoint(node, lane, snapshot)?;
            } else {
                capture.record_row(node, lane)?;
            }
        }
        Ok(())
    }

    fn record_slot_pair(
        &mut self,
        scope: &Scope,
        (first, second): (usize, usize),
        arena: &mut ActivationArena,
        (device_profile, profile): (&DeviceProfile, &mut Option<profile::SlotProfile>),
    ) -> StdResult<bool, DeviceError> {
        let recorded = record_pair(
            scope,
            &self.graph.nodes[first],
            &self.graph.nodes[second],
            arena,
            (self.weights, device_profile),
            PairContext::Verify,
            self.verify_pairs.as_deref_mut(),
        )?;
        if recorded {
            if let Some(profile) = profile.as_mut() {
                profile
                    .graph()
                    .boundary(scope, second, &self.graph.nodes[second])?;
            }
            self.nvfp4.clear_quant_reuse();
        }
        Ok(recorded)
    }

    fn record_slot_mlp(
        &mut self,
        scope: &Scope,
        index: usize,
        arena: &mut ActivationArena,
        plan: &MlpPlan,
    ) -> StdResult<bool, DeviceError> {
        if !plan.record(
            scope,
            (index, &self.graph.nodes),
            arena,
            self.weights,
            self.nvfp4,
        )? {
            return Ok(false);
        }
        self.nvfp4.clear_quant_reuse();
        Ok(true)
    }

    fn capture_slot_verify(
        &mut self,
        lane_states: &mut [States],
        lane_fp8: &mut [Fp8Caches],
        geometry: (usize, &[Tensor<i32>], &super::slot_states::VerifyBatch<'_>),
        arena: &mut ActivationArena,
        (checkpoints, staging): (&mut [Vec<States>], &mut super::slot_recurrent::Staging),
        profile: &mut Option<profile::SlotProfile>,
    ) -> Result<CudaGraph<()>> {
        let (verify, metadata, batch) = geometry;
        let fused = fused_recurrent(self.graph, verify);
        let device = self.device;
        let device_profile = device.profile()?;
        let value_split = super::slot_recurrent::value_split(device_profile.multiprocessors);
        let pairs = gdn_ba_pairs(self.graph, self.weights);
        let mlp = MlpPlan::for_verify(self.graph, self.weights, (arena.lanes(), self.nvfp4));
        CudaGraph::scope(&device.stream, |scope| {
            let mut no_fusion = None;
            let mut fused_second = None;
            for (index, node) in self.graph.nodes.iter().enumerate() {
                if fused_second == Some(index) || mlp.absorbs(index) {
                    continue;
                }
                if let Some(profile) = profile.as_mut() {
                    profile.graph().boundary(scope, index, node)?;
                }
                if self.record_slot_mlp(scope, index, arena, &mlp)? {
                    continue;
                }
                if let Some(&second) = pairs.get(&index)
                    && self.record_slot_pair(
                        scope,
                        (index, second),
                        arena,
                        (device_profile, &mut *profile),
                    )?
                {
                    fused_second = Some(second);
                    continue;
                }
                if node.op == TensorOp::Linear {
                    super::batch_projection::record_slots(
                        scope,
                        node,
                        arena,
                        self.weights,
                        self.nvfp4,
                    )?;
                    continue;
                }
                if super::slot_rope::record(scope, node, arena, self.weights, batch.metadata)? {
                    continue;
                }
                if self.record_slot_verify_attention(
                    scope,
                    node,
                    arena,
                    (lane_states, lane_fp8),
                    (verify, metadata, batch.metadata),
                )? {
                    continue;
                }
                match dispatch32(node, self.weights) {
                    Dispatch32::Linear => return Err(error("slot verify linear dispatch")),
                    Dispatch32::Batched => Capture {
                        scope,
                        arena,
                        weights: self.weights,
                        nvfp4: self.nvfp4,
                        attention: self.attention,
                        states: self.states,
                        fp8_states: self.fp8_states,
                        metadata: &metadata[0],
                        table: &self.tables[0],
                        external: self.external,
                        capacity: self.capacity,
                        fusion: &mut no_fusion,
                        mode: CaptureMode::Batched,
                    }
                    .record(node)?,
                    Dispatch32::Row if fused && super::slot_recurrent::covers(node) => {
                        self.record_slot_recurrent(
                            scope,
                            node,
                            arena,
                            (lane_states, metadata),
                            (staging, index, value_split, batch),
                        )?;
                    }
                    Dispatch32::Row => self.record_slot_verify_rows(
                        scope,
                        node,
                        arena,
                        (lane_states, lane_fp8),
                        (verify, metadata),
                        checkpoints,
                    )?,
                }
            }
            if let Some(profile) = profile.as_mut() {
                profile.graph().finish(scope)?;
            }
            Ok(())
        })
        .map_err(device_error)
    }

    /// Capture the pooled speculation graph over per-slot state sets. Lane rows of the
    /// shared arena feed the row-grouped fused GEMV directly; pure activation nodes record
    /// once across every lane; state, metadata and checkpoint copies record per lane
    /// against `lane_states[lane / verify]`.
    /// # Errors
    /// Rejects mismatched lane slices, a slot count outside `2..=CB_DECODE_SLOTS`, a verify
    /// width outside `2..=MAX_VERIFICATION_WIDTH`, an exceeded checkpoint budget or CUDA
    /// capture failures.
    pub fn build_slot_verify(
        &mut self,
        lane_states: &mut super::slot_states::SlotStates,
        lane_fp8: &mut [Fp8Caches],
        verify: usize,
    ) -> Result<SlotVerifyGraph> {
        let slots = lane_states.len();
        let lanes = slots
            .checked_mul(verify)
            .ok_or_else(|| Error::invalid("slot verify width overflow"))?;
        if !(2..=CB_DECODE_SLOTS).contains(&slots)
            || !(2..=MAX_VERIFICATION_WIDTH).contains(&verify)
            || lane_fp8.len() != slots
            || self.lane_external.len() < lanes
            || self.lane_fusion.len() < lanes
        {
            return Err(Error::invalid("slot verify lanes"));
        }
        let (lane_states, roots) = lane_states.parts();
        let fused = fused_recurrent(self.graph, verify);
        let mut checkpoints = if fused {
            (0..slots).map(|_| Vec::new()).collect()
        } else {
            self.slot_checkpoints(lane_states, verify)?
        };
        let mut arena = ActivationArena::new_batched(
            self.device,
            self.graph,
            lanes,
            usize::try_from(self.device.profile()?.arena_budget_bytes()).map_err(device_error)?,
        )?;
        let (metadata_owner, metadata) = super::batch::batched_metadata(self.device, lanes)?;
        let mut profile = profile::SlotProfile::from_env(self.device);
        let mut recurrent = RecurrentCapture::new(self, &arena, slots, fused)?;
        let batch = super::slot_states::VerifyBatch {
            roots,
            parameters: &recurrent.parameters,
            metadata: &metadata_owner,
        };
        let graph = self.capture_slot_verify(
            lane_states,
            lane_fp8,
            (verify, &metadata, &batch),
            &mut arena,
            (&mut checkpoints, &mut recurrent.staging),
            &mut profile,
        )?;
        let batched_commit = if fused
            && CapturedCommit::supported(&self.graph.nodes, &recurrent.parameters, roots, slots)
        {
            Some(CapturedCommit::new(
                self.device,
                &self.graph.nodes,
                roots,
                (&recurrent.staging, &recurrent.parameters, &metadata_owner),
                slots,
            )?)
        } else {
            None
        };
        let restore = if fused {
            self.capture_commits(lane_states, &arena, &metadata, (verify, &recurrent.staging))?
        } else {
            self.capture_restores(lane_states, &checkpoints)?
        };
        let hidden = take_result(&mut arena, self.graph.hidden)?;
        let logits = take_result(&mut arena, self.graph.logits)?;
        let token_readout = super::token_readout::TokenReadout::new(
            self.device,
            from_ref(&logits),
            slots * verify,
            logits.size() / (slots * verify),
            self.weights,
        )?;
        Ok(SlotVerifyGraph {
            token_readout,
            graph,
            restore,
            batched_commit,
            fused,
            readbacks: Readbacks::contiguous(self.device, &hidden, &logits, lanes)?,
            metadata_owner,
            metadata_upload: Vec::with_capacity(lanes * METADATA_FIELDS),
            metadata_staging: PinnedUpload::single(self.device, lanes * METADATA_FIELDS)?,
            hidden,
            logits,
            slots,
            verify,
            cursors: vec![0; slots],
            pending: vec![0; slots],
            profile,
            _recurrent: recurrent,
            _arena: arena,
        })
    }
}

impl SlotVerifyGraph {
    /// Replay one speculation step. Every slot runs all `verify` lanes: active lanes carry
    /// the submitted token plus its draft candidates, inactive slots mask their lanes with
    /// a `-1` state position. Returns one `(hidden, logits)` row per lane in lane order.
    /// # Errors
    /// Rejects unknown slots, duplicate lanes, out-of-vocabulary tokens, cursors without
    /// room for the whole chain, positions out of capacity and CUDA replay failures.
    pub(crate) async fn run_verify_readout_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotVerifyLane],
        capacity: usize,
        vocabulary: usize,
        plain_greedy: bool,
    ) -> Result<BatchReadout> {
        let device_tokens = plain_greedy && self.token_readout.is_some();
        if self.pending.iter().any(|&count| count != 0) {
            return Err(Error::invalid("uncommitted slot verification"));
        }
        let per_lane = verify_lanes(lanes, &self.cursors, self.verify, capacity, vocabulary)?;
        let total = self.slots * self.verify;
        self.metadata_upload.clear();
        for entry in per_lane.iter().take(total) {
            let values = match *entry {
                Some((token, rope_pos, state_pos)) => [
                    i32::try_from(rope_pos).map_err(device_error)?,
                    i32::try_from(token).map_err(device_error)?,
                    i32::try_from(state_pos).map_err(device_error)?,
                    0,
                ],
                None => [0, 0, INACTIVE_LANE_STATE_POSITION, 0],
            };
            self.metadata_upload.extend_from_slice(&values);
        }
        self.metadata_staging
            .copy(device, &mut self.metadata_owner, &self.metadata_upload)?;
        let hidden_size = self.hidden.size() / total;
        let vocabulary = self.logits.size() / total;
        let mut sources = Vec::with_capacity(2 * total);
        for (lane, entry) in per_lane.iter().enumerate() {
            let active = entry.is_some();
            sources.push(active.then(|| ReadbackSource {
                tensor: Arc::clone(&self.hidden),
                skip: lane * hidden_size,
                len: hidden_size,
            }));
            sources.push((active && !device_tokens).then(|| ReadbackSource {
                tensor: Arc::clone(&self.logits),
                skip: lane * vocabulary,
                len: vocabulary,
            }));
        }
        let (output, tokens) = if device_tokens {
            let active: Vec<_> = per_lane
                .iter()
                .enumerate()
                .filter_map(|(row, entry)| entry.as_ref().map(|_| row))
                .collect();
            let (rows, selected) = self
                .token_readout
                .as_mut()
                .ok_or_else(|| Error::invariant("slot verify token readout"))?
                .run_async(device, &self.graph, &mut self.readbacks, &sources, &active)
                .await?;
            let mut tokens = vec![None; total];
            for (&row, token) in active.iter().zip(selected) {
                tokens[row] = Some(token);
            }
            (Some(rows), Some(tokens))
        } else {
            (None, None)
        };
        let rows = if output.is_none() {
            self.readbacks
                .run_async(device, &self.graph, &sources)
                .await?
        } else {
            Vec::new()
        };
        if let Some(profile) = &self.profile
            && let Some(&(.., rope_pos, _)) = lanes.first()
        {
            profile.report("slot_verify", lanes.len(), rope_pos);
        }
        for (slot, group) in per_lane.chunks_exact(self.verify).enumerate() {
            self.pending[slot] = group.iter().flatten().count();
        }
        if let Some(rows) = output {
            return Ok(BatchReadout { rows, tokens });
        }
        let mut rows = rows.into_iter();
        (0..total)
            .map(|_| {
                Ok((
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot verify hidden readback"))?,
                    rows.next()
                        .ok_or_else(|| Error::invariant("slot verify logits readback"))?,
                ))
            })
            .collect::<Result<_>>()
            .map(BatchReadout::host)
    }

    /// Commit `accepted` candidate lanes of `slot` (not counting the submitted token, which
    /// always commits): a partial chain restores the conv/delta snapshot taken after the
    /// last accepted lane; a full chain keeps the live state. The cursor advances past the
    /// submitted token plus the accepted candidates.
    /// # Errors
    /// Rejects unknown slots, out-of-range counts and CUDA restore failures.
    pub fn commit(&mut self, device: &CudaDevice, slot: usize, accepted: usize) -> Result<()> {
        self.commit_many(device, from_ref(&(slot, accepted)))
    }

    /// Apply independently accepted prefixes together, after validating the complete batch.
    /// # Errors
    /// Rejects unknown, duplicate or non-pending slots, invalid counts and CUDA failures.
    #[expect(
        unsafe_code,
        reason = "Audited detached restore: single-stream ordering before any later replay, \
                  graph-owned tensors, drained by the owning pool's drop barrier"
    )]
    pub fn commit_many(&mut self, device: &CudaDevice, commits: &[(usize, usize)]) -> Result<()> {
        validate_commits(&self.pending, commits)?;
        if commits.is_empty() {
            return Ok(());
        }
        if let Some(batch) = self.batched_commit.as_mut().filter(|_| commits.len() > 1) {
            batch.update(device, commits)?;
            // SAFETY: same capture stream, retained state/count owners and the pool's drop barrier.
            unsafe { batch.graph().launch().async_on(&device.stream) }.map_err(device_error)?;
        } else {
            for &(slot, accepted) in commits {
                if let Some(restore) = self
                    .restore
                    .get(slot)
                    .and_then(|per_slot| per_slot.get(accepted))
                    .filter(|_| self.fused || accepted + 1 < self.verify)
                {
                    // SAFETY: ordered on the same stream; the pool's teardown drains all captures.
                    unsafe { restore.launch().async_on(&device.stream) }.map_err(device_error)?;
                }
            }
        }
        for &(slot, accepted) in commits {
            self.cursors[slot] += 1 + accepted;
            self.pending[slot] = 0;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) const fn has_batched_commit(&self) -> bool {
        self.batched_commit.is_some()
    }

    /// Host-side state cursors of every slot.
    #[must_use]
    pub fn cursors(&self) -> &[usize] {
        &self.cursors
    }

    /// Drain the capture stream: the pool's teardown barrier for detached restores.
    /// # Errors
    /// Returns the driver error when the stream cannot be synchronized.
    #[expect(
        unsafe_code,
        reason = "Audited stream drain at teardown: no work is enqueued after the owning \
                  pool starts dropping"
    )]
    pub(super) fn barrier(&self) -> Result<()> {
        // SAFETY: teardown only; no work is enqueued after the owning pool starts dropping.
        unsafe { self.graph.stream().synchronize() }.map_err(device_error)
    }

    /// Forget a failed replay when the owning sequence releases its lease.
    pub fn discard(&mut self, slot: usize) {
        if let Some(pending) = self.pending.get_mut(slot) {
            *pending = 0;
        }
    }

    /// Seed a slot's cursor after binding a sequence's covered prefix into it.
    /// # Errors
    /// Rejects unknown slot indices.
    pub fn set_cursor(&mut self, slot: usize, position: usize) -> Result<()> {
        let cursor = self
            .cursors
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("slot verify cursor"))?;
        *cursor = position;
        self.pending[slot] = 0;
        Ok(())
    }
}

type VerifyMetadata = Option<(u32, usize, usize)>;

/// Validate every active slot as a contiguous prefix before updating any device metadata.
fn verify_lanes(
    lanes: &[SlotVerifyLane],
    cursors: &[usize],
    verify: usize,
    capacity: usize,
    vocabulary: usize,
) -> Result<Vec<VerifyMetadata>> {
    let total = cursors.len() * verify;
    if lanes.is_empty() || lanes.len() > total {
        return Err(Error::invalid("slot verify lane count"));
    }
    let mut per_lane = vec![None; total];
    for &(slot, position, token, rope_pos, state_pos) in lanes {
        if slot >= cursors.len() || position >= verify {
            return Err(Error::invalid("slot verify lane"));
        }
        let lane = slot * verify + position;
        if per_lane[lane].is_some()
            || usize::try_from(token).map_err(device_error)? >= vocabulary
            || state_pos >= capacity
            || state_pos != rope_pos
            || cursors[slot].checked_add(position) != Some(state_pos)
        {
            return Err(Error::invalid("slot verify token or position"));
        }
        per_lane[lane] = Some((token, rope_pos, state_pos));
    }
    for group in per_lane.chunks_exact(verify) {
        let prefix = group.iter().take_while(|entry| entry.is_some()).count();
        if group[prefix..].iter().any(Option::is_some) {
            return Err(Error::invalid("slot verify chain is not a prefix"));
        }
    }
    Ok(per_lane)
}

#[cfg(test)]
#[path = "../../tests/unit/resident_slot_verify.rs"]
mod tests;
