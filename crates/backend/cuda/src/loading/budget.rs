//! Admission accounting for allocated tensor storage, plus driver graph headroom.
use super::LoadedModel;
use crate::constants::ATTENTION_SPLIT_MAX_PARTS;
use crate::constants::F32_BYTES;
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::MTP_FUSION_ROWS;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::constants::PREFILL_LANES;
use crate::constants::PREFILL_SPLIT_K;
use crate::mlp::ProjectionWeight;
use crate::resident::arena::ActivationArena;
use crate::resident::attention_mma::catalog::DIM;
use crate::resident::attention_mma::catalog::KV_HEADS;
use crate::resident::attention_mma::catalog::workspace_bytes;
use crate::resident::attention_mma::eligible;
use crate::resident::capacity::for_model;
use crate::resident::fusion_batch;
use crate::resident::fusion_batch::required_bytes;
use crate::resident::prompt_logits;
use crate::resident::recurrent_prefill::required_bytes as recurrent_prefill_required_bytes;
use crate::resident::slot_batch::SlotPool;
use crate::tuning::attention::decode::GROUPED_ROWS;
use crate::tuning::attention::decode::MAX_GROUPED_PARTS;
use crate::{
    device::DeviceProfile,
    resident::{ProgramGeometry, ProgramWeights},
};
use cutile::half::bf16;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, OutputReadout, StateKind, TensorOp, TensorStorage};
use std::array::from_fn;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// Bytes per F32 element in u64 budget arithmetic.
const F32: u64 = F32_BYTES as u64;
/// Full-history readback stages a device copy and returns a second host copy.
const READBACK_COPIES: u64 = 2;
/// Private/shared verification crossed with full/other readout.
const PRICE_GEOMETRIES: usize = 4;

/// Conservative control words per pooled lane, covering metadata and token readout.
const POOLED_CONTROL_WORDS: usize = 16;

/// Exact storage price for one immutable program geometry.
#[derive(Clone, Copy, Default)]
struct SequencePrice {
    fixed: u64,
    per_token: u64,
}

impl SequencePrice {
    fn at(self, capacity: u64) -> Result<u64> {
        add(self.fixed, mul(self.per_token, capacity)?)
    }

    fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            fixed: add(self.fixed, other.fixed)?,
            per_token: add(self.per_token, other.per_token)?,
        })
    }
}

/// Four bounded, successful prices: private/shared verification × full/other readout.
pub(super) struct SequencePrices([OnceLock<SequencePrice>; PRICE_GEOMETRIES]);

impl Default for SequencePrices {
    fn default() -> Self {
        Self(from_fn(|_| OnceLock::new()))
    }
}

/// Immutable component quote captured before a pooled serving plan allocates device storage.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PooledServingQuote {
    pub width: usize,
    pub capacity: usize,
    pub kv_tokens: usize,
    pub prompt_width: usize,
    pub target_pool_bytes: u64,
    pub draft_pool_bytes: u64,
    pub target_worker_bytes: u64,
    pub draft_worker_bytes: u64,
    pub target_homes_bytes: u64,
    pub draft_homes_bytes: u64,
    pub graph_headroom_bytes: u64,
    pub total_bytes: u64,
}

impl LoadedModel {
    pub(super) fn validate_capacity(&self, capacity: usize) -> Result<usize> {
        for_model(capacity, self.model.max_sequence)
    }

    /// Upper bound for known tensor storage plus 256 MiB of graph/allocator headroom.
    /// This is admission accounting, not a measurement of CUDA driver allocations.
    ///
    /// `private_verification` prices the rollback snapshots and batch workspaces of a
    /// sequence-owned verification graph. A sequence that will speculate through the shared
    /// slot pool omits it unless prefill needs the private batch as its fallback graph.
    /// # Errors
    /// Rejects invalid capacity or arithmetic overflow before any device allocation.
    pub fn sequence_budget(
        &self,
        capacity: usize,
        readout: OutputReadout,
        private_verification: bool,
    ) -> Result<u64> {
        let capacity = self.validate_capacity(capacity)? as u64;
        // Graph geometry is immutable after load; only KV and full-history storage grow
        // with capacity. Keep one exact affine price per readout/verification geometry.
        let index =
            usize::from(private_verification) * 2 + usize::from(readout == OutputReadout::Full);
        let cached = &self.sequence_prices.0[index];
        let price = if let Some(price) = cached.get() {
            *price
        } else {
            let price = self.sequence_price(readout, private_verification)?;
            // Concurrent first quotes can calculate the same immutable price.
            let _ = cached.set(price);
            price
        };
        price.at(capacity)
    }

    fn sequence_price(
        &self,
        readout: OutputReadout,
        private_verification: bool,
    ) -> Result<SequencePrice> {
        let hidden = self.model.hidden_size;
        let vocabulary = self.model.vocab_size;
        let profile = self.device().profile()?;
        let mut price = SequencePrice {
            fixed: profile.graph_headroom_bytes(),
            per_token: F32,
        };
        price = price.combine(program_price(
            &self.graph,
            &self.weights,
            hidden,
            vocabulary,
            readout,
            ProgramGeometry::new(&self.weights, private_verification),
            profile,
        )?)?;
        if let Some(draft) = &self.draft {
            // Draft priming retains a prefill arena and always captures private verification.
            price = price.combine(program_price(
                &draft.graph,
                &draft.weights,
                hidden,
                vocabulary,
                readout,
                ProgramGeometry::new(&draft.weights, true),
                profile,
            )?)?;
        }
        Ok(price)
    }

    /// Device bytes of the prompt scratch the target and draft programs share across all of
    /// their sequences. Charged once against the state budget, never per sequence.
    /// # Errors
    /// Rejects arithmetic overflow.
    pub fn shared_scratch_bytes(&self) -> Result<u64> {
        let profile = self.device().profile()?;
        let mut bytes = prompt_scratch_bytes(
            &self.graph,
            &self.weights,
            ProgramGeometry::new(&self.weights, true),
            profile,
        )?;
        if let Some(draft) = &self.draft {
            bytes = add(
                bytes,
                prompt_scratch_bytes(
                    &draft.graph,
                    &draft.weights,
                    ProgramGeometry::new(&draft.weights, true),
                    profile,
                )?,
            )?;
        }
        Ok(bytes)
    }

    /// KV bytes one block of `KV_BLOCK_TOKENS` positions occupies across every attention
    /// state of the target graph, fp8-packed where the checkpoint carries scales. This is
    /// the physical page price the slot pool's block accounting reports to the engine; the
    /// recurrent state does not grow with tokens and is priced by admission, not per block.
    /// # Errors
    /// Rejects a malformed KV shape or arithmetic overflow.
    pub fn kv_block_bytes(&self) -> Result<u64> {
        kv_block_bytes(&self.graph, &self.weights)
    }
}

/// Joint quote for the pooled target, draft, prompt worker and resident slot homes.
/// The shared scratch is charged at the worker's geometry, rather than the private
/// sequence's maximum prompt width. No graph is captured while evaluating a plan.
impl LoadedModel {
    pub(crate) fn pooled_serving_quote(
        &self,
        width: usize,
        capacity: usize,
        kv_tokens: usize,
        prompt_width: usize,
    ) -> Result<PooledServingQuote> {
        let capacity = self.validate_capacity(capacity)?;
        let profile = self.device.profile()?;
        let hidden = self.model.hidden_size;
        let vocabulary = self.model.vocab_size;
        let verify = if self.mtp_depth == 0 {
            1
        } else {
            self.mtp_depth + 1
        };
        let target_pool_bytes = pool_storage_bytes(
            &self.graph,
            &self.weights,
            (width, capacity, kv_tokens),
            verify,
            hidden,
            profile,
        )?;
        // A target home is a state handle: no forward/attention/projection capture.
        let target_home = add(
            ActivationArena::required_bytes(&self.graph, 1)? as u64,
            external_bytes(
                &self.weights,
                hidden,
                ProgramGeometry {
                    verification: 0,
                    prompts: [0, 0],
                },
                profile,
            )?,
        )?;
        let target_homes_bytes = mul(target_home, width as u64)?;
        let target_worker_bytes = worker_storage_bytes(
            &self.graph,
            &self.weights,
            (capacity, hidden, vocabulary),
            prompt_width,
            true,
            profile,
        )?;
        let (draft_pool_bytes, draft_homes_bytes, draft_worker_bytes) =
            if let Some(draft) = &self.draft {
                let pool = pool_storage_bytes(
                    &draft.graph,
                    &draft.weights,
                    (width, capacity, width * capacity),
                    1,
                    hidden,
                    profile,
                )?;
                let geometry = ProgramGeometry {
                    verification: 0,
                    prompts: [0, 0],
                };
                let home = program_price(
                    &draft.graph,
                    &draft.weights,
                    hidden,
                    vocabulary,
                    OutputReadout::Logits,
                    geometry,
                    profile,
                )?;
                // Slot homes alias the pool's recurrent and KV state, but own their workspaces.
                let home_fixed = home
                    .fixed
                    .saturating_sub(recurrent_state_bytes(&draft.graph)?);
                let homes = mul(home_fixed, width as u64)?;
                let worker = worker_storage_bytes(
                    &draft.graph,
                    &draft.weights,
                    (capacity, hidden, vocabulary),
                    prompt_width,
                    false,
                    profile,
                )?;
                (pool, homes, worker)
            } else {
                (0, 0, 0)
            };
        let graph_headroom_bytes = profile.graph_headroom_bytes();
        let mut total_bytes = add(target_pool_bytes, target_homes_bytes)?;
        total_bytes = add(total_bytes, target_worker_bytes)?;
        total_bytes = add(total_bytes, draft_pool_bytes)?;
        total_bytes = add(total_bytes, draft_homes_bytes)?;
        total_bytes = add(total_bytes, draft_worker_bytes)?;
        // Captures and allocator metadata are not tensor storage.
        total_bytes = add(total_bytes, graph_headroom_bytes)?;
        Ok(PooledServingQuote {
            width,
            capacity,
            kv_tokens,
            prompt_width,
            target_pool_bytes,
            draft_pool_bytes,
            target_worker_bytes,
            draft_worker_bytes,
            target_homes_bytes,
            draft_homes_bytes,
            graph_headroom_bytes,
            total_bytes,
        })
    }

    /// Fund physical KV pages from the remaining whole-serving budget. Logical slot
    /// capacity stays unchanged; admission leases pages for each request's actual capacity.
    pub(crate) fn funded_pooled_serving(
        &self,
        width: usize,
        capacity: usize,
        maximum_kv_tokens: usize,
        prompt_width: usize,
        envelope: u64,
    ) -> Result<Option<PooledServingQuote>> {
        let capacity = self.validate_capacity(capacity)?;
        let maximum_kv_tokens = maximum_kv_tokens / KV_BLOCK_TOKENS * KV_BLOCK_TOKENS;
        let full = self.pooled_serving_quote(width, capacity, maximum_kv_tokens, prompt_width)?;
        if full.total_bytes <= envelope {
            return Ok(Some(full));
        }
        let fixed = self
            .pooled_serving_quote(width, capacity, 0, prompt_width)?
            .total_bytes;
        let blocks = maximum_kv_tokens / KV_BLOCK_TOKENS;
        if fixed >= envelope || blocks == 0 {
            return Ok(None);
        }
        let block_bytes = full.total_bytes.saturating_sub(fixed) / blocks as u64;
        let Some(tokens) = funded_kv_tokens(
            width,
            capacity,
            maximum_kv_tokens,
            block_bytes,
            fixed,
            envelope,
        )?
        else {
            return Ok(None);
        };
        let bytes = add(fixed, mul((tokens / KV_BLOCK_TOKENS) as u64, block_bytes)?)?;
        let quote = self.pooled_serving_quote(width, capacity, tokens, prompt_width)?;
        debug_assert_eq!(quote.total_bytes, bytes);
        Ok(Some(quote))
    }

    /// Physical scratch a new private program still needs, beyond its affine state quote.
    /// This matters when a narrower pooled worker has not allocated the private prompt width.
    pub(crate) fn pending_prompt_scratch_bytes(&self, private: bool) -> Result<u64> {
        let profile = self.device.profile()?;
        let mut bytes = missing_prompt_scratch_bytes(
            &self.graph,
            &self.weights,
            ProgramGeometry::new(&self.weights, private),
            profile,
        )?;
        if let Some(draft) = &self.draft {
            bytes = add(
                bytes,
                missing_prompt_scratch_bytes(
                    &draft.graph,
                    &draft.weights,
                    ProgramGeometry::new(&draft.weights, true),
                    profile,
                )?,
            )?;
        }
        Ok(bytes)
    }

    pub(crate) const fn pooled_prompt_width(&self) -> usize {
        self.weights.prefill_width
    }
}

/// A logical power-of-two bucket starts just above half its padded capacity. Keep
/// enough pages for every resident slot to admit a request from that bucket, instead
/// of funding many unusable state handles with only a handful of physical KV blocks.
fn funded_kv_tokens(
    width: usize,
    capacity: usize,
    maximum: usize,
    block_bytes: u64,
    fixed: u64,
    envelope: u64,
) -> Result<Option<usize>> {
    if block_bytes == 0 || fixed >= envelope {
        return Ok(None);
    }
    let per_slot = (capacity / 2 / KV_BLOCK_TOKENS + 1) * KV_BLOCK_TOKENS;
    let minimum = width
        .checked_mul(per_slot)
        .ok_or_else(|| Error::invalid("pooled minimum KV overflow"))?;
    let pages = (envelope - fixed) / block_bytes;
    let tokens = usize::try_from(pages)
        .unwrap_or(usize::MAX)
        .min(maximum / KV_BLOCK_TOKENS)
        * KV_BLOCK_TOKENS;
    Ok((tokens >= minimum).then_some(tokens))
}

fn missing_prompt_scratch_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    geometry: ProgramGeometry,
    profile: &DeviceProfile,
) -> Result<u64> {
    let mut bytes = 0;
    for width in geometry.prompts.into_iter().filter(|width| *width > 0) {
        if !weights.prompt_scratch.is_allocated(width)? {
            bytes = add(
                bytes,
                prompt_scratch_bytes(
                    graph,
                    weights,
                    ProgramGeometry {
                        verification: 0,
                        prompts: [width, 0],
                    },
                    profile,
                )?,
            )?;
        }
    }
    Ok(bytes)
}

fn recurrent_state_bytes(graph: &DataflowGraph) -> Result<u64> {
    graph
        .tensors
        .iter()
        .filter(|spec| {
            matches!(
                spec.storage,
                TensorStorage::State {
                    kind: StateKind::Conv | StateKind::LinearAttention,
                    ..
                }
            )
        })
        .try_fold(0, |bytes, spec| {
            add(bytes, mul(spec.elements()? as u64, F32)?)
        })
}

fn pool_storage_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    (width, capacity, kv_tokens): (usize, usize, usize),
    verify: usize,
    hidden: usize,
    profile: &DeviceProfile,
) -> Result<u64> {
    let lanes = width * verify;
    let mut bytes = mul(recurrent_state_bytes(graph)?, width as u64)?;
    for spec in &graph.tensors {
        if !matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            }
        ) {
            continue;
        }
        let columns = *spec
            .shape
            .get(1)
            .ok_or_else(|| Error::invalid("pooled KV shape"))? as u64;
        let (tokens, size) = if weights.kv_scales.contains_key(&spec.id) {
            (kv_tokens, 1)
        } else {
            (width * capacity, F32)
        };
        bytes = add(bytes, mul(columns, mul(tokens as u64, 2 * size)?)?)?;
    }
    bytes = add(bytes, ActivationArena::required_bytes(graph, lanes)? as u64)?;
    bytes = add(bytes, SlotPool::rollback_bytes(graph, width, verify)?)?;
    bytes = add(bytes, SlotPool::parameter_bytes(graph, verify)?)?;
    bytes = add(
        bytes,
        SlotPool::attention_bytes(graph, weights, (width, verify, capacity), profile)?,
    )?;
    // These widths prepare packed projection inputs; pooled verification writes directly
    // into arena rows and needs none of the private graph's pack/unpack buffers.
    bytes = add(
        bytes,
        projection_workspace(
            graph,
            weights,
            ProgramGeometry {
                verification: 0,
                prompts: [lanes, 0],
            },
        )?,
    )?;
    bytes = add(bytes, attention_scratch(graph)?)?;
    bytes = add(bytes, required_bytes(weights, lanes, profile))?;
    let external_rows = 1 + lanes as u64
        * if weights.fusion.is_some() {
            MTP_FUSION_ROWS
        } else {
            1
        };
    bytes = add(bytes, mul(mul(hidden as u64, external_rows)?, F32)?)?;
    // A block table per slot, plus scalar and per-lane control words.
    let controls = width * capacity.div_ceil(KV_BLOCK_TOKENS) + (1 + lanes) * POOLED_CONTROL_WORDS;
    bytes = add(bytes, mul(controls as u64, F32)?)?;
    Ok(bytes)
}

fn worker_storage_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    (capacity, hidden, vocabulary): (usize, usize, usize),
    prompt_width: usize,
    shared_kv: bool,
    profile: &DeviceProfile,
) -> Result<u64> {
    let geometry = ProgramGeometry::worker(weights, prompt_width);
    let price = program_price(
        graph,
        weights,
        hidden,
        vocabulary,
        OutputReadout::Logits,
        geometry,
        profile,
    )?;
    let mut bytes = price.at(capacity as u64)?;
    if shared_kv {
        let shared = graph
            .tensors
            .iter()
            .filter(|spec| weights.kv_scales.contains_key(&spec.id))
            .try_fold(0, |sum, spec| {
                let columns = *spec
                    .shape
                    .get(1)
                    .ok_or_else(|| Error::invalid("shared worker KV shape"))?
                    as u64;
                add(sum, mul(columns, 2 * capacity as u64)?)
            })?;
        bytes = bytes.saturating_sub(shared);
    }
    add(
        bytes,
        prompt_scratch_bytes(graph, weights, geometry, profile)?,
    )
}

/// One KV block's bytes over the graph's attention states: `columns` per position, keys and
/// values, packed to one byte where the checkpoint carries fp8 scales.
fn kv_block_bytes(graph: &DataflowGraph, weights: &ProgramWeights) -> Result<u64> {
    let block = u64::try_from(KV_BLOCK_TOKENS).map_err(|_| Error::invalid("KV block size"))?;
    let mut bytes = 0;
    for spec in &graph.tensors {
        if !matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            }
        ) {
            continue;
        }
        let columns = *spec
            .shape
            .get(1)
            .ok_or_else(|| Error::invalid("KV shape"))? as u64;
        let size = if weights.kv_scales.contains_key(&spec.id) {
            1
        } else {
            F32
        };
        bytes = add(bytes, mul(mul(block, columns)?, 2 * size)?)?;
    }
    Ok(bytes)
}

/// Per-program admission: activation arenas, KV pages, projection workspaces and readout staging.
fn program_price(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    hidden: usize,
    vocabulary: usize,
    readout: OutputReadout,
    geometry: ProgramGeometry,
    profile: &DeviceProfile,
) -> Result<SequencePrice> {
    let verify = geometry.verification as u64;
    let prompt = geometry.prompt_lanes() as u64;
    let lanes = geometry.arena_lanes();
    let mut bytes = 0;
    let mut per_token = 0;
    bytes = add(bytes, attention_scratch(graph)?)?;
    let max_prompt = geometry.prompts.into_iter().max().unwrap_or(0);
    if eligible(&profile.architecture, graph, weights, max_prompt) {
        let fixed = workspace_bytes(0, max_prompt)?;
        bytes = add(
            bytes,
            u64::try_from(fixed).map_err(|_| Error::invalid("MMA scratch budget"))?,
        )?;
        // One BF16 K/V pair shared across every attention layer, not one pair per layer.
        let token = KV_HEADS * DIM * 2 * size_of::<bf16>();
        per_token = add(
            per_token,
            u64::try_from(token).map_err(|_| Error::invalid("MMA token budget"))?,
        )?;
    }
    // Activations live in a reuse arena sized by the peak live set, not by the sum of every
    // tensor in the graph; charge what the arena will actually allocate. Prompt arenas and the
    // rest of the prompt scratch are shared by every sequence ([`prompt_scratch_bytes`]).
    let other_arenas = ActivationArena::required_bytes(graph, lanes - geometry.prompt_lanes())?;
    bytes = add(
        bytes,
        u64::try_from(other_arenas).map_err(|_| Error::invalid("activation budget"))?,
    )?;
    for spec in &graph.tensors {
        let elements = spec.elements()? as u64;
        let amount = match spec.storage {
            TensorStorage::Activation | TensorStorage::Weight { .. } => 0,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            } => {
                let columns = *spec
                    .shape
                    .get(1)
                    .ok_or_else(|| Error::invalid("KV shape"))?
                    as u64;
                let size = if weights.kv_scales.contains_key(&spec.id) {
                    1
                } else {
                    F32
                };
                per_token = add(per_token, mul(columns, 2 * size)?)?;
                0
            }
            TensorStorage::State { .. } => mul(elements, mul(1 + verify, F32)?)?,
        };
        bytes = add(bytes, amount)?;
    }
    bytes = add(bytes, projection_workspace(graph, weights, geometry)?)?;
    bytes = add(bytes, external_bytes(weights, hidden, geometry, profile)?)?;
    // Pinned readback is allocated on demand and capped per batch (device/readback.rs), so
    // charge what a step actually stages: one decode lane, the verification batch when it is
    // private, and the last prompt lane (prompt modes read only the last logits row).
    let readout_elements = add(hidden as u64, vocabulary as u64)?;
    let readout_lanes = 1 + if verify > 1 { verify } else { 0 } + u64::from(prompt > 0);
    bytes = add(bytes, mul(mul(readout_elements, readout_lanes)?, F32)?)?;
    if readout == OutputReadout::Full {
        // Full-history CPU readback and its returned copy are included in admission.
        per_token = add(per_token, mul(hidden as u64, READBACK_COPIES * F32)?)?;
    }
    Ok(SequencePrice {
        fixed: bytes,
        per_token,
    })
}
/// Split-KV scratch is shared by sequential attention nodes with the same geometry.
/// Charge the supported upper bound (16 partitions), independent of device tuning.
/// External hidden rows for every lane; fused draft programs also retain embedding and two
/// normalized rows per lane, except where batched prompt fusion covers the prompt graph and only
/// decode and verification lanes keep their own.
/// Prompt-graph scratch one program allocates once and shares across its sequences
/// (`resident::prompt_scratch`): per prompt width, the arena with its one logits row
/// (`prompt_logits`), the chunked recurrent workspace and the fusion workspace.
fn prompt_scratch_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    geometry: ProgramGeometry,
    profile: &DeviceProfile,
) -> Result<u64> {
    let mut bytes = 0;
    for width in geometry.prompts.into_iter().filter(|width| *width > 0) {
        let arena = prompt_logits::required_bytes(graph, weights, width)?;
        bytes = add(
            bytes,
            u64::try_from(arena).map_err(|_| Error::invalid("activation budget"))?,
        )?;
        bytes = add(
            bytes,
            recurrent_prefill_required_bytes(graph, weights, width)?,
        )?;
        bytes = add(bytes, required_bytes(weights, width, profile))?;
        bytes = add(bytes, prompt_partials_bytes(graph, weights, width)?)?;
    }
    Ok(bytes)
}

/// Split-K partial rows for every FP4 prompt projection at `width`, allocated once by the
/// shared prompt scratch and reused by every sequence.
fn prompt_partials_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    width: usize,
) -> Result<u64> {
    let sizes = graph
        .tensors
        .iter()
        .map(|tensor| Ok((tensor.id, tensor.elements()? as u64)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut partials = BTreeSet::new();
    let mut bytes = 0;
    for node in &graph.nodes {
        if node.op != TensorOp::Linear
            || weights.input_scales.contains_key(&node.inputs[1])
            || !matches!(
                weights.projections.get(&node.inputs[1]),
                Some(ProjectionWeight::Fp4(..))
            )
        {
            continue;
        }
        let rows = *sizes
            .get(&node.outputs[0])
            .ok_or_else(|| Error::invalid("prefill linear output"))?;
        if partials.insert(rows) {
            bytes = add(bytes, mul(rows, (PREFILL_SPLIT_K * width) as u64 * F32)?)?;
        }
    }
    Ok(bytes)
}

fn external_bytes(
    weights: &ProgramWeights,
    hidden: usize,
    geometry: ProgramGeometry,
    profile: &DeviceProfile,
) -> Result<u64> {
    let width = geometry.widths().into_iter().max().unwrap_or(1) as u64;
    let external_lanes = 1 + width;
    let fused_lanes = if weights.fusion.is_none() {
        0
    } else if fusion_batch::batched(weights, weights.prefill_width, profile) {
        1 + (weights.batch_width.max(1) as u64).min(width)
    } else {
        external_lanes
    };
    let rows = external_lanes + fused_lanes * (MTP_FUSION_ROWS - 1);
    mul(mul(hidden as u64, rows)?, F32)
}

/// Partial rows per KV head of the grouped decode kernel: 64 splits of the single-lane kernel's
/// 16 query rows plus the three-lane kernel's 32 stacked rows.
const GROUPED_DECODE_ROWS: usize = 3 * MAX_GROUPED_PARTS * GROUPED_ROWS;
fn attention_scratch(graph: &DataflowGraph) -> Result<u64> {
    let mut bytes = 0;
    let mut shapes = BTreeSet::new();
    for node in &graph.nodes {
        if let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            ..
        } = node.op
            && shapes.insert((query_heads, head_dim))
        {
            // Either one partial row per query head and split, or the grouped decode layout of
            // 32 rows per KV head and split (resident/attention_decode/workspace.rs).
            let rows =
                (query_heads * ATTENTION_SPLIT_MAX_PARTS).max(kv_heads * GROUPED_DECODE_ROWS);
            bytes = add(bytes, mul(rows as u64, mul(head_dim as u64 + 2, F32)?)?)?;
        }
    }
    Ok(bytes)
}
/// Projection buffers are shared by geometry; prefill views use the activation arena.
fn projection_workspace(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    geometry: ProgramGeometry,
) -> Result<u64> {
    let sizes = graph
        .tensors
        .iter()
        .map(|t| Ok((t.id, t.elements()? as u64)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut verification = BTreeSet::new();
    let mut partials = BTreeSet::new();
    let mut quantized = BTreeSet::new();
    let mut fp8_quantized = BTreeSet::new();
    let quantized_rows = geometry
        .widths()
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .sum::<usize>() as u64;
    let mut bytes = 0;
    for node in &graph.nodes {
        if node.op != TensorOp::Linear {
            continue;
        }
        let columns = *sizes
            .get(&node.inputs[0])
            .ok_or_else(|| Error::invariant("projection input budget"))?;
        let rows = *sizes
            .get(&node.outputs[0])
            .ok_or_else(|| Error::invariant("projection output budget"))?;
        if weights.fp8_inputs.contains(&node.inputs[1]) && fp8_quantized.insert(columns) {
            bytes = add(bytes, mul(columns + F32, quantized_rows)?)?;
        }
        if weights.input_scales.contains_key(&node.inputs[1]) && quantized.insert(columns) {
            // Two FP4 values per byte, plus one E4M3 scale per block of 16.
            bytes = add(
                bytes,
                mul(
                    columns / 2 + columns / NVFP4_GROUP_SIZE as u64,
                    quantized_rows,
                )?,
            )?;
        }
        if geometry.verification > 1 && verification.insert((rows, columns)) {
            bytes = add(
                bytes,
                mul(add(rows, columns)?, geometry.verification as u64 * F32)?,
            )?;
        }
        if geometry.prompts.contains(&PREFILL_LANES)
            && !weights.input_scales.contains_key(&node.inputs[1])
            && matches!(
                weights.projections.get(&node.inputs[1]),
                Some(ProjectionWeight::Fp4(..))
            )
            && partials.insert(rows)
        {
            bytes = add(
                bytes,
                mul(rows, (PREFILL_SPLIT_K * PREFILL_LANES) as u64 * F32)?,
            )?;
        }
    }
    Ok(bytes)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error::invalid("CUDA state budget overflow"))
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::invalid("CUDA state budget overflow"))
}

#[cfg(test)]
#[path = "../../tests/unit/loading_budget.rs"]
mod tests;
