//! One prompt graph for many sequences: every weight is read once per replay.
//!
//! Row-independent nodes (embedding, norm, projections, elementwise) run once over
//! `segments * SEGMENT_ROWS` rows. Sequential nodes (convolution, gated delta, attention, `RoPE`)
//! record once per segment, against that segment's rows and the resident slot's own state, so no
//! state moves through a worker. A segment with a zero token count writes nothing.
use super::batch::{BatchBuilder, States};
use super::fp8_cache::Fp8Caches;
use super::{ActivationArena, ProgramWeights};
use crate::constants::METADATA_FIELDS;
use crate::device::{CudaDevice, PinnedUpload, ReadbackSource, Readbacks, device_error};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::DataflowGraph;
use infer_models::mrope::ROTARY_AXES;
use std::collections::BTreeMap;
use std::slice::from_ref;
use std::sync::Arc;

/// Rows per sequence: one chunk of the tensor-core recurrent rule.
pub const SEGMENT_ROWS: usize = 64;
/// Columns one `last_rows` block copies.
pub(super) const ROW_TILE: usize = 256;
/// The state position sequential kernels skip.
const INACTIVE_STATE_POSITION: i32 = -1;

/// One sequence's view of the graph while it is captured.
pub(super) struct Segment<'a> {
    pub arena: ActivationArena,
    pub states: &'a mut States,
    pub fp8: &'a mut Fp8Caches,
    pub table: &'a Tensor<i32>,
    pub info: &'a Tensor<i32>,
    /// Per-row `(t, h, w)` when the graph's `RoPE` reads absolute coordinates.
    pub rotary: Option<&'a Tensor<i32>>,
}

/// Everything a multi-sequence capture records beyond an ordinary prompt graph.
pub(super) struct Binding<'a> {
    pub segments: Vec<Segment<'a>>,
    pub recurrent: super::recurrent_prefill::Workspace,
    pub info_batch: &'a Tensor<i32>,
    pub last_rows: &'a mut Tensor<f32>,
    pub logits: &'a mut Tensor<f32>,
    pub segment_rows: usize,
}

/// One sequence's tokens for a replay.
pub struct Lane<'a> {
    pub segment: usize,
    pub tokens: &'a [u32],
    pub position: usize,
}

/// A captured multi-sequence prompt graph, retaining everything the graph addresses.
pub struct PromptBurst {
    graph: CudaGraph<()>,
    profile: Option<super::profile::PrefillProfile>,
    segments: usize,
    capacity: usize,
    vocabulary: usize,
    state_offset: i32,
    metadata: Tensor<i32>,
    metadata_upload: Vec<i32>,
    metadata_staging: PinnedUpload<i32>,
    info: Tensor<i32>,
    info_upload: Vec<i32>,
    info_staging: PinnedUpload<i32>,
    rotary: Option<Tensor<i32>>,
    rotary_upload: Vec<i32>,
    rotary_staging: PinnedUpload<i32>,
    logits: Arc<Tensor<f32>>,
    readbacks: Readbacks,
    _retained: Retained,
    keep: Keep,
}

/// Workspaces the graph addresses and nothing reads through this struct.
struct Retained {
    _segments: Vec<ActivationArena>,
    _partials: BTreeMap<usize, Tensor<f32>>,
    _recurrent: super::recurrent_prefill::Workspace,
    _nvfp4: super::nvfp4_gemm::Workspace,
    _attention: super::attention_decode::Workspace,
    _placeholder_states: States,
    _placeholder_fp8: Fp8Caches,
}

/// Device tensors the graph addresses and nothing reads through this struct.
struct Keep {
    arena: ActivationArena,
    _metadata_rows: Vec<Tensor<i32>>,
    _info_rows: Vec<Tensor<i32>>,
    _rotary_rows: Vec<Tensor<i32>>,
    _prompt_row: Tensor<f32>,
    _external_row: Tensor<f32>,
    _external: Tensor<f32>,
    _lane_external: Vec<Tensor<f32>>,
    _lane_fusion: Vec<Option<super::program::FusionWorkspace>>,
    _last_rows: Arc<Tensor<f32>>,
}

/// Everything a capture binds before the graph exists.
struct Storage {
    arena: ActivationArena,
    metadata: Tensor<i32>,
    metadata_rows: Vec<Tensor<i32>>,
    info: Tensor<i32>,
    info_rows: Vec<Tensor<i32>>,
    rotary: Option<Tensor<i32>>,
    rotary_rows: Vec<Tensor<i32>>,
    last_rows: Tensor<f32>,
    logits: Tensor<f32>,
    prompt_row: Tensor<f32>,
    external_row: Tensor<f32>,
    external: Tensor<f32>,
    lane_external: Vec<Tensor<f32>>,
    lane_fusion: Vec<Option<super::program::FusionWorkspace>>,
}

/// Pair each resident program with its rows and metadata.
fn bind_segments<'a>(
    programs: &'a mut [&mut super::program::DeviceProgram],
    views: Vec<ActivationArena>,
    (info_rows, rotary_rows): (&'a [Tensor<i32>], &'a [Tensor<i32>]),
) -> Vec<Segment<'a>> {
    programs
        .iter_mut()
        .zip(views)
        .zip(info_rows)
        .enumerate()
        .map(|(segment, ((program, arena), info))| {
            let (states, fp8, table) = program.burst_binding();
            Segment {
                arena,
                states,
                fp8,
                table,
                info,
                rotary: rotary_rows.get(segment),
            }
        })
        .collect()
}

/// Projection, attention and recurrent workspaces of one capture.
fn workspaces(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    capacity: usize,
    widths: [usize; 2],
) -> Result<(
    super::nvfp4_gemm::Workspace,
    super::attention_decode::Workspace,
    super::recurrent_prefill::Workspace,
)> {
    let nvfp4 = super::nvfp4_gemm::Workspace::new(device, graph, weights, &widths)?;
    let mut attention = super::attention_decode::Workspace::new(device, graph, capacity)?;
    attention.prepare_mma(device, graph, weights, capacity, SEGMENT_ROWS)?;
    attention.prepare_prefill(device, graph, weights, SEGMENT_ROWS);
    let recurrent = super::recurrent_prefill::Workspace::new(device, graph, weights, SEGMENT_ROWS)?;
    Ok((nvfp4, attention, recurrent))
}

fn zeros<T: DType>(device: &CudaDevice, shape: &[usize]) -> Result<Tensor<T>> {
    api::zeros::<T>(shape)
        .sync_on(&device.stream)
        .map_err(device_error)
}

impl Storage {
    /// # Safety
    /// Row views alias the returned owners, which must outlive every graph captured from them.
    #[expect(
        unsafe_code,
        reason = "Audited row aliases of owners retained beside the graph"
    )]
    unsafe fn allocate(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        (segments, rows, hidden, vocabulary): (usize, usize, usize, usize),
        wider: Option<&ActivationArena>,
    ) -> Result<Self> {
        let budget =
            usize::try_from(device.profile()?.arena_budget_bytes()).map_err(device_error)?;
        let arena = match wider {
            // SAFETY: forwarded from this function's contract: replays share one stream.
            Some(wider) => unsafe { wider.narrowed(graph, rows, graph.logits)? },
            None => ActivationArena::new_prompt(device, graph, rows, budget)?,
        };
        let (metadata, metadata_rows) = super::batch::batched_metadata(device, rows)?;
        let (info, info_rows) = super::batch::batched_metadata(device, segments)?;
        let rotary = if weights.mrope {
            Some(zeros::<i32>(device, &[rows * ROTARY_AXES])?)
        } else {
            None
        };
        let mut rotary_rows = Vec::new();
        if let Some(owner) = rotary.as_ref() {
            for segment in 0..segments {
                // SAFETY: forwarded from this function's contract; the rows are disjoint.
                rotary_rows.push(unsafe { super::arena::row_view(owner, segments, segment)? });
            }
        }
        let external = zeros::<f32>(device, &[rows * hidden])?;
        let mut lane_external = Vec::with_capacity(rows);
        for lane in 0..rows {
            // SAFETY: as above.
            lane_external.push(unsafe { super::arena::row_view(&external, rows, lane)? });
        }
        Ok(Self {
            arena,
            metadata,
            metadata_rows,
            info,
            info_rows,
            rotary,
            rotary_rows,
            last_rows: zeros(device, &[segments, hidden])?,
            logits: zeros(device, &[segments, vocabulary])?,
            prompt_row: zeros(device, &[1, hidden])?,
            external_row: zeros(device, &[hidden])?,
            external,
            lane_external,
            lane_fusion: (0..rows).map(|_| None).collect(),
        })
    }

    /// # Safety
    /// As [`ActivationArena::segment`]; the arena is retained beside the graph.
    #[expect(unsafe_code, reason = "Forwards the arena segment contract")]
    unsafe fn segment_views(
        &self,
        graph: &DataflowGraph,
        segments: usize,
    ) -> Result<Vec<ActivationArena>> {
        (0..segments)
            .map(|segment| {
                // SAFETY: forwarded from this function's contract.
                unsafe {
                    self.arena
                        .segment(graph, SEGMENT_ROWS, segment * SEGMENT_ROWS, graph.logits)
                }
            })
            .collect()
    }
}

impl PromptBurst {
    /// Capture one graph over `programs`, each a resident slot's program.
    /// # Safety
    /// The programs' slot storage must outlive this graph, and it may replay only while no
    /// other stream touches the slots of its active segments.
    /// # Errors
    /// Rejects geometries the multi-sequence graph does not cover, and CUDA capture failures.
    #[expect(
        unsafe_code,
        reason = "Audited segment aliases: the burst retains the wide arena and metadata owner \
                  beside the graph, and the caller keeps the slot pool alive"
    )]
    pub unsafe fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        (capacity, hidden, vocabulary): (usize, usize, usize),
        programs: &mut [&mut super::program::DeviceProgram],
        wider: Option<&ActivationArena>,
    ) -> Result<Self> {
        let segments = programs.len();
        let rows = segments * SEGMENT_ROWS;
        // Fused inputs have no per-segment binding; media prompts stay on the worker.
        if segments == 0 || weights.fusion.is_some() || !hidden.is_multiple_of(ROW_TILE) {
            return Err(Error::unsupported(
                "prompt burst covers plain text graphs with tiled hidden rows",
            ));
        }
        // SAFETY: forwarded from this function's contract.
        let mut storage = unsafe {
            Storage::allocate(
                device,
                graph,
                weights,
                (segments, rows, hidden, vocabulary),
                wider,
            )?
        };
        let (mut nvfp4, mut attention, recurrent) =
            workspaces(device, graph, weights, capacity, [rows, segments])?;
        // SAFETY: forwarded from this function's contract.
        let views = unsafe { storage.segment_views(graph, segments)? };
        let bound = bind_segments(programs, views, (&storage.info_rows, &storage.rotary_rows));
        let mut placeholder_states = States::new();
        let mut placeholder_fp8 = Fp8Caches::new();
        let tables = [super::metadata::identity_table(device, capacity)?];
        let binding = Binding {
            segments: bound,
            recurrent,
            info_batch: &storage.info,
            last_rows: &mut storage.last_rows,
            logits: &mut storage.logits,
            segment_rows: SEGMENT_ROWS,
        };
        let mut builder = BatchBuilder {
            device,
            graph,
            weights,
            nvfp4: &mut nvfp4,
            delegated: None,
            attention: &mut attention,
            states: &mut placeholder_states,
            fp8_states: &mut placeholder_fp8,
            external: &storage.external_row,
            lane_external: &mut storage.lane_external,
            lane_external_owner: Some(&storage.external),
            lane_fusion: &mut storage.lane_fusion,
            tables: &tables,
            capacity,
            width: rows,
            burst: Some(binding),
        };
        let mut partials = builder.burst_partials()?;
        let mut profile = BatchBuilder::burst_profile(device);
        // `info_rows[0]` is unused: every sequential node resolves through its segment.
        let captured = builder.capture_burst(
            &mut storage.arena,
            (
                &storage.metadata_rows,
                &storage.metadata,
                &storage.info_rows[0],
                None,
            ),
            &mut partials,
            &mut storage.prompt_row,
            &mut profile,
        )?;
        let Binding {
            recurrent,
            segments: bound,
            ..
        } = builder
            .burst
            .take()
            .ok_or_else(|| Error::invariant("prompt burst binding"))?;
        let segment_arenas = bound.into_iter().map(|segment| segment.arena).collect();
        Self::finish(
            device,
            storage,
            captured,
            profile,
            (segments, rows, capacity, vocabulary),
            BatchBuilder::burst_state_offset(weights),
            Retained {
                _segments: segment_arenas,
                _partials: partials,
                _recurrent: recurrent,
                _nvfp4: nvfp4,
                _attention: attention,
                _placeholder_states: placeholder_states,
                _placeholder_fp8: placeholder_fp8,
            },
        )
    }

    fn finish(
        device: &CudaDevice,
        storage: Storage,
        graph: CudaGraph<()>,
        profile: Option<super::profile::PrefillProfile>,
        (segments, rows, capacity, vocabulary): (usize, usize, usize, usize),
        state_offset: i32,
        retained: Retained,
    ) -> Result<Self> {
        let last_rows = Arc::new(storage.last_rows);
        let logits = Arc::new(storage.logits);
        let readbacks = Readbacks::pairs(device, from_ref(&last_rows), from_ref(&logits))?;
        Ok(Self {
            graph,
            profile,
            segments,
            capacity,
            vocabulary,
            state_offset,
            metadata: storage.metadata,
            metadata_upload: Vec::with_capacity(rows * METADATA_FIELDS),
            metadata_staging: PinnedUpload::single(device, rows * METADATA_FIELDS)?,
            info: storage.info,
            info_upload: Vec::with_capacity(segments * METADATA_FIELDS),
            info_staging: PinnedUpload::single(device, segments * METADATA_FIELDS)?,
            rotary: storage.rotary,
            rotary_upload: Vec::with_capacity(rows * ROTARY_AXES),
            rotary_staging: PinnedUpload::single(device, rows * ROTARY_AXES)?,
            logits,
            readbacks,
            _retained: retained,
            keep: Keep {
                arena: storage.arena,
                _metadata_rows: storage.metadata_rows,
                _info_rows: storage.info_rows,
                _rotary_rows: storage.rotary_rows,
                _prompt_row: storage.prompt_row,
                _external_row: storage.external_row,
                _external: storage.external,
                _lane_external: storage.lane_external,
                _lane_fusion: storage.lane_fusion,
                _last_rows: last_rows,
            },
        })
    }

    /// The rows narrower rungs view.
    pub(crate) const fn arena(&self) -> &ActivationArena {
        &self.keep.arena
    }

    /// Sequences this graph covers: the first slots of the pool.
    #[must_use]
    pub const fn segments(&self) -> usize {
        self.segments
    }

    /// Replay every lane's tokens; returns each lane's final logits in lane order.
    /// # Errors
    /// Rejects duplicate or oversized lanes, positions past the KV arena and CUDA failures.
    pub async fn run_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[Lane<'_>],
    ) -> Result<Vec<Vec<f32>>> {
        self.stage(device, lanes)?;
        let sources = [
            None,
            Some(ReadbackSource {
                tensor: Arc::clone(&self.logits),
                skip: 0,
                len: self.segments * self.vocabulary,
            }),
        ];
        let mut rows = self
            .readbacks
            .run_async(device, &self.graph, &sources)
            .await?;
        if let Some(profile) = self.profile.as_ref() {
            let tokens = lanes.iter().map(|lane| lane.tokens.len()).sum();
            profile.report(true, tokens, 0);
        }
        let logits = rows
            .pop()
            .ok_or_else(|| Error::invariant("prompt burst readback"))?;
        lanes
            .iter()
            .map(|lane| {
                let start = lane.segment * self.vocabulary;
                logits
                    .get(start..start + self.vocabulary)
                    .map(<[f32]>::to_vec)
                    .ok_or_else(|| Error::invariant("prompt burst logits row"))
            })
            .collect()
    }

    fn stage(&mut self, device: &CudaDevice, lanes: &[Lane<'_>]) -> Result<()> {
        let mut bound: Vec<Option<&Lane<'_>>> = vec![None; self.segments];
        for lane in lanes {
            let entry = bound
                .get_mut(lane.segment)
                .ok_or_else(|| Error::invalid("prompt burst segment"))?;
            if entry.is_some()
                || lane.tokens.is_empty()
                || lane.tokens.len() > SEGMENT_ROWS
                || lane.position + lane.tokens.len() > self.capacity
                || lane
                    .tokens
                    .iter()
                    .any(|&token| usize::try_from(token).map_or(true, |t| t >= self.vocabulary))
            {
                return Err(Error::invalid("prompt burst lane"));
            }
            *entry = Some(lane);
        }
        self.metadata_upload.clear();
        self.info_upload.clear();
        self.rotary_upload.clear();
        for lane in bound {
            let (tokens, position) = lane.map_or((&[][..], 0), |lane| (lane.tokens, lane.position));
            self.info_upload.extend_from_slice(&[
                i32::try_from(position).map_err(device_error)?,
                i32::try_from(tokens.len()).map_err(device_error)?,
                self.state_offset,
                0,
            ]);
            for row in 0..SEGMENT_ROWS {
                let rope = i32::try_from(position + row).map_err(device_error)?;
                let (token, state) =
                    tokens
                        .get(row)
                        .map_or(Ok((0, INACTIVE_STATE_POSITION)), |&token| {
                            i32::try_from(token)
                                .map(|token| (token, rope.saturating_add(self.state_offset)))
                                .map_err(device_error)
                        })?;
                self.metadata_upload
                    .extend_from_slice(&[rope, token, state, 0]);
                self.rotary_upload.extend_from_slice(&[rope; ROTARY_AXES]);
            }
        }
        self.metadata_staging
            .copy(device, &mut self.metadata, &self.metadata_upload)?;
        if let Some(rotary) = self.rotary.as_mut() {
            self.rotary_staging
                .copy(device, rotary, &self.rotary_upload)?;
        }
        self.info_staging
            .copy(device, &mut self.info, &self.info_upload)
    }
}
