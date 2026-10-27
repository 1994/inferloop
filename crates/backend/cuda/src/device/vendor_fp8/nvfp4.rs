//! Zero-workspace FP4 Lt candidates for sole-consumer fused-down scratch.
//! Shape decisions and full-output exact proofs are made before graph capture.
use super::{
    ALIGNMENT, COMPUTE_F32, Context, Descriptors, DeviceFuture, DeviceOp, ExecutionContext,
    GraphNode, MATRIX_LAYOUTS, OP_N, OP_T, Prepared, Shape, TRANSA, TRANSB, TYPE_F32, error,
    heuristic_candidates, same_context, set, status, with_default_device_policy,
};
use super::{LibraryIdentity, algorithm_config};
use crate::constants::NVFP4_GROUP_SIZE;
use crate::mlp::ProjectionWeight;
use crate::tuning::projection_budget::ProjectionKind;
use crate::tuning::resident_gemm::{self, GemmConfig, ResidentTuningSession};
use crate::tuning::{nvfp4_lt as cache, table_path};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
#[cfg(test)]
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};
use std::{collections::BTreeMap, ffi::c_void, ptr::null_mut, sync::Arc};
mod kernels;
mod memo;
mod timing;
pub use kernels::entries::probe_input;
pub use memo::ModelCache;
/// CUDA 13 `library_types.h`: packed E2M1 pairs.
const TYPE_FP4: i32 = 33;
/// CUDA 13 cublasLt.h: one UE4M3 scale per sixteen elements.
const VEC16_UE4M3: i32 = 1;
/// SDK descriptor attribute for the A scale device pointer.
const A_SCALE_POINTER: i32 = 17;
/// SDK descriptor attribute for the B scale device pointer.
const B_SCALE_POINTER: i32 = 18;
/// SDK descriptor attribute for the A scale mode.
const A_SCALE_MODE: i32 = 31;
/// SDK descriptor attribute for the B scale mode.
const B_SCALE_MODE: i32 = 32;
/// Packed E2M1 values per stored byte.
const CODES_PER_BYTE: usize = 2;
/// Contiguous rows within one SDK scale block.
pub const SCALE_ROWS: usize = 128;
/// K groups within one SDK scale block.
const SCALE_GROUPS: usize = 4;
/// Bytes within one SDK scale block.
const SCALE_BYTES: usize = SCALE_ROWS * SCALE_GROUPS;
type PlanKey = (Shape, u64);
type ProofKey = (PlanKey, u64, u64, u32, u32);
type WeightOwners = (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>);

pub struct Operands<'a> {
    pub(crate) weight: &'a ProjectionWeight,
    pub(crate) scale: f32,
    pub(crate) q: &'a Tensor<f4e2m1fnx2>,
    pub(crate) qs: &'a Tensor<f8e4m3fn>,
    /// Byte view of the unused F32 product; no allocation is owned by this view.
    pub(crate) scratch: &'a mut Tensor<f8e4m3fn>,
    pub(crate) output: &'a mut Tensor<f32>,
}
impl Operands<'_> {
    fn shape(&self) -> Result<Option<Shape>, DeviceError> {
        let ProjectionWeight::Fp4(w, s, global) = self.weight else {
            return Ok(None);
        };
        let ([m, half], [n, whalf], [om, on]) = (self.q.shape(), w.shape(), self.output.shape())
        else {
            return Ok(None);
        };
        if m != om || n != on || half != whalf || *m <= 0 || *n <= 0 || *half <= 0 {
            return Ok(None);
        }
        let shape = Shape {
            m: usize::try_from(*m).map_err(|e| error(e.to_string()))?,
            n: usize::try_from(*n).map_err(|e| error(e.to_string()))?,
            k: usize::try_from(*half)
                .map_err(|e| error(e.to_string()))?
                .checked_mul(CODES_PER_BYTE)
                .ok_or_else(|| error("FP4 K overflow"))?,
        };
        let groups = shape.k / NVFP4_GROUP_SIZE;
        let scales = shape
            .m
            .checked_add(shape.n)
            .and_then(|v| v.checked_mul(groups));
        if !shape.m.is_multiple_of(SCALE_ROWS)
            || !shape.n.is_multiple_of(SCALE_ROWS)
            || !shape.k.is_multiple_of(NVFP4_GROUP_SIZE * SCALE_GROUPS)
            || scales.is_none_or(|v| v > self.scratch.size())
            || self.qs.shape() != [*m, i32::try_from(groups).map_err(|e| error(e.to_string()))?]
            || s.shape() != [*n, i32::try_from(groups).map_err(|e| error(e.to_string()))?]
            || !self.scale.is_finite()
            || self.scale <= 0.0
            || !global.is_finite()
            || *global <= 0.0
            || !(1.0 / (self.scale * global)).is_finite()
        {
            return Ok(None);
        }
        let expected = [shape.m / CODES_PER_BYTE, groups * SCALE_ROWS];
        if self
            .scratch
            .shape()
            .iter()
            .zip(expected)
            .any(|(&a, b)| usize::try_from(a) != Ok(b))
            || self.scratch.shape().len() != expected.len()
        {
            return Ok(None);
        }
        if !self.q.is_contiguous()
            || !self.qs.is_contiguous()
            || !w.is_contiguous()
            || !s.is_contiguous()
            || !self.output.is_contiguous()
            || !self.scratch.is_contiguous()
        {
            return Ok(None);
        }
        let scratch = self.scratch.device_pointer().cu_deviceptr();
        let end = scratch
            .checked_add(u64::try_from(self.scratch.num_bytes()).map_err(|e| error(e.to_string()))?)
            .ok_or_else(|| error("FP4 scratch overflow"))?;
        for (pointer, bytes) in [
            (self.q.device_pointer().cu_deviceptr(), self.q.num_bytes()),
            (self.qs.device_pointer().cu_deviceptr(), self.qs.num_bytes()),
            (w.device_pointer().cu_deviceptr(), w.num_bytes()),
            (s.device_pointer().cu_deviceptr(), s.num_bytes()),
            (
                self.output.device_pointer().cu_deviceptr(),
                self.output.num_bytes(),
            ),
        ] {
            let other_end = pointer
                .checked_add(u64::try_from(bytes).map_err(|e| error(e.to_string()))?)
                .ok_or_else(|| error("FP4 operand overflow"))?;
            if scratch < other_end && pointer < end || !pointer.is_multiple_of(u64::from(ALIGNMENT))
            {
                return Ok(None);
            }
        }
        if !scratch.is_multiple_of(u64::from(ALIGNMENT)) {
            return Ok(None);
        }
        Ok(Some(shape))
    }
    fn proof(&self, shape: Shape) -> ProofKey {
        let (pointer, scales, global) = match self.weight {
            ProjectionWeight::Fp4(w, s, g) => (
                w.device_pointer().cu_deviceptr(),
                s.device_pointer().cu_deviceptr(),
                g.to_bits(),
            ),
            _ => (0, 0, 0),
        };
        (
            (shape, self.scratch.device_pointer().cu_deviceptr()),
            pointer,
            scales,
            global,
            self.scale.to_bits(),
        )
    }
}

/// Metadata-only admission. The caller initializes a numerical probe only after this is issued.
pub struct Candidate {
    context: Arc<Context>,
    prepared: Arc<Prepared>,
    shape: Shape,
    proof: ProofKey,
    native: GemmConfig,
    key: memo::DecisionKey,
    disk: Option<cache::Key>,
    timed: Option<bool>,
    numerical: Option<bool>,
}
impl Candidate {
    pub(crate) const fn needs_probe(&self) -> bool {
        self.numerical.is_none()
    }
}
enum Qualification {
    Cached,
    Search(Box<cache::Key>),
    Native,
}

/// Shares the already funded FP8 Lt handle; only host metadata and Arc owners are added.
pub struct Support {
    session: Arc<ResidentTuningSession>,
    autotune: bool,
    library: Option<LibraryIdentity>,
    model: Arc<ModelCache>,
    context: Option<Arc<Context>>,
    plans: BTreeMap<PlanKey, Option<Arc<Prepared>>>,
    proofs: BTreeMap<ProofKey, bool>,
    weights: BTreeMap<(u64, u64), WeightOwners>,
}
impl Support {
    pub(crate) fn new(
        shared: &super::Support,
        session: &Arc<ResidentTuningSession>,
        autotune: bool,
    ) -> Self {
        let context = shared.context.clone();
        #[cfg(test)]
        let context = (!NATIVE_ONLY.with(Cell::get)).then_some(context).flatten();
        Self {
            model: Arc::clone(&session.vendor_fp4),
            session: Arc::clone(session),
            autotune,
            library: shared.library_identity(),
            context,
            plans: BTreeMap::new(),
            proofs: BTreeMap::new(),
            weights: BTreeMap::new(),
        }
    }
    pub(crate) const fn enabled(&self) -> bool {
        self.context.is_some()
    }
    pub(crate) fn known(&self, op: &Operands<'_>) -> Result<Option<bool>, DeviceError> {
        if self.context.is_none() {
            return Ok(Some(false));
        }
        let Some(shape) = op.shape()? else {
            return Ok(Some(false));
        };
        let proof = op.proof(shape);
        if let Some(&accepted) = self.proofs.get(&proof) {
            return Ok(Some(accepted));
        }
        if self.plans.get(&proof.0).is_some_and(Option::is_none) {
            return Ok(Some(false));
        }
        Ok(None)
    }
    pub(crate) fn candidate(
        &mut self,
        op: &mut Operands<'_>,
        native: GemmConfig,
    ) -> Result<Option<Candidate>, DeviceError> {
        let Some(context) = self.context.clone() else {
            return Ok(None);
        };
        let Some(shape) = op.shape()? else {
            return Ok(None);
        };
        let proof = op.proof(shape);
        if self.plans.get(&proof.0).is_some_and(Option::is_none) {
            return Ok(None);
        }
        context
            .device
            .reclaim_barrier()
            .map_err(|e| error(e.to_string()))?;
        let ProjectionWeight::Fp4(w, s, _) = op.weight else {
            return Ok(None);
        };
        for pointer in [
            op.q.device_pointer().cu_deviceptr(),
            op.qs.device_pointer().cu_deviceptr(),
            w.device_pointer().cu_deviceptr(),
            s.device_pointer().cu_deviceptr(),
            op.scratch.device_pointer().cu_deviceptr(),
            op.output.device_pointer().cu_deviceptr(),
        ] {
            same_context(&context, pointer)?;
        }
        let Some(prepared) = (match self.plans.get(&proof.0).cloned().flatten() {
            Some(plan) => Some(plan),
            None => match make_candidate(&context, shape, op.scratch) {
                Ok(candidate) => candidate,
                Err(reason) => {
                    tracing::warn!(target:"infer::tuning",?shape,%reason,"vendor FP4 metadata unavailable; retain native");
                    None
                }
            },
        }) else {
            self.plans.insert(proof.0, None);
            return Ok(None);
        };
        let key = memo::key(&context, &prepared, shape, native);
        let (mut timed, numerical) = self.model.qualification(key, op)?;
        if timed == Some(false) || numerical == Some(false) {
            return Ok(None);
        }
        let disk = if timed.is_none() {
            match self.qualification(&context, op, &prepared, shape, native) {
                Qualification::Cached => {
                    timed = Some(self.model.save_timing(key, true)?);
                    None
                }
                Qualification::Search(key) => Some(*key),
                Qualification::Native => {
                    self.plans.insert(proof.0, None);
                    return Ok(None);
                }
            }
        } else {
            None
        };
        Ok(Some(Candidate {
            context,
            prepared,
            shape,
            proof,
            native,
            key,
            disk,
            timed,
            numerical,
        }))
    }
    fn qualification(
        &self,
        context: &Context,
        op: &mut Operands<'_>,
        prepared: &Prepared,
        shape: Shape,
        native: GemmConfig,
    ) -> Qualification {
        let Some(library) = self.library else {
            return Qualification::Native;
        };
        let key = match disk_key(context, op, prepared, shape, native, library) {
            Ok(key) => key,
            Err(reason) => {
                tracing::warn!(target:"infer::tuning",?shape,%reason,"vendor FP4 qualification identity unavailable");
                return Qualification::Native;
            }
        };
        if let Err(reason) = self.session.plan_projection(
            ProjectionKind::Fp4Lt,
            [shape.m, shape.n, shape.k],
            timing::SEARCH_TRIALS,
        ) {
            tracing::warn!(target:"infer::tuning",%reason,"vendor FP4 workload unavailable");
            return Qualification::Native;
        }
        let path = table_path();
        match cache::cached(path.as_deref(), &key) {
            Ok(true) => {
                tracing::info!(target:"infer::tuning",?shape,source="table-cache","vendor FP4 qualification selected");
                Qualification::Cached
            }
            Ok(false) if self.autotune && self.session.permits_search() => {
                Qualification::Search(Box::new(key))
            }
            Ok(false) => Qualification::Native,
            Err(reason) => {
                tracing::warn!(target:"infer::tuning",?shape,%reason,"vendor FP4 cache rejected");
                Qualification::Native
            }
        }
    }
    pub(crate) fn prepare(
        &mut self,
        mut op: Operands<'_>,
        candidate: Candidate,
    ) -> Result<bool, DeviceError> {
        let Candidate {
            context,
            prepared,
            shape,
            proof,
            native,
            key,
            disk,
            timed,
            numerical,
        } = candidate;
        if op.shape()? != Some(shape) || op.proof(shape) != proof {
            return Err(error("vendor FP4 preparation operands changed"));
        }
        let cached = self.plans.get(&proof.0).cloned().flatten();
        let graph = if cached.is_none() || numerical.is_none() {
            Some(capture(&context, &mut op, native, Some(&prepared), shape)?)
        } else {
            None
        };
        if numerical.is_none() {
            let baseline = capture(&context, &mut op, native, None, shape)?;
            let graph = graph
                .as_ref()
                .ok_or_else(|| error("FP4 proof candidate missing"))?;
            if !exact(&context, &baseline, graph, op.output)? {
                self.session
                    .complete_projection(ProjectionKind::Fp4Lt, [shape.m, shape.n, shape.k])
                    .map_err(|e| error(e.to_string()))?;
                self.model.save_numerical(key, &op, false)?;
                self.proofs.insert(proof, false);
                return Ok(false);
            }
            if timed.is_none() {
                let _search = self
                    .session
                    .projection_search()
                    .map_err(|e| error(e.to_string()))?;
                if !self.session.permits_search() {
                    return Ok(false);
                }
                let fresh_a = capture(&context, &mut op, native, None, shape)?;
                let fresh_b = capture(&context, &mut op, native, Some(&prepared), shape)?;
                let Some(trials) =
                    timing::confirm(&context.device, &baseline, graph, &fresh_a, &fresh_b, shape)?
                else {
                    self.session
                        .complete_projection(ProjectionKind::Fp4Lt, [shape.m, shape.n, shape.k])
                        .map_err(|e| error(e.to_string()))?;
                    self.model.save_timing(key, false)?;
                    self.plans.insert(proof.0, None);
                    return Ok(false);
                };
                if !self.model.save_timing(key, true)? {
                    return Ok(false);
                }
                if let Some(key) = disk {
                    let path = table_path();
                    let saved = cache::save(path.as_deref(), cache::Entry { key, trials });
                    tracing::info!(target:"infer::tuning",?shape,saved,"vendor FP4 qualification recorded");
                }
            }
            self.model.save_numerical(key, &op, true)?;
        } else if let Some(graph) = &graph {
            graph.launch().sync_on(&context.device.stream)?;
        }
        let ProjectionWeight::Fp4(w, s, _) = op.weight else {
            return Ok(false);
        };
        self.session
            .complete_projection(ProjectionKind::Fp4Lt, [shape.m, shape.n, shape.k])
            .map_err(|e| error(e.to_string()))?;
        self.plans.insert(proof.0, Some(prepared));
        self.proofs.insert(proof, true);
        self.weights
            .entry((proof.1, proof.2))
            .or_insert_with(|| (Arc::clone(w), Arc::clone(s)));
        tracing::info!(target:"infer::tuning",?shape,cached_proof=numerical.is_some(),
            "vendor FP4 complete exact proof accepted");
        Ok(true)
    }
    /// False records nothing. Errors after append abort the containing capture.
    pub(crate) fn record(&self, scope: &Scope, mut op: Operands<'_>) -> Result<bool, DeviceError> {
        let Some(context) = &self.context else {
            return Ok(false);
        };
        let Some(shape) = op.shape()? else {
            return Ok(false);
        };
        let proof = op.proof(shape);
        if self.proofs.get(&proof) != Some(&true) {
            return Ok(false);
        }
        let Some(Some(prepared)) = self.plans.get(&proof.0) else {
            return Ok(false);
        };
        let ProjectionWeight::Fp4(w, s, _) = op.weight else {
            return Ok(false);
        };
        for pointer in [
            op.q.device_pointer().cu_deviceptr(),
            op.qs.device_pointer().cu_deviceptr(),
            w.device_pointer().cu_deviceptr(),
            s.device_pointer().cu_deviceptr(),
            op.output.device_pointer().cu_deviceptr(),
            op.scratch.device_pointer().cu_deviceptr(),
        ] {
            same_context(context, pointer)?;
        }
        append(scope, context, prepared, &mut op, shape)?;
        #[cfg(test)]
        CAPTURES.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }
}

fn disk_key(
    context: &Context,
    op: &mut Operands<'_>,
    prepared: &Prepared,
    shape: Shape,
    native: GemmConfig,
    library: LibraryIdentity,
) -> Result<cache::Key, DeviceError> {
    let ProjectionWeight::Fp4(_, s, _) = op.weight else {
        return Err(error("FP4 weight identity"));
    };
    let pack_fingerprint = kernels::pack_pair(
        (&mut *op.scratch).partition_prefix([1, SCALE_BYTES]),
        s,
        op.qs,
    )
    .generics(vec![shape.n.to_string()])
    .grid((
        u32::try_from((shape.n + shape.m) / SCALE_ROWS).map_err(|e| error(e.to_string()))?,
        u32::try_from(shape.k / NVFP4_GROUP_SIZE / SCALE_GROUPS)
            .map_err(|e| error(e.to_string()))?,
        1,
    ))
    .l2_cache_key_on(&context.device.stream)?;
    Ok(cache::Key {
        native: resident_gemm::cache_key(&context.device, [shape.m, shape.n, shape.k])
            .map_err(|e| error(e.to_string()))?,
        baseline: native,
        library,
        algorithm: algorithm_config(context.api, &prepared.algo)?,
        pack_fingerprint,
        schema: cache::SCHEMA.into(),
    })
}

fn exact(
    context: &Context,
    baseline: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    out: &Tensor<f32>,
) -> Result<bool, DeviceError> {
    baseline.launch().sync_on(&context.device.stream)?;
    let expected = context
        .device
        .read_borrowed(out)
        .map_err(|e| error(e.to_string()))?;
    candidate.launch().sync_on(&context.device.stream)?;
    let actual = context
        .device
        .read_borrowed(out)
        .map_err(|e| error(e.to_string()))?;
    Ok(expected.len() == actual.len()
        && expected
            .iter()
            .zip(&actual)
            .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits()))
}

fn capture(
    context: &Arc<Context>,
    op: &mut Operands<'_>,
    native: GemmConfig,
    prepared: Option<&Arc<Prepared>>,
    shape: Shape,
) -> Result<CudaGraph<()>, DeviceError> {
    CudaGraph::scope(&context.device.stream, |scope| {
        if let Some(prepared) = prepared {
            append(scope, context, prepared, op, shape)
        } else {
            native.record(scope, op.weight, op.scale, op.q, op.qs, op.output, shape.k)
        }
    })
}
fn append(
    scope: &Scope,
    context: &Arc<Context>,
    prepared: &Arc<Prepared>,
    op: &mut Operands<'_>,
    shape: Shape,
) -> Result<(), DeviceError> {
    let ProjectionWeight::Fp4(w, s, g) = op.weight else {
        return Err(error("FP4 weight"));
    };
    scope.record(
        kernels::pack_pair(
            (&mut *op.scratch).partition_prefix([1, SCALE_BYTES]),
            s,
            op.qs,
        )
        .generics(vec![shape.n.to_string()])
        .grid((
            u32::try_from((shape.n + shape.m) / SCALE_ROWS).map_err(|e| error(e.to_string()))?,
            u32::try_from(shape.k / NVFP4_GROUP_SIZE / SCALE_GROUPS)
                .map_err(|e| error(e.to_string()))?,
            1,
        )),
    )?;
    scope.record(Fp4Node {
        context: context.clone(),
        prepared: prepared.clone(),
        input: op.q,
        weight: w,
        output: op.output,
        scratch: op.scratch,
        alpha: 1.0 / (g * op.scale),
    })?;
    Ok(())
}

#[expect(
    unsafe_code,
    reason = "Owned descriptors use the documented CUDA 13 FP4/F32 and scale-pointer ABI"
)]
fn make_candidate(
    context: &Context,
    shape: Shape,
    scratch: &Tensor<f8e4m3fn>,
) -> Result<Option<Arc<Prepared>>, DeviceError> {
    let api = context.api;
    let _guard = context
        .calls
        .lock()
        .map_err(|_| error("vendor FP4 heuristic mutex"))?;
    let mut d = Descriptors {
        api,
        desc: null_mut(),
        layouts: [null_mut(); MATRIX_LAYOUTS],
    };
    // SAFETY: Handles are uniquely owned by d and its Drop cleans up partial construction.
    status(
        // SAFETY: d uniquely owns the output handle slot and cleans up partial construction.
        unsafe { (api.desc_create)(&raw mut d.desc, COMPUTE_F32, TYPE_F32) },
        "FP4 descriptor",
    )?;
    set(api, d.desc, TRANSA, OP_T)?;
    set(api, d.desc, TRANSB, OP_N)?;
    set(api, d.desc, A_SCALE_MODE, VEC16_UE4M3)?;
    set(api, d.desc, B_SCALE_MODE, VEC16_UE4M3)?;
    let input_offset = shape
        .n
        .checked_mul(shape.k / NVFP4_GROUP_SIZE)
        .ok_or_else(|| error("FP4 SF offset"))?;
    let weight_pointer = scratch.device_pointer().cu_deviceptr();
    let input_pointer = weight_pointer
        .checked_add(u64::try_from(input_offset).map_err(|e| error(e.to_string()))?)
        .ok_or_else(|| error("FP4 SF pointer overflow"))?;
    set(
        api,
        d.desc,
        A_SCALE_POINTER,
        weight_pointer as *const c_void,
    )?;
    set(api, d.desc, B_SCALE_POINTER, input_pointer as *const c_void)?;
    for (i, (dtype, r, c, ld)) in [
        (TYPE_FP4, shape.k, shape.n, shape.k),
        (TYPE_FP4, shape.k, shape.m, shape.k),
        (TYPE_F32, shape.n, shape.m, shape.n),
        (TYPE_F32, shape.n, shape.m, shape.n),
    ]
    .into_iter()
    .enumerate()
    {
        // SAFETY: Checked dimensions describe complete contiguous operands validated by Operands::shape.
        status(
            // SAFETY: Checked extents and exact SDK types match the complete live allocations.
            unsafe {
                (api.layout_create)(
                    &raw mut d.layouts[i],
                    dtype,
                    u64::try_from(r).map_err(|e| error(e.to_string()))?,
                    u64::try_from(c).map_err(|e| error(e.to_string()))?,
                    i64::try_from(ld).map_err(|e| error(e.to_string()))?,
                )
            },
            "FP4 layout",
        )?;
    }
    let d = Arc::new(d);
    Ok(heuristic_candidates(context, &d)?
        .into_iter()
        .find(|r| r.state == 0 && r.workspace == 0)
        .map(|r| {
            Arc::new(Prepared {
                descriptors: d,
                algo: r.algo,
            })
        }))
}
struct Fp4Node<'a> {
    context: Arc<Context>,
    prepared: Arc<Prepared>,
    input: &'a Tensor<f4e2m1fnx2>,
    weight: &'a Tensor<f4e2m1fnx2>,
    output: &'a Tensor<f32>,
    scratch: &'a Tensor<f8e4m3fn>,
    alpha: f32,
}
impl GraphNode for Fp4Node<'_> {}
#[expect(
    unsafe_code,
    reason = "Validated live FP4/F32 operands, owned immutable descriptors and serialized zero-workspace Lt capture"
)]
impl DeviceOp for Fp4Node<'_> {
    type Output = ();
    unsafe fn execute(self, execution: &ExecutionContext) -> Result<(), DeviceError> {
        if execution.get_cuda_stream().cu_stream() != self.context.stream {
            return Err(error("vendor FP4 stream mismatch"));
        }
        same_context(&self.context, self.scratch.device_pointer().cu_deviceptr())?;
        execution.retain(self.context.clone())?;
        execution.retain(self.prepared.clone())?;
        let _guard = self
            .context
            .calls
            .lock()
            .map_err(|_| error("vendor FP4 handle mutex"))?;
        let beta = 0.0_f32;
        // SAFETY: prepare checks context, shape, exact output and zero workspace; operands and scratch
        // are retained by the program/arena beyond every replay. Lt copies these host scalars into capture.
        status(
            unsafe {
                (self.context.api.matmul)(
                    self.context.handle,
                    self.prepared.desc,
                    (&raw const self.alpha).cast(),
                    self.weight.device_pointer().cu_deviceptr() as *const c_void,
                    self.prepared.layouts[0],
                    self.input.device_pointer().cu_deviceptr() as *const c_void,
                    self.prepared.layouts[1],
                    (&raw const beta).cast(),
                    self.output.device_pointer().cu_deviceptr() as *const c_void,
                    self.prepared.layouts[2],
                    self.output.device_pointer().cu_deviceptr() as *mut c_void,
                    self.prepared.layouts[3],
                    &raw const self.prepared.algo,
                    null_mut(),
                    0,
                    self.context.stream,
                )
            },
            "FP4 matmul",
        )
    }
}
impl IntoFuture for Fp4Node<'_> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|p| p.next_stream()) {
            Ok(Ok(s)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(s)),
            Ok(Err(e)) | Err(e) => Self::IntoFuture::failed(e),
        }
    }
}

#[cfg(test)]
thread_local! {
    static NATIVE_ONLY: Cell<bool> = const {Cell::new(false)};
}
#[cfg(test)]
static CAPTURES: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.load(Ordering::Relaxed)
}
#[cfg(test)]
pub fn with_native<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            NATIVE_ONLY.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(NATIVE_ONLY.with(|flag| flag.replace(true)));
    run()
}

#[cfg(test)]
#[path = "../../../tests/unit/vendor_nvfp4_numerics.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/vendor_nvfp4.rs"]
mod rotation_tests;

#[cfg(test)]
static CONFIRMATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub fn confirmations() -> usize {
    CONFIRMATIONS.load(Ordering::Relaxed)
}
