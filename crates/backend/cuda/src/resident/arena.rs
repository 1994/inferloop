use crate::constants::F32_BYTES;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::device::vendor_fp8::nvfp4::SCALE_ROWS;
use crate::device::{CudaDevice, device_error};
use cuda_core::f8e4m3fn;
use cuda_core::sys::CUdeviceptr;
use cutile::{cuda_async::device_buffer::DeviceAllocation, prelude::*};
use infer_core::{Error, Result, TensorId};
use infer_ir::DataflowGraph;
use std::collections::BTreeMap;
use std::slice::from_ref;
use std::sync::Arc;

/// Slot layout of one graph: `last_node` and element count per reused slot.
type ArenaLayout = (Vec<(usize, usize)>, BTreeMap<TensorId, usize>);

/// Reuses equal-sized device buffers after their last consumer.
///
/// Graph results stay live until readback; cuTile mutable ownership remains exclusive.
/// Batched arenas flatten lanes into one axis: every slot is `[lanes * size]`.
pub struct ActivationArena {
    indices: BTreeMap<TensorId, usize>,
    pub(crate) buffers: Vec<Option<Tensor<f32>>>,
    lanes: usize,
    elements: usize,
}

impl ActivationArena {
    /// # Errors
    /// Rejects invalid graphs, overflowing budgets and failed device allocation.
    pub fn new(device: &CudaDevice, graph: &DataflowGraph, byte_budget: usize) -> Result<Self> {
        Self::plan(device, graph, 1, byte_budget)
    }

    /// One slot per plan entry holding `lanes` concatenated rows: activations of
    /// a whole prompt batch, laid out so row `lane` starts at `lane * size`.
    /// # Errors
    /// Rejects zero lanes, invalid graphs, overflowing budgets and failed device allocation.
    pub fn new_batched(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::invalid("batched activation lanes"));
        }
        Self::plan(device, graph, lanes, byte_budget)
    }

    /// Slot layout after last-consumer reuse, with the tensor-to-slot mapping. A `single`
    /// tensor gets a private slot that is never shared.
    fn layout(graph: &DataflowGraph, single: Option<TensorId>) -> Result<ArenaLayout> {
        let mut planned = graph.clone();
        planned.plan_lifetimes()?;
        let mut slots: Vec<(usize, usize)> = Vec::new();
        let mut indices = BTreeMap::new();
        for lifetime in &planned.lifetimes {
            if Some(lifetime.tensor) == single {
                slots.push((usize::MAX, lifetime.elements));
                indices.insert(lifetime.tensor, slots.len() - 1);
                continue;
            }
            let slot = slots
                .iter()
                .position(|(last, size)| *last < lifetime.first_node && *size == lifetime.elements)
                .unwrap_or_else(|| {
                    slots.push((0, lifetime.elements));
                    slots.len() - 1
                });
            slots[slot].0 = lifetime.last_node;
            indices.insert(lifetime.tensor, slot);
        }
        Ok((slots, indices))
    }

    /// Elements of every slot at `lanes` rows; the `single` tensor's slot holds one row.
    fn sizes(
        graph: &DataflowGraph,
        lanes: usize,
        single: Option<TensorId>,
    ) -> Result<(Vec<usize>, BTreeMap<TensorId, usize>)> {
        let (slots, indices) = Self::layout(graph, single)?;
        let single_slot = single.and_then(|id| indices.get(&id).copied());
        let sizes = slots
            .iter()
            .enumerate()
            .map(|(index, (_, size))| {
                if Some(index) == single_slot {
                    Some(*size)
                } else {
                    size.checked_mul(lanes)
                }
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::invalid("activation size overflow"))?;
        Ok((sizes, indices))
    }

    /// Device bytes this graph's activations occupy at `lanes` rows, without allocating.
    /// # Errors
    /// Rejects invalid graphs or overflowing sizes.
    pub fn required_bytes(graph: &DataflowGraph, lanes: usize) -> Result<usize> {
        Self::bytes_for(graph, lanes, None)
    }

    /// [`Self::required_bytes`] for a prompt arena that keeps one row of logits.
    /// # Errors
    /// Rejects invalid graphs or overflowing sizes.
    pub fn required_prompt_bytes(graph: &DataflowGraph, lanes: usize) -> Result<usize> {
        Self::bytes_for(graph, lanes, graph.logits)
    }

    fn bytes_for(graph: &DataflowGraph, lanes: usize, single: Option<TensorId>) -> Result<usize> {
        Self::sizes(graph, lanes, single)?
            .0
            .iter()
            .try_fold(0usize, |sum, size| sum.checked_add(*size))
            .and_then(|elements| elements.checked_mul(F32_BYTES))
            .ok_or_else(|| Error::invalid("activation size overflow"))
    }

    /// One lane of this batched arena as a single-row arena whose slots alias that lane's rows,
    /// so per-lane captures and whole-slot captures of the batched arena share storage.
    ///
    /// # Safety
    /// The returned arena does not own its storage: `self` must outlive every use of it and of
    /// every graph captured against it, and a lane's rows must not be used as concurrent mutable
    /// outputs together with the whole slot.
    /// # Errors
    /// Rejects a lane outside the arena or a slot that is currently taken.
    #[expect(
        unsafe_code,
        reason = "Audited lane alias: the batched owner outlives the views by the caller's contract"
    )]
    pub(super) unsafe fn lane_view(&self, lane: usize) -> Result<Self> {
        if lane >= self.lanes {
            return Err(Error::invalid("activation lane out of bounds"));
        }
        let mut buffers = Vec::with_capacity(self.buffers.len());
        for buffer in &self.buffers {
            let slot = buffer
                .as_ref()
                .ok_or_else(|| Error::invariant("batched slot is currently an output"))?;
            // SAFETY: forwarded from this function's contract; the row lies inside the slot.
            buffers.push(Some(unsafe { row_view(slot, self.lanes, lane)? }));
        }
        Ok(Self {
            indices: self.indices.clone(),
            buffers,
            lanes: 1,
            elements: self.elements / self.lanes,
        })
    }

    /// An arena whose slots alias this one's storage with the same shapes.
    ///
    /// Prompt graphs of different sequences capture against views of one shared arena, so the
    /// activation memory is paid once per prompt width instead of once per admitted sequence.
    /// # Safety
    /// `self` must outlive every graph captured through the view, replays through any two
    /// views must be serialized on one stream, and no activation may be read after another
    /// view's replay has started.
    /// # Errors
    /// Rejects an arena with a slot currently taken out as an output.
    #[expect(
        unsafe_code,
        reason = "Audited arena alias: the shared owner outlives the views by the caller's contract"
    )]
    pub(super) unsafe fn shared_view(&self) -> Result<Self> {
        let mut buffers = Vec::with_capacity(self.buffers.len());
        for buffer in &self.buffers {
            let slot = buffer
                .as_ref()
                .ok_or_else(|| Error::invariant("shared arena slot is currently an output"))?;
            let shape = slot
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(device_error))
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: forwarded from this function's contract; one lane spans the whole slot.
            let view = unsafe { row_view(slot, 1, 0)? };
            buffers.push(Some(view.reshape(&shape).map_err(device_error)?));
        }
        Ok(Self {
            indices: self.indices.clone(),
            buffers,
            lanes: self.lanes,
            elements: self.elements,
        })
    }

    /// Batched prompt arena whose logits slot holds only the final row: prompt graphs project
    /// just the last active lane onto the vocabulary.
    /// # Errors
    /// Rejects zero lanes, invalid graphs, overflowing budgets and failed device allocation.
    pub fn new_prompt(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::invalid("batched activation lanes"));
        }
        Self::plan_with(device, graph, lanes, byte_budget, graph.logits)
    }

    fn plan(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
    ) -> Result<Self> {
        Self::plan_with(device, graph, lanes, byte_budget, None)
    }

    fn plan_with(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
        single: Option<TensorId>,
    ) -> Result<Self> {
        let (sizes, indices) = Self::sizes(graph, lanes, single)?;
        let elements = sizes
            .iter()
            .try_fold(0usize, |sum, size| sum.checked_add(*size))
            .ok_or_else(|| Error::invalid("activation size overflow"))?;
        if elements
            .checked_mul(F32_BYTES)
            .is_none_or(|bytes| bytes > byte_budget)
        {
            return Err(Error::invalid("device activation budget exceeded"));
        }
        let mut buffers = Vec::with_capacity(sizes.len());
        for size in sizes {
            buffers.push(Some(
                api::zeros::<f32>(&[size])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ));
        }
        Ok(Self {
            indices,
            buffers,
            lanes,
            elements,
        })
    }

    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.elements * F32_BYTES
    }

    #[must_use]
    pub(crate) const fn lanes(&self) -> usize {
        self.lanes
    }

    /// Resolve once during graph preparation, not during token execution.
    /// # Errors
    /// Rejects tensors which are not graph activations.
    pub fn slot(&self, id: TensorId) -> Result<usize> {
        self.indices
            .get(&id)
            .copied()
            .ok_or_else(|| Error::invalid("unknown device activation"))
    }

    pub(crate) fn get(&self, id: TensorId) -> Result<&Tensor<f32>> {
        self.buffers[self.slot(id)?]
            .as_ref()
            .ok_or_else(|| Error::invariant("activation is currently an output"))
    }

    /// Flat view of one lane inside a batched slot.
    /// # Errors
    /// Rejects non-batched arenas and out-of-range lanes.
    pub(crate) fn row(&self, id: TensorId, lane: usize) -> Result<TensorView<'_, f32>> {
        let tensor = self.get(id)?;
        if self.lanes == 1 || !tensor.size().is_multiple_of(self.lanes) {
            return Err(Error::invalid("activation slot is not batched"));
        }
        let size = tensor.size() / self.lanes;
        if lane >= self.lanes {
            return Err(Error::invalid("activation lane out of bounds"));
        }
        let range = lane * size..(lane + 1) * size;
        tensor.slice(from_ref(&range)).map_err(device_error)
    }
}

/// `DeviceAllocation` view of one row inside a batched slot. Carries no owner:
/// the slot itself is kept alive by the arena stored in `BatchGraph`, which
/// outlives every captured graph that references these rows.
struct RowAllocation {
    pointer: CUdeviceptr,
    bytes: usize,
    device: usize,
}

#[expect(
    unsafe_code,
    reason = "Audited row aliasing: the batched arena outlives its captured graphs; \
              row extents are checked against the slot at construction; kernels on one \
              stream order every row access"
)]
// SAFETY: getters are stable for the token's lifetime; `pointer`/`bytes` describe a
// row range verified against the slot's own extent at construction in `row_tensor`.
unsafe impl DeviceAllocation for RowAllocation {
    fn device_ptr(&self) -> CUdeviceptr {
        self.pointer
    }

    fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn device_id(&self) -> usize {
        self.device
    }
}

/// Borrow one lane of a batched tensor without taking ownership of the slot.
///
/// Used where the owner already outlives the capture (metadata rows and the batched
/// fusion gather). The caller must keep `slot` alive for every use of the view and
/// must not use two views of the same lane as concurrent mutable outputs.
///
/// # Safety
/// The returned tensor aliases `slot`'s device memory; its lifetime is not tracked.
#[expect(
    unsafe_code,
    reason = "Audited lane view: the owner is retained by the caller for the graph lifetime, \
              the lane extent is checked against the slot, and same-stream replay serializes access"
)]
pub(super) unsafe fn row_view<T: DType>(
    slot: &Tensor<T>,
    lanes: usize,
    lane: usize,
) -> Result<Tensor<T>> {
    if lanes == 0 || !slot.size().is_multiple_of(lanes) {
        return Err(Error::invalid("activation slot is not batched"));
    }
    let size = slot.size() / lanes;
    if lane >= lanes {
        return Err(Error::invalid("activation lane out of bounds"));
    }
    let stride = size_of::<T>();
    let offset = lane
        .checked_mul(size)
        .and_then(|row| row.checked_mul(stride))
        .ok_or_else(|| Error::invalid("activation row offset overflow"))?;
    let total = slot.num_bytes();
    let allocation = RowAllocation {
        pointer: slot.device_pointer().cu_deviceptr()
            + u64::try_from(offset).map_err(device_error)?,
        bytes: total
            .checked_sub(offset)
            .ok_or_else(|| Error::invalid("activation row extent"))?,
        device: slot.device_id(),
    };
    // SAFETY: the owner is retained by the caller; the row lies fully inside the slot by
    // construction, and the caller serializes disjoint lane access on one stream.
    Ok(unsafe {
        Tensor::from_foreign(
            Arc::new(allocation),
            vec![i32::try_from(size).map_err(device_error)?],
            vec![1],
        )
    })
}

/// Borrow the first `kept` of `rows` leading-axis rows of a tensor, reshaped to `shape`.
///
/// Used by narrower pooled graphs that address only the low slots of a pool-wide state
/// family. The caller must keep `whole` alive for every use of the view.
///
/// # Safety
/// The returned tensor aliases `whole`'s device memory; its lifetime is not tracked.
#[expect(
    unsafe_code,
    reason = "Audited prefix view: the pool retains the owner for the graph lifetime, the               prefix extent is checked against it, and same-stream replay serializes access"
)]
pub(super) unsafe fn prefix_view<T: DType>(
    whole: &Tensor<T>,
    rows: usize,
    kept: usize,
    shape: &[usize],
) -> Result<Tensor<T>> {
    if rows == 0 || kept == 0 || kept > rows || !whole.size().is_multiple_of(rows) {
        return Err(Error::invalid("prefix view of a non-batched tensor"));
    }
    let elements = whole.size() / rows * kept;
    if shape.iter().product::<usize>() != elements {
        return Err(Error::invalid("prefix view shape"));
    }
    let allocation = RowAllocation {
        pointer: whole.device_pointer().cu_deviceptr(),
        bytes: elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("prefix view extent"))?,
        device: whole.device_id(),
    };
    let dims = shape
        .iter()
        .map(|&dim| i32::try_from(dim).map_err(device_error))
        .collect::<Result<Vec<_>>>()?;
    let mut strides = vec![1i32; dims.len()];
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1]
            .checked_mul(dims[axis + 1])
            .ok_or_else(|| Error::invalid("prefix view strides"))?;
    }
    // SAFETY: the owner is retained by the caller and the prefix lies inside it.
    Ok(unsafe { Tensor::from_foreign(Arc::new(allocation), dims, strides) })
}

/// `DeviceAllocation` view of one batched lane that also owns the slot storage.
///
/// Unlike [`RowAllocation`], this row carries a shared handle to the whole slot, so it
/// stays valid after the caller drops or replaces its own tensor handle.
struct OwnedRowAllocation<T: DType> {
    storage: Arc<Tensor<T>>,
    pointer: CUdeviceptr,
    bytes: usize,
}

#[expect(
    unsafe_code,
    reason = "Audited owned row alias: `storage` keeps the slot allocation alive, the lane \
              is extent-checked against it, and only one stream orders row accesses"
)]
// SAFETY: getters are stable for the token's lifetime; `pointer`/`bytes` describe a
// row range verified against the slot's own extent at construction in `batch_rows`.
unsafe impl<T: DType> DeviceAllocation for OwnedRowAllocation<T> {
    fn device_ptr(&self) -> CUdeviceptr {
        self.pointer
    }

    fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn device_id(&self) -> usize {
        self.storage.device_id()
    }
}

/// Own every disjoint lane of a batched tensor as a row view.
///
/// Each returned row holds a shared handle to the slot's storage, so dropping the caller's
/// tensor handle does not invalidate the row. Lanes are non-overlapping and remain exclusive
/// outputs; one device stream serializes uploads and graph replays that reference them.
///
/// # Errors
/// Rejects malformed batched slots, zero lanes and offset overflows.
#[expect(
    unsafe_code,
    reason = "Audited batched ownership: every row holds the slot's storage and is bounded by \
              the slot extent; lanes are disjoint and one stream serializes access"
)]
pub(super) fn batch_rows<T: DType>(owner: &Tensor<T>, lanes: usize) -> Result<Vec<Tensor<T>>> {
    if lanes == 0 {
        return Err(Error::invalid("batched activation lanes"));
    }
    if !owner.size().is_multiple_of(lanes) {
        return Err(Error::invalid("activation slot is not batched"));
    }
    let size = owner.size() / lanes;
    let bytes = size
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::invalid("activation row extent"))?;
    // SAFETY: the shared alias is kept by every row, so the slot storage outlives all of
    // them; `batch_rows` never writes through the alias and row uses stay on one stream.
    let storage = unsafe { owner.into_shared_alias() };
    (0..lanes)
        .map(|lane| {
            let offset = lane
                .checked_mul(bytes)
                .ok_or_else(|| Error::invalid("activation row offset overflow"))?;
            let allocation = OwnedRowAllocation {
                storage: Arc::clone(&storage),
                pointer: owner.device_pointer().cu_deviceptr()
                    + u64::try_from(offset).map_err(device_error)?,
                bytes,
            };
            // SAFETY: `allocation` owns the slot storage, and this lane lies fully inside
            // the slot by construction. Disjoint row writes are serialized on one stream.
            Ok(unsafe {
                Tensor::from_foreign(
                    Arc::new(allocation),
                    vec![i32::try_from(size).map_err(device_error)?],
                    vec![1],
                )
            })
        })
        .collect()
}

/// Owned tensor covering row `lane` of a taken batched slot; dropping it frees
/// nothing. The returned Arc restores the slot once every row tensor is dropped.
/// # Errors
/// Rejects malformed slots, out-of-range lanes and offset overflows.
#[expect(
    unsafe_code,
    reason = "Audited from_foreign row view: the caller re-anchors the slot in its owning \
              arena or workspace before any launch; that owner outlives its graphs; \
              construction bounds the row; same-stream ordering serializes accesses"
)]
pub fn row_tensor(
    slot: Tensor<f32>,
    lanes: usize,
    lane: usize,
) -> Result<(Tensor<f32>, Arc<Tensor<f32>>)> {
    if lanes == 0 || !slot.size().is_multiple_of(lanes) {
        return Err(Error::invalid("activation slot is not batched"));
    }
    let size = slot.size() / lanes;
    if lane >= lanes {
        return Err(Error::invalid("activation lane out of bounds"));
    }
    let offset = lane
        .checked_mul(size)
        .and_then(|row| row.checked_mul(F32_BYTES))
        .ok_or_else(|| Error::invalid("activation row offset overflow"))?;
    let shared = Arc::new(slot);
    let allocation = RowAllocation {
        pointer: shared.device_pointer().cu_deviceptr()
            + u64::try_from(offset).map_err(device_error)?,
        bytes: shared.num_bytes() - offset,
        device: shared.device_id(),
    };
    // SAFETY: the slot outlives every graph replay through the arena anchor; the row
    // lies fully inside it by construction; same-stream replay serializes row access.
    let row = unsafe {
        Tensor::from_foreign(
            Arc::new(allocation),
            vec![i32::try_from(size).map_err(device_error)?],
            vec![1],
        )
    };
    Ok((row, shared))
}

/// Two metadata views under one exclusive lease of an unused fused FP4 product.
/// Probe completion must precede byte packing; neither view adds device allocation.
pub(super) struct Fp4Scratch<'a> {
    floats: Tensor<f32>,
    bytes: Tensor<f8e4m3fn>,
    _source: &'a mut Tensor<f32>,
}
impl<'a> Fp4Scratch<'a> {
    pub(super) const fn floats(&mut self) -> &mut Tensor<f32> {
        &mut self.floats
    }
    pub(super) const fn bytes(&mut self) -> &mut Tensor<f8e4m3fn> {
        &mut self.bytes
    }
    /// # Safety
    /// The source arena must outlive every captured replay. Only its current lease holder
    /// may access these aliases, on one owning stream; probe completion precedes byte packing.
    #[expect(
        unsafe_code,
        reason = "Exclusive source lease anchors complete checked aliases; the owning stream orders probe and scale packing"
    )]
    pub(super) unsafe fn new(
        source: &'a mut Tensor<f32>,
        rows: usize,
        columns: usize,
    ) -> Result<Option<Self>> {
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| Error::invalid("FP4 product overflow"))?;
        if rows == 0
            || columns == 0
            || !rows.is_multiple_of(SCALE_ROWS)
            || !columns.is_multiple_of(NVFP4_GROUP_SIZE)
            || source.size() != elements
            || !source.is_contiguous()
        {
            return Ok(None);
        }
        // SAFETY: The exclusive source lease lasts until both aliases are destroyed, their
        // full extents equal the source, and the caller orders probe completion before packing.
        let floats = unsafe { row_view(source, 1, 0) }?
            .reshape(&[rows, columns])
            .map_err(device_error)?;
        // SAFETY: Same full extent and exclusive source lease; only metadata is created.
        let owner = Arc::new(unsafe { row_view(source, 1, 0) }?);
        let scale_row = columns / NVFP4_GROUP_SIZE * SCALE_ROWS;
        let bytes = owner
            .reinterpret::<f8e4m3fn>(&[source.num_bytes() / scale_row, scale_row])
            .map_err(device_error)?;
        drop(owner);
        let bytes =
            Arc::try_unwrap(bytes).map_err(|_| Error::invariant("FP4 exclusive byte metadata"))?;
        Ok(Some(Self {
            floats,
            bytes,
            _source: source,
        }))
    }
}
/// A shaped output alias whose exclusive owner remains borrowed through synchronous tuning.
pub(super) struct Fp4Output<'a> {
    tensor: Tensor<f32>,
    _source: &'a mut Tensor<f32>,
}
impl<'a> Fp4Output<'a> {
    pub(super) const fn tensor(&mut self) -> &mut Tensor<f32> {
        &mut self.tensor
    }
    /// # Safety
    /// Retain the source until every temporary preparation graph completes and is destroyed.
    /// Serialize all accesses with the arena's owning stream and exclusive lease.
    #[expect(
        unsafe_code,
        reason = "Complete shape-checked output alias is bounded by the exclusive source lease"
    )]
    pub(super) unsafe fn new(
        source: &'a mut Tensor<f32>,
        rows: usize,
        columns: usize,
    ) -> Result<Self> {
        if rows.checked_mul(columns) != Some(source.size()) || !source.is_contiguous() {
            return Err(Error::invalid("FP4 output shape"));
        }
        // SAFETY: The source is exclusively borrowed for the alias lifetime; temporary
        // preparation graphs complete and are destroyed before this lease is released.
        let tensor = unsafe { row_view(source, 1, 0) }?
            .reshape(&[rows, columns])
            .map_err(device_error)?;
        Ok(Self {
            tensor,
            _source: source,
        })
    }
}
