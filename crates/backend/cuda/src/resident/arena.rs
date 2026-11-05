use super::buffers::buffer;
use crate::constants::F32_BYTES;
use crate::constants::NVFP4_GROUP_SIZE;
use crate::device::vendor_fp8::nvfp4::SCALE_ROWS;
use crate::device::{CudaDevice, device_error};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_cuda_kernels::contract::{Buffer, Fp8};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::DataflowGraph;
use std::collections::BTreeMap;
use std::result::Result as StdResult;
use std::sync::Arc;

/// Slot layout of one graph: `last_node` and element count per reused slot.
type ArenaLayout = (Vec<(usize, usize)>, BTreeMap<TensorId, usize>);

/// Reuses equal-sized device buffers after their last consumer.
///
/// Graph results stay live until readback. Batched arenas flatten lanes into one axis: every
/// slot is `[lanes * size]`. Views of an arena share its allocations, so a view, and any tensor
/// taken from one, stays valid whatever happens to the arena it came from.
pub struct ActivationArena {
    indices: BTreeMap<TensorId, usize>,
    slots: Vec<Slot>,
    lanes: usize,
    /// Whether slots hold concatenated lane rows. A one-lane arena is still batched when the
    /// pool captures a narrow decode rung over it, unlike a scalar `new` arena.
    batched: bool,
    elements: usize,
}

impl ActivationArena {
    /// # Errors
    /// Rejects invalid graphs, overflowing budgets and failed device allocation.
    pub fn new(device: &CudaDevice, graph: &DataflowGraph, byte_budget: usize) -> Result<Self> {
        Self::plan(device, graph, 1, byte_budget, false)
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
        Self::plan(device, graph, lanes, byte_budget, true)
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

    /// One lane of this batched arena as a single-row arena whose slots are that lane's rows,
    /// so per-lane captures and whole-slot captures of the batched arena share storage.
    ///
    /// A lane's rows must not be used as concurrent mutable outputs together with the whole slot.
    /// # Errors
    /// Rejects a lane outside the arena or a slot that does not divide into lanes.
    pub(super) fn lane_view(&self, lane: usize) -> Result<Self> {
        let slots = self
            .slots
            .iter()
            .map(|slot| slot.row(self.lanes, lane))
            .collect::<Result<_>>()?;
        Ok(Self {
            indices: self.indices.clone(),
            slots,
            lanes: 1,
            batched: true,
            elements: self.elements / self.lanes,
        })
    }

    /// An arena whose slots are this one's storage.
    ///
    /// Prompt graphs of different sequences capture against views of one shared arena, so the
    /// activation memory is paid once per prompt width instead of once per admitted sequence.
    /// Replays through any two views must be serialized on one stream, and no activation may be
    /// read after another view's replay has started.
    /// # Errors
    /// Rejects a failed view construction.
    pub(super) fn shared_view(&self) -> Result<Self> {
        let slots = self
            .slots
            .iter()
            .map(|slot| slot.range(0, slot.len))
            .collect::<Result<_>>()?;
        Ok(Self {
            indices: self.indices.clone(),
            slots,
            lanes: self.lanes,
            batched: self.batched,
            elements: self.elements,
        })
    }

    /// An arena planned for `lanes` rows of `graph` whose slots are the leading elements of
    /// this wider arena's slots, so a narrower prompt graph needs no activation memory of its
    /// own. `single` is the tensor this arena keeps one row of, as when it was planned.
    /// Replays through it and through `self` must be serialized on one stream with no activation
    /// read across them.
    /// # Errors
    /// Rejects a wider `lanes` or a different slot layout.
    pub(super) fn narrowed(
        &self,
        graph: &DataflowGraph,
        lanes: usize,
        single: Option<TensorId>,
    ) -> Result<Self> {
        let (sizes, indices) = Self::sizes(graph, lanes, single)?;
        if lanes == 0 || lanes > self.lanes || indices != self.indices {
            return Err(Error::invalid("narrowed arena layout"));
        }
        let slots = self
            .slots
            .iter()
            .zip(&sizes)
            .map(|(slot, &size)| {
                if size > slot.len {
                    return Err(Error::invalid("narrowed arena slot"));
                }
                slot.range(0, size)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            indices,
            slots,
            lanes,
            batched: self.batched,
            elements: sizes.iter().sum(),
        })
    }

    /// Views of rows `start..start + rows` of every slot, as an arena of `rows` lanes.
    ///
    /// A slot holding one row whatever the lane count (the prompt logits row) is viewed whole.
    /// Replays through the views must be serialized on one stream.
    /// # Errors
    /// Rejects a layout that differs from this arena's or a segment outside it.
    pub(super) fn segment(
        &self,
        graph: &DataflowGraph,
        rows: usize,
        start: usize,
        single: Option<TensorId>,
    ) -> Result<Self> {
        let (sizes, indices) = Self::sizes(graph, rows, single)?;
        if rows == 0 || start + rows > self.lanes || indices != self.indices {
            return Err(Error::invalid("arena segment layout"));
        }
        let slots = self
            .slots
            .iter()
            .zip(&sizes)
            .map(|(slot, &size)| {
                let offset = if size == slot.len {
                    0
                } else {
                    start * (size / rows)
                };
                if offset + size > slot.len {
                    return Err(Error::invalid("arena segment slot"));
                }
                slot.range(offset, size)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            indices,
            slots,
            lanes: rows,
            batched: self.batched,
            elements: sizes.iter().sum(),
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
        Self::plan_with(device, graph, lanes, byte_budget, graph.logits, true)
    }

    fn plan(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
        batched: bool,
    ) -> Result<Self> {
        Self::plan_with(device, graph, lanes, byte_budget, None, batched)
    }

    fn plan_with(
        device: &CudaDevice,
        graph: &DataflowGraph,
        lanes: usize,
        byte_budget: usize,
        single: Option<TensorId>,
        batched: bool,
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
        let mut slots = Vec::with_capacity(sizes.len());
        for size in sizes {
            let memory = api::zeros::<f32>(&[size])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            slots.push(Slot::new(Allocation::adopt(Arc::new(memory))?, 0, size)?);
        }
        Ok(Self {
            indices,
            slots,
            lanes,
            batched,
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

    /// Activation `id` as its slot memory: a kernel operand of known length that holds a share
    /// of the arena.
    /// # Errors
    /// Rejects tensors which are not graph activations.
    pub(crate) fn get(&self, id: TensorId) -> Result<SlotMemory> {
        self.memory(self.slot(id)?)
    }

    /// Activation `id` as a tensor, for kernels still launched through cuTile.
    /// # Errors
    /// Rejects tensors which are not graph activations.
    pub(crate) fn view(&self, id: TensorId) -> Result<&Tensor<f32>> {
        self.tensor(self.slot(id)?)
    }

    /// Slot `index` as a tensor, for kernels still launched through cuTile.
    /// # Errors
    /// Rejects an index outside the arena.
    pub(crate) fn tensor(&self, index: usize) -> Result<&Tensor<f32>> {
        self.at(index).map(|slot| &slot.view)
    }

    /// Activation `id` as a kernel operand, kept alive by any recording that retains it.
    /// # Errors
    /// Rejects tensors which are not graph activations.
    pub(crate) fn activation(&self, id: TensorId) -> Result<Buffer<'_, f32>> {
        self.buffer(self.slot(id)?)
    }

    /// Slot `index` as a kernel operand, kept alive by any recording that retains it.
    /// # Errors
    /// Rejects an index outside the arena.
    pub(crate) fn buffer(&self, index: usize) -> Result<Buffer<'_, f32>> {
        self.at(index)?.buffer()
    }

    /// A tensor over slot `index` for a kernel to write; it holds the slot's memory, so it
    /// needs no returning, and it aliases every other view of the slot.
    /// # Errors
    /// Rejects an index outside the arena.
    pub(crate) fn output(&self, index: usize) -> Result<Tensor<f32>> {
        self.at(index)?.tensor()
    }

    /// Slot `index` read as `U`s of `shape`, as [`Self::output`] gives it as floats.
    /// # Errors
    /// Rejects an index outside the arena and a shape that does not cover the slot's bytes.
    pub(crate) fn output_as<U: DType>(&self, index: usize, shape: &[usize]) -> Result<Tensor<U>> {
        let slot = self.at(index)?;
        slot.memory
            .view_as(slot.offset..slot.offset + slot.len, shape)
    }

    /// Slot `index` as memory a kernel writes, held apart from the arena: it shares the slot's
    /// allocation, so a capture can write it while it borrows the arena elsewhere.
    /// # Errors
    /// Rejects an index outside the arena.
    pub(crate) fn memory(&self, index: usize) -> Result<SlotMemory> {
        Ok(self.at(index)?.memory())
    }

    /// Row `lane` of batched slot `index`, as [`Self::memory`] gives the whole slot.
    /// # Errors
    /// Rejects an index outside the arena, a lane outside it or a slot that is not batched.
    pub(crate) fn row_memory(&self, index: usize, lane: usize) -> Result<SlotMemory> {
        Ok(self.at(index)?.row(self.lanes, lane)?.memory())
    }

    fn at(&self, index: usize) -> Result<&Slot> {
        self.slots
            .get(index)
            .ok_or_else(|| Error::invalid("activation slot index"))
    }
}

/// Elements `offset..offset + len` of an arena slot, sharing its allocation.
#[derive(Clone)]
pub struct SlotMemory {
    memory: Allocation<f32>,
    offset: usize,
    len: usize,
}

impl SlotMemory {
    /// The memory as a kernel operand, kept alive by any recording that retains it.
    /// # Errors
    /// As [`Buffer::slice`], which cannot fail for a range the arena checked.
    pub(crate) fn buffer(&self) -> Result<Buffer<'_, f32>> {
        self.memory
            .buffer()
            .slice(self.offset..self.offset + self.len)
    }

    #[must_use]
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Ordinal of the device holding the memory.
    #[must_use]
    pub(crate) fn device(&self) -> usize {
        self.memory.device()
    }

    /// The slot as an allocation of its own, holding a share of the arena memory, for an owner
    /// that outlives this handle, such as a graph's readout rows.
    /// # Errors
    /// As [`Allocation::slice`], which cannot fail for a range the arena checked.
    pub(crate) fn allocation(&self) -> Result<Allocation<f32>> {
        self.memory.slice(self.offset..self.offset + self.len)
    }

    /// The memory shaped `shape`, for the test-only taps and kernels still launched through cuTile.
    /// # Errors
    /// Rejects a shape that does not cover the memory.
    #[cfg(test)]
    pub(crate) fn tensor(&self, shape: &[usize]) -> Result<Tensor<f32>> {
        self.memory.view(self.offset..self.offset + self.len, shape)
    }
}

/// Elements `offset..offset + len` of an allocation that every view of the slot shares.
struct Slot {
    memory: Allocation<f32>,
    offset: usize,
    len: usize,
    /// The range as a tensor, for kernels still launched through cuTile.
    view: Tensor<f32>,
}

impl Slot {
    fn new(memory: Allocation<f32>, offset: usize, len: usize) -> Result<Self> {
        let view = slot_tensor(&memory, offset, len)?;
        Ok(Self {
            memory,
            offset,
            len,
            view,
        })
    }

    /// Elements `offset..offset + len` of this slot, sharing its memory.
    fn range(&self, offset: usize, len: usize) -> Result<Self> {
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(Error::invalid("activation slot range"));
        }
        Self::new(self.memory.clone(), self.offset + offset, len)
    }

    /// Row `lane` of this slot taken as `lanes` concatenated rows.
    fn row(&self, lanes: usize, lane: usize) -> Result<Self> {
        if lanes == 0 || !self.len.is_multiple_of(lanes) {
            return Err(Error::invalid("activation slot is not batched"));
        }
        if lane >= lanes {
            return Err(Error::invalid("activation lane out of bounds"));
        }
        let row = self.len / lanes;
        self.range(lane * row, row)
    }

    fn buffer(&self) -> Result<Buffer<'_, f32>> {
        self.memory
            .buffer()
            .slice(self.offset..self.offset + self.len)
    }

    fn memory(&self) -> SlotMemory {
        SlotMemory {
            memory: self.memory.clone(),
            offset: self.offset,
            len: self.len,
        }
    }

    fn tensor(&self) -> Result<Tensor<f32>> {
        slot_tensor(&self.memory, self.offset, self.len)
    }
}

/// A 1-D tensor over elements `offset..offset + len` of `memory`, owning a share of it.
fn slot_tensor(memory: &Allocation<f32>, offset: usize, len: usize) -> Result<Tensor<f32>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::invalid("activation slot range"))?;
    memory.view(offset..end, &[len])
}

/// Every row of `memory` taken as `lanes` concatenated rows, each holding a share of it, so a
/// row stays valid whatever happens to any other handle. Rows are disjoint, so each can be a
/// launch's output.
///
/// # Errors
/// Rejects zero lanes and memory that does not divide into `lanes` rows.
pub(super) fn rows<T: DType>(memory: &Allocation<T>, lanes: usize) -> Result<Vec<Tensor<T>>> {
    if lanes == 0 || !memory.len().is_multiple_of(lanes) {
        return Err(Error::invalid("activation slot is not batched"));
    }
    let size = memory.len() / lanes;
    (0..lanes)
        .map(|lane| memory.view(lane * size..(lane + 1) * size, &[size]))
        .collect()
}

/// All of `memory` shaped `shape`, as the one tensor a kernel may write it through; its storage
/// is its own, so cuTile can partition it mutably.
///
/// # Errors
/// Rejects a shape that does not cover the memory.
pub(super) fn whole<T: DType>(memory: &Allocation<T>, shape: &[usize]) -> Result<Tensor<T>> {
    memory.view(0..memory.len(), shape)
}

/// Float and byte views of an unused fused FP4 product, both holding the arena's memory: the
/// probe writes the floats, then scale packing reuses the bytes, ordered on the owning stream.
pub(super) struct Fp4Scratch {
    floats: Tensor<f32>,
    bytes: Tensor<f8e4m3fn>,
}

impl Fp4Scratch {
    pub(super) const fn floats(&mut self) -> &mut Tensor<f32> {
        &mut self.floats
    }

    /// The product's bytes as `[rows / 2, groups * 128]` rows of the FP4 scale layout.
    /// # Errors
    /// As [`buffer`].
    pub(super) fn bytes(&self) -> StdResult<Buffer<'_, Fp8>, DeviceError> {
        buffer(&self.bytes)
    }

    /// Views of arena slot `slot` as a `[rows, columns]` product; `None` when the product's
    /// geometry does not fit the FP4 scale layout.
    /// # Errors
    /// Rejects an unknown slot and overflowing geometry.
    pub(super) fn new(
        arena: &ActivationArena,
        slot: usize,
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
            || arena.buffer(slot)?.len() != elements
        {
            return Ok(None);
        }
        let scale_row = columns / NVFP4_GROUP_SIZE * SCALE_ROWS;
        Ok(Some(Self {
            floats: arena
                .output(slot)?
                .reshape(&[rows, columns])
                .map_err(device_error)?,
            bytes: arena.output_as(slot, &[elements * F32_BYTES / scale_row, scale_row])?,
        }))
    }
}
