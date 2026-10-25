//! Bounded groups of paged verification slots, retaining the original arithmetic.
use super::workspace::{STACKED_LANE_ROWS, STACKED_ROWS, VERIFY_LANES, grouped_options};
use crate::constants::{F32_BYTES, KV_BLOCK_TOKENS, METADATA_FIELDS};
use crate::device::{CudaDevice, DeviceProfile, device_error};
use crate::resident::{ProgramWeights, capture::error};
use crate::tuning::attention::decode::{grouped_parts_for, partitions};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, OpId, Result};
use infer_ir::{DataflowGraph, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;
use std::result::Result as StdResult;
use std::slice::from_ref;

mod device;
use device::kernels;

const MAX_BATCH: usize = 4;
const ATTENTION_INPUTS: usize = 3;
const CONTROL_ROWS: usize = 6;
const CONTROL_TILE: usize = 32;
const ALIGNMENT: usize = 256;

type Header = [i32; CONTROL_TILE];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Geometry {
    heads: usize,
    kv_heads: usize,
    dim: usize,
}

#[derive(Clone, Copy)]
struct Layout {
    geometry: Geometry,
    slots: usize,
    batch: usize,
    groups: usize,
    parts: usize,
    blocks: usize,
    columns: usize,
}

fn product(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |value, &factor| {
        value
            .checked_mul(factor)
            .ok_or_else(|| Error::invalid("pooled attention size overflow"))
    })
}

fn aligned(bytes: usize) -> Result<u64> {
    let bytes = bytes
        .checked_add(ALIGNMENT - 1)
        .map(|value| value / ALIGNMENT * ALIGNMENT)
        .ok_or_else(|| Error::invalid("pooled attention alignment overflow"))?;
    u64::try_from(bytes).map_err(device_error)
}

impl Layout {
    fn new(geometry: Geometry, slots: usize, capacity: usize, parts: usize) -> Result<Self> {
        let batch = slots.min(MAX_BATCH);
        if batch == 0 {
            return Err(Error::invalid("empty pooled attention"));
        }
        let metadata = product(&[batch, VERIFY_LANES, METADATA_FIELDS])?;
        let columns = capacity
            .div_ceil(KV_BLOCK_TOKENS)
            .max(metadata)
            .max(CONTROL_TILE)
            .checked_next_power_of_two()
            .ok_or_else(|| Error::invalid("pooled attention table width overflow"))?;
        let layout = Self {
            geometry,
            slots,
            batch,
            groups: slots.div_ceil(batch),
            parts,
            blocks: product(&[batch, geometry.kv_heads, parts])?,
            columns,
        };
        layout.bytes(0)?;
        Ok(layout)
    }

    const fn prefix(self) -> usize {
        self.slots
    }

    fn bytes(self, parameters: usize) -> Result<u64> {
        let g = self.geometry;
        let allocations = [
            (product(&[self.blocks, STACKED_ROWS, g.dim, F32_BYTES])?, 1),
            (product(&[self.blocks, STACKED_ROWS, F32_BYTES])?, 2),
            (
                product(&[CONTROL_ROWS, self.columns, size_of::<i32>()])?,
                self.groups,
            ),
            (CONTROL_TILE * size_of::<i32>(), parameters),
        ];
        allocations
            .into_iter()
            .try_fold(0u64, |total, (bytes, count)| {
                let bytes = aligned(bytes)?
                    .checked_mul(count as u64)
                    .ok_or_else(|| Error::invalid("pooled attention allocation count overflow"))?;
                total
                    .checked_add(bytes)
                    .ok_or_else(|| Error::invalid("pooled attention quote overflow"))
            })
    }
}

struct Blueprint {
    layout: Layout,
    parameters: BTreeMap<OpId, Header>,
}

fn blueprints(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    slots: usize,
    capacity: usize,
    profile: &DeviceProfile,
) -> Result<BTreeMap<Geometry, Blueprint>> {
    let mut plans: BTreeMap<Geometry, Blueprint> = BTreeMap::new();
    #[cfg(test)]
    if DISABLED.get() {
        return Ok(plans);
    }
    if slots < 2 || capacity == 0 {
        return Ok(plans);
    }
    let sms = usize::try_from(profile.multiprocessors).map_err(device_error)?;
    for node in &graph.nodes {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            continue;
        };
        if node.inputs.len() != ATTENTION_INPUTS
            || node
                .inputs
                .iter()
                .any(|id| weights.constants.contains_key(id))
            || partitions(query_heads, window, sms) == 1
            || !head_dim.is_power_of_two()
            || head_dim < CONTROL_TILE
        {
            continue;
        }
        let Some(parts) = grouped_parts_for(query_heads, kv_heads, sms, slots.min(MAX_BATCH))
        else {
            continue;
        };
        if query_heads / kv_heads > STACKED_LANE_ROWS {
            continue;
        }
        let Some(state) = node.states.first() else {
            continue;
        };
        let Some(scales) = weights.kv_scales.get(state) else {
            continue;
        };
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(Error::invalid("pooled attention cache scale"));
        }
        let geometry = Geometry {
            heads: query_heads,
            kv_heads,
            dim: head_dim,
        };
        let layout = Layout::new(geometry, slots, capacity, parts)?;
        let entry = plans.entry(geometry).or_insert(Blueprint {
            layout,
            parameters: BTreeMap::new(),
        });
        let mut header = [0; CONTROL_TILE];
        header[0] = i32::from_ne_bytes(scales[0].to_ne_bytes());
        header[1] = i32::from_ne_bytes(scales[1].to_ne_bytes());
        header[2] = i32::try_from(window.unwrap_or(0)).map_err(device_error)?;
        entry.parameters.insert(node.id, header);
    }
    Ok(plans)
}

pub(super) fn required_bytes(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    slots: usize,
    capacity: usize,
    profile: &DeviceProfile,
) -> Result<u64> {
    blueprints(graph, weights, slots, capacity, profile)?
        .values()
        .try_fold(0u64, |total, plan| {
            total
                .checked_add(plan.layout.bytes(plan.parameters.len())?)
                .ok_or_else(|| Error::invalid("pooled attention total overflow"))
        })
}

struct Buffers {
    layout: Layout,
    numerator: Tensor<f32>,
    maxima: Tensor<f32>,
    sums: Tensor<f32>,
    controls: Vec<Tensor<i32>>,
    parameters: BTreeMap<OpId, Arc<Tensor<i32>>>,
}

impl Buffers {
    fn new(device: &CudaDevice, plan: Blueprint) -> Result<Self> {
        let l = plan.layout;
        l.bytes(plan.parameters.len())?;
        let zeros = |shape: &[usize]| {
            api::zeros::<f32>(shape)
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        Ok(Self {
            layout: l,
            numerator: zeros(&[l.blocks, STACKED_ROWS, l.geometry.dim])?,
            maxima: zeros(&[l.blocks, STACKED_ROWS])?,
            sums: zeros(&[l.blocks, STACKED_ROWS])?,
            controls: (0..l.groups)
                .map(|_| {
                    api::zeros::<i32>(&[CONTROL_ROWS, l.columns])
                        .sync_on(&device.stream)
                        .map_err(device_error)
                })
                .collect::<Result<_>>()?,
            parameters: plan
                .parameters
                .into_iter()
                .map(|(id, values)| {
                    device
                        .upload(values.to_vec(), &[CONTROL_TILE])
                        .map(|tensor| (id, tensor))
                })
                .collect::<Result<_>>()?,
        })
    }
}

pub(super) struct Workspace {
    buffers: BTreeMap<Geometry, Buffers>,
    nodes: BTreeMap<OpId, Geometry>,
}

pub struct Append<'a> {
    pub node: OpId,
    pub keys: &'a mut Tensor<f8e4m3fn>,
    pub values: &'a mut Tensor<f8e4m3fn>,
    pub inputs: [&'a Tensor<f32>; 2],
    pub metadata: &'a Tensor<i32>,
    pub tables: &'a [Tensor<i32>],
}

pub struct Attend<'a> {
    pub node: OpId,
    pub output: &'a mut Tensor<f32>,
    pub queries: &'a Tensor<f32>,
    pub keys: &'a Tensor<f8e4m3fn>,
    pub values: &'a Tensor<f8e4m3fn>,
}

impl Workspace {
    pub(super) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        slots: usize,
        capacity: usize,
    ) -> Result<Self> {
        let mut buffers = BTreeMap::new();
        let mut nodes = BTreeMap::new();
        for (geometry, plan) in blueprints(graph, weights, slots, capacity, device.profile()?)? {
            for &id in plan.parameters.keys() {
                nodes.insert(id, geometry);
            }
            buffers.insert(geometry, Buffers::new(device, plan)?);
        }
        #[cfg(test)]
        BUILDS.set(BUILDS.get() + nodes.len());
        Ok(Self { buffers, nodes })
    }

    pub(super) fn prefix(&self, node: OpId, slots: usize) -> usize {
        self.nodes
            .get(&node)
            .and_then(|key| self.buffers.get(key))
            .filter(|work| work.layout.slots == slots)
            .map_or(0, |work| work.layout.prefix())
    }

    fn buffer(&mut self, node: OpId) -> StdResult<&mut Buffers, DeviceError> {
        let key = self
            .nodes
            .get(&node)
            .ok_or_else(|| error("pooled attention node"))?;
        self.buffers
            .get_mut(key)
            .ok_or_else(|| error("pooled attention buffers"))
    }

    /// All appends finish before attention may reuse the K/V activation storage.
    pub(super) fn append(&mut self, scope: &Scope, args: Append<'_>) -> StdResult<(), DeviceError> {
        let Append {
            node,
            keys,
            values,
            inputs,
            metadata,
            tables,
        } = args;
        let work = self.buffer(node)?;
        let l = work.layout;
        let g = l.geometry;
        let size = product(&[l.slots, VERIFY_LANES, g.kv_heads, g.dim]).map_err(error)?;
        if inputs.iter().any(|input| input.size() != size)
            || tables.len() != l.slots
            || metadata.size() != l.slots * VERIFY_LANES * METADATA_FIELDS
        {
            return Err(error("pooled attention append dimensions"));
        }
        let physical = keys.size() / (g.kv_heads * g.dim);
        let header = work
            .parameters
            .get(&node)
            .ok_or_else(|| error("pooled attention header"))?;
        let key_input = inputs[0].view(&[size])?;
        let value_input = inputs[1].view(&[size])?;
        let metadata = metadata.view(&[metadata.size()])?;
        for (group, control) in work.controls.iter_mut().enumerate() {
            let first = group * l.batch;
            let last = (first + l.batch).min(l.slots);
            let batch = last - first;
            let first_lane = first * VERIFY_LANES;
            let last_lane = last * VERIFY_LANES;
            let table = |offset: usize| &tables[first + offset.min(batch - 1)];
            let words = first_lane * METADATA_FIELDS..last_lane * METADATA_FIELDS;
            let fragment = metadata.slice(from_ref(&words))?;
            scope.record(
                kernels::pack_control(
                    control.partition([1, l.columns]),
                    table(0),
                    table(1),
                    table(2),
                    table(MAX_BATCH - 1),
                    header.as_ref(),
                    &fragment,
                )
                .generics(vec![l.columns.to_string()]),
            )?;
            let range = first_lane * g.kv_heads * g.dim..last_lane * g.kv_heads * g.dim;
            let key_rows = key_input.slice(from_ref(&range))?;
            let value_rows = value_input.slice(from_ref(&range))?;
            scope.record(
                kernels::append_pages(
                    (&mut *keys).partition([1, physical, g.dim]),
                    (&mut *values).partition([1, physical, g.dim]),
                    &key_rows.view(&[batch * VERIFY_LANES, g.kv_heads, g.dim])?,
                    &value_rows.view(&[batch * VERIFY_LANES, g.kv_heads, g.dim])?,
                    &*control,
                )
                .generics(
                    [
                        g.dim,
                        physical,
                        g.kv_heads,
                        batch * VERIFY_LANES,
                        METADATA_FIELDS,
                    ]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                ),
            )?;
        }
        Ok(())
    }

    pub(super) fn attend(&mut self, scope: &Scope, args: Attend<'_>) -> StdResult<(), DeviceError> {
        let Attend {
            node,
            output,
            queries,
            keys,
            values,
        } = args;
        let work = self.buffer(node)?;
        let l = work.layout;
        let g = l.geometry;
        let size = product(&[l.slots, VERIFY_LANES, g.heads, g.dim]).map_err(error)?;
        if queries.size() != size || output.size() != size {
            return Err(error("pooled attention query dimensions"));
        }
        let queries = queries.view(&[size])?;
        let physical = keys.size() / (g.kv_heads * g.dim);
        let keys = keys.view(&[g.kv_heads, physical, g.dim])?;
        let values = values.view(&[g.kv_heads, physical, g.dim])?;
        for (group, control) in work.controls.iter().enumerate() {
            let first = group * l.batch;
            let batch = (l.slots - first).min(l.batch);
            let range = first * VERIFY_LANES * g.heads * g.dim
                ..(first + batch) * VERIFY_LANES * g.heads * g.dim;
            let fragment = queries.slice(from_ref(&range))?;
            scope.record(
                kernels::partial_slots(
                    (&mut work.numerator).partition_prefix([1, STACKED_ROWS, g.dim]),
                    (&mut work.maxima).partition_prefix([1, STACKED_ROWS]),
                    (&mut work.sums).partition_prefix([1, STACKED_ROWS]),
                    &fragment.view(&[
                        batch * VERIFY_LANES * g.kv_heads,
                        g.heads / g.kv_heads,
                        g.dim,
                    ])?,
                    &keys,
                    &values,
                    control,
                )
                .generics(
                    [g.dim, g.kv_heads, l.parts, METADATA_FIELDS]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                )
                .compile_options(grouped_options())
                .grid((
                    u32::try_from(batch * g.kv_heads * l.parts).map_err(error)?,
                    1,
                    1,
                )),
            )?;
            scope.record(
                kernels::merge_slots(
                    (&mut *output).partition([1, g.dim]),
                    &work
                        .numerator
                        .view(&[l.batch * g.kv_heads, l.parts, STACKED_ROWS, g.dim])?,
                    &work
                        .maxima
                        .view(&[l.batch * g.kv_heads, l.parts, STACKED_ROWS])?,
                    &work
                        .sums
                        .view(&[l.batch * g.kv_heads, l.parts, STACKED_ROWS])?,
                    i32::try_from(first).map_err(error)?,
                    i32::try_from(batch).map_err(error)?,
                )
                .generics(
                    [g.dim, l.parts, g.heads / g.kv_heads, g.heads]
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                ),
            )?;
        }
        Ok(())
    }
}

impl super::workspace::Workspace {
    pub(crate) fn pooled_bytes(
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, usize, usize),
        profile: &DeviceProfile,
    ) -> Result<u64> {
        let (slots, verify, capacity) = geometry;
        if verify != VERIFY_LANES {
            return Ok(0);
        }
        required_bytes(graph, weights, slots, capacity, profile)
    }

    pub(crate) fn prepare_pooled(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        geometry: (usize, usize, usize),
        paged: bool,
    ) -> Result<()> {
        let (slots, verify, capacity) = geometry;
        self.pooled = if paged && verify == VERIFY_LANES {
            Some(Workspace::new(device, graph, weights, slots, capacity)?)
        } else {
            None
        };
        Ok(())
    }

    pub(crate) fn pooled_prefix(&self, node: OpId, slots: usize) -> usize {
        self.pooled
            .as_ref()
            .map_or(0, |work| work.prefix(node, slots))
    }

    pub(crate) fn record_pooled_append(
        &mut self,
        scope: &Scope,
        args: Append<'_>,
    ) -> StdResult<(), DeviceError> {
        self.pooled
            .as_mut()
            .ok_or_else(|| error("pooled attention absent"))?
            .append(scope, args)
    }

    pub(crate) fn record_pooled_attention(
        &mut self,
        scope: &Scope,
        args: Attend<'_>,
    ) -> StdResult<(), DeviceError> {
        self.pooled
            .as_mut()
            .ok_or_else(|| error("pooled attention absent"))?
            .attend(scope, args)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_pooled_attention.rs"]
mod tests;

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static BUILDS: Cell<usize> = const { Cell::new(0) };
}

/// Build the unchanged original captures on this thread without a production flag.
#[cfg(test)]
pub fn with_row_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let restore = Restore(DISABLED.replace(true));
    let value = run();
    drop(restore);
    value
}

#[cfg(test)]
pub fn builds() -> usize {
    BUILDS.get()
}
