//! Bounded groups of paged verification slots, retaining the original arithmetic.
use super::workspace::{STACKED_LANE_ROWS, STACKED_ROWS, VERIFY_LANES};
use crate::constants::{F32_BYTES, KV_BLOCK_TOKENS, METADATA_FIELDS};
use crate::device::{CudaDevice, DeviceProfile, device_error};
use crate::resident::buffers::buffer;
use crate::resident::{ProgramWeights, capture::error};
use crate::tuning::attention::decode::{grouped_parts_for, partitions};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, OpId, Result};
use infer_cuda_kernels::contract::{
    Attention, Partials, PooledAppendBuffers, PooledDecodeBuffers, PooledPlan, Recording,
};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;
use std::result::Result as StdResult;

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
    numerator: Allocation<f32>,
    maxima: Allocation<f32>,
    sums: Allocation<f32>,
    controls: Vec<Allocation<i32>>,
    parameters: BTreeMap<OpId, Arc<Tensor<i32>>>,
}

impl Buffers {
    fn new(device: &CudaDevice, plan: Blueprint) -> Result<Self> {
        let l = plan.layout;
        l.bytes(plan.parameters.len())?;
        let runtime = device.runtime();
        let zeros = |shape: &[usize]| runtime.zeros::<f32>(shape.iter().product());
        Ok(Self {
            layout: l,
            numerator: zeros(&[l.blocks, STACKED_ROWS, l.geometry.dim])?,
            maxima: zeros(&[l.blocks, STACKED_ROWS])?,
            sums: zeros(&[l.blocks, STACKED_ROWS])?,
            controls: (0..l.groups)
                .map(|_| runtime.zeros::<i32>(CONTROL_ROWS * l.columns))
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

impl Buffers {
    const fn plan(&self, first: usize, batch: usize) -> PooledPlan {
        let l = self.layout;
        PooledPlan {
            heads: l.geometry.heads,
            kv_heads: l.geometry.kv_heads,
            dim: l.geometry.dim,
            parts: l.parts,
            columns: l.columns,
            fields: METADATA_FIELDS,
            full_batch: l.batch,
            first,
            batch,
        }
    }
}

pub(super) struct Workspace {
    buffers: BTreeMap<Geometry, Buffers>,
    nodes: BTreeMap<OpId, Geometry>,
}

pub struct Append<'a> {
    pub node: OpId,
    pub keys: &'a Tensor<f8e4m3fn>,
    pub values: &'a Tensor<f8e4m3fn>,
    pub inputs: [&'a Tensor<f32>; 2],
    pub metadata: &'a Tensor<i32>,
    pub tables: &'a [Tensor<i32>],
}

pub struct Attend<'a> {
    pub node: OpId,
    pub output: &'a Tensor<f32>,
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

    fn buffer(&self, node: OpId) -> StdResult<&Buffers, DeviceError> {
        let key = self
            .nodes
            .get(&node)
            .ok_or_else(|| error("pooled attention node"))?;
        self.buffers
            .get(key)
            .ok_or_else(|| error("pooled attention buffers"))
    }

    /// All appends finish before attention may reuse the K/V activation storage.
    pub(super) fn append(
        &self,
        attention: &dyn Attention,
        recording: &Recording<'_>,
        args: &Append<'_>,
    ) -> StdResult<(), DeviceError> {
        let Append {
            node,
            keys,
            values,
            inputs,
            metadata,
            tables,
        } = *args;
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
        let header = work
            .parameters
            .get(&node)
            .ok_or_else(|| error("pooled attention header"))?;
        for (group, control) in work.controls.iter().enumerate() {
            let first = group * l.batch;
            let batch = (l.slots - first).min(l.batch);
            let table = |offset: usize| buffer(&tables[first + offset.min(batch - 1)]);
            attention
                .append_pooled(
                    recording,
                    work.plan(first, batch),
                    PooledAppendBuffers {
                        control: buffer(control)?,
                        tables: [table(0)?, table(1)?, table(2)?, table(MAX_BATCH - 1)?],
                        header: buffer(header.as_ref())?,
                        metadata: buffer(metadata)?,
                        keys: buffer(keys)?,
                        values: buffer(values)?,
                        key_rows: buffer(inputs[0])?,
                        value_rows: buffer(inputs[1])?,
                    },
                )
                .map_err(error)?;
        }
        Ok(())
    }

    pub(super) fn attend(
        &self,
        attention: &dyn Attention,
        recording: &Recording<'_>,
        args: &Attend<'_>,
    ) -> StdResult<(), DeviceError> {
        let Attend {
            node,
            output,
            queries,
            keys,
            values,
        } = *args;
        let work = self.buffer(node)?;
        let l = work.layout;
        let g = l.geometry;
        let size = product(&[l.slots, VERIFY_LANES, g.heads, g.dim]).map_err(error)?;
        if queries.size() != size || output.size() != size {
            return Err(error("pooled attention query dimensions"));
        }
        for (group, control) in work.controls.iter().enumerate() {
            let first = group * l.batch;
            let batch = (l.slots - first).min(l.batch);
            attention
                .decode_pooled(
                    recording,
                    work.plan(first, batch),
                    PooledDecodeBuffers {
                        output: buffer(output)?,
                        partials: Partials {
                            numerator: buffer(&work.numerator)?,
                            maxima: buffer(&work.maxima)?,
                            sums: buffer(&work.sums)?,
                        },
                        queries: buffer(queries)?,
                        keys: buffer(keys)?,
                        values: buffer(values)?,
                        control: buffer(control)?,
                    },
                )
                .map_err(error)?;
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
        &self,
        attention: &dyn Attention,
        recording: &Recording<'_>,
        args: &Append<'_>,
    ) -> StdResult<(), DeviceError> {
        self.pooled
            .as_ref()
            .ok_or_else(|| error("pooled attention absent"))?
            .append(attention, recording, args)
    }

    pub(crate) fn record_pooled_attention(
        &self,
        attention: &dyn Attention,
        recording: &Recording<'_>,
        args: &Attend<'_>,
    ) -> StdResult<(), DeviceError> {
        self.pooled
            .as_ref()
            .ok_or_else(|| error("pooled attention absent"))?
            .attend(attention, recording, args)
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
