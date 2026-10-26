//! Historical main four-slot recording used only as a performance/correctness baseline.
//! The host layout and record methods below come from 2be509a6; device entries are unchanged.
use super::super::workspace::{STACKED_ROWS, VERIFY_LANES, grouped_options};
use super::device::kernels;
use super::{CONTROL_TILE, Geometry, Header, aligned, product};
use crate::constants::{F32_BYTES, KV_BLOCK_TOKENS, METADATA_FIELDS};
use crate::device::{CudaDevice, device_error};
use crate::resident::capture::error;
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, OpId, Result};
use std::collections::BTreeMap;
use std::result::Result as StdResult;
use std::slice::from_ref;

const MAX_BATCH: usize = 4;
const CONTROL_ROWS: usize = 6;

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
    pub(super) fn fixture(
        device: &CudaDevice,
        geometry: Geometry,
        (slots, capacity, parts): (usize, usize, usize),
        header: Header,
    ) -> Result<Self> {
        let node = OpId::ONE;
        let plan = Blueprint {
            layout: Layout::new(geometry, slots, capacity, parts)?,
            parameters: BTreeMap::from([(node, header)]),
        };
        Ok(Self {
            buffers: BTreeMap::from([(geometry, Buffers::new(device, plan)?)]),
            nodes: BTreeMap::from([(node, geometry)]),
        })
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
