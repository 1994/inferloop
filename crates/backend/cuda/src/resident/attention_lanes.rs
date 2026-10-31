//! One launch of decode attention across a pool's lanes instead of three per lane.
use super::{
    ActivationArena, attention::attention, attention_decode::kernels, batch::BatchBuilder,
    capture::error,
};
use crate::constants::ATTENTION_SPLIT_THRESHOLD;
use crate::device::{CudaDevice, DeviceProfile, device_error};
use crate::tuning::attention::decode::{GROUPED_ROWS, grouped_parts_for, partitions};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

/// Elements of one split's `numerator`, `maxima` and `sums` rows beyond the head dimension.
const PARTIAL_ROW_SCALARS: usize = 2;
/// Keys and values of an unquantized attention state.
const KV_TENSORS: usize = 2;

/// Scratch of one geometry's lane-batched splits.
pub struct Buffers {
    parts: usize,
    numerator: Tensor<f32>,
    maxima: Tensor<f32>,
    sums: Tensor<f32>,
}

/// Lane-batched scratch per `(heads, dim)`.
#[derive(Default)]
pub struct Workspace(BTreeMap<(usize, usize), Buffers>);

/// `(heads, kv_heads, dim, parts)` of each covered geometry.
fn geometries(
    graph: &DataflowGraph,
    lanes: usize,
    multiprocessors: usize,
) -> BTreeMap<(usize, usize), (usize, usize)> {
    let mut plans = BTreeMap::new();
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
        if partitions(query_heads, window, multiprocessors) == 1 {
            continue;
        }
        if let Some(parts) = grouped_parts_for(query_heads, kv_heads, multiprocessors, lanes) {
            plans.insert((query_heads, head_dim), (kv_heads, parts));
        }
    }
    plans
}

/// Device bytes of [`Workspace::new`].
/// # Errors
/// Rejects an unrepresentable multiprocessor count.
pub fn required_bytes(graph: &DataflowGraph, lanes: usize, profile: &DeviceProfile) -> Result<u64> {
    let multiprocessors = usize::try_from(profile.multiprocessors).map_err(device_error)?;
    let bytes: usize = geometries(graph, lanes, multiprocessors)
        .into_iter()
        .map(|((_, dim), (kv_heads, parts))| {
            lanes * kv_heads * parts * GROUPED_ROWS * (dim + PARTIAL_ROW_SCALARS) * size_of::<f32>()
        })
        .sum();
    u64::try_from(bytes).map_err(device_error)
}

impl Workspace {
    /// # Errors
    /// Reports CUDA allocation failures.
    pub fn new(device: &CudaDevice, graph: &DataflowGraph, lanes: usize) -> Result<Self> {
        let multiprocessors =
            usize::try_from(device.profile()?.multiprocessors).map_err(device_error)?;
        let zeros = |shape: &[usize]| {
            api::zeros::<f32>(shape)
                .sync_on(&device.stream)
                .map_err(device_error)
        };
        let mut buffers = BTreeMap::new();
        for ((heads, dim), (kv_heads, parts)) in geometries(graph, lanes, multiprocessors) {
            let blocks = lanes
                .checked_mul(kv_heads * parts)
                .ok_or_else(|| Error::invalid("lane attention overflow"))?;
            buffers.insert(
                (heads, dim),
                Buffers {
                    parts,
                    numerator: zeros(&[blocks, GROUPED_ROWS, dim])?,
                    maxima: zeros(&[blocks, GROUPED_ROWS])?,
                    sums: zeros(&[blocks, GROUPED_ROWS])?,
                },
            );
        }
        Ok(Self(buffers))
    }
}

/// The restored KV pair and the outcome of recording against it.
type Recorded<E> = (Tensor<E>, Tensor<E>, StdResult<(), DeviceError>);

/// One KV pair and how a launch addresses it.
struct Cache<E: DType> {
    keys: Tensor<E>,
    values: Tensor<E>,
    scales: [f32; 2],
    /// One row per lane of a pooled root, rather than pages shared by every lane.
    per_lane: bool,
}

impl BatchBuilder<'_> {
    /// False leaves the node to one launch per lane.
    pub(super) fn record_lanes_attention(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        caches: &mut [super::fp8_cache::Fp8Caches],
        metadata: &Tensor<i32>,
    ) -> StdResult<bool, DeviceError> {
        let TensorOp::Attention {
            query_heads,
            head_dim,
            window,
            ..
        } = node.op
        else {
            return Ok(false);
        };
        let Some(&state) = node.states.first() else {
            return Ok(false);
        };
        if self.table_root.is_none()
            || window.is_some_and(|w| w < ATTENTION_SPLIT_THRESHOLD)
            || node
                .inputs
                .iter()
                .any(|id| self.weights.constants.contains_key(id))
            || !self
                .lanes_attention
                .as_ref()
                .is_some_and(|work| work.0.contains_key(&(query_heads, head_dim)))
        {
            return Ok(false);
        }
        if let Some(scales) = self.weights.kv_scales.get(&state).copied() {
            // Only paged slots alias one cache; separate slot caches keep their own launches.
            let Some((first, rest)) = caches.split_first_mut() else {
                return Ok(false);
            };
            let pointer = |cache: &(Tensor<f8e4m3fn>, Tensor<f8e4m3fn>)| {
                cache.0.device_pointer().cu_deviceptr()
            };
            let Some(shared) = first.get(&state).map(pointer) else {
                return Ok(false);
            };
            if rest
                .iter()
                .any(|slot| slot.get(&state).map(pointer) != Some(shared))
            {
                return Ok(false);
            }
            let Some((keys, values)) = first.remove(&state) else {
                return Ok(false);
            };
            let cache = Cache {
                keys,
                values,
                scales,
                per_lane: false,
            };
            let (keys, values, recorded) =
                self.record_lanes::<f8e4m3fn>(scope, node, arena, cache, metadata)?;
            // A per-lane capture leaves each slot's views in the head-major shape state copies
            // compare; this launch records none of them, so shape them here.
            let shape = keys.shape().to_vec();
            let dims = shape
                .iter()
                .map(|&dim| usize::try_from(dim).map_err(error))
                .collect::<StdResult<Vec<_>, _>>()?;
            first.insert(state, (keys, values));
            for slot in rest {
                let (keys, values) = slot
                    .remove(&state)
                    .ok_or_else(|| error("pooled attention slot cache"))?;
                slot.insert(state, (keys.reshape(&dims)?, values.reshape(&dims)?));
            }
            return recorded.map(|()| true);
        }
        let mut owned = self
            .states
            .remove(&state)
            .ok_or_else(|| error("pooled attention state"))?;
        if owned.len() != KV_TENSORS {
            return Err(error("pooled attention state layout"));
        }
        let keys = owned.remove(0);
        let values = owned.remove(0);
        let cache = Cache {
            keys,
            values,
            scales: [1.0, 1.0],
            per_lane: true,
        };
        let (keys, values, recorded) =
            self.record_lanes::<f32>(scope, node, arena, cache, metadata)?;
        owned.push(keys);
        owned.push(values);
        self.states.insert(state, owned);
        recorded.map(|()| true)
    }

    /// Returns the KV whatever the recording did, so its owner can restore it.
    fn record_lanes<E: DType>(
        &mut self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
        cache: Cache<E>,
        metadata: &Tensor<i32>,
    ) -> StdResult<Recorded<E>, DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            window,
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let Cache {
            keys,
            values,
            scales,
            per_lane,
        } = cache;
        let lanes = arena.lanes();
        let table = self
            .table_root
            .ok_or_else(|| error("pooled attention tables"))?;
        let groups = if per_lane { lanes * kv_heads } else { kv_heads };
        let rows = keys.size() / (groups * head_dim);
        let original_kv = keys
            .shape()
            .iter()
            .map(|&dim| usize::try_from(dim).map_err(error))
            .collect::<StdResult<Vec<_>, _>>()?;
        let mut keys = keys.reshape(&[groups, rows, head_dim])?;
        let mut values = values.reshape(&[groups, rows, head_dim])?;
        let flag = i32::from(per_lane);
        let recorded = (|| -> StdResult<(), DeviceError> {
            Self::record_lanes_append::<E>(
                scope,
                node,
                arena,
                (&mut keys, &mut values),
                (table, metadata),
                (scales, flag, rows),
            )?;
            let work = self
                .lanes_attention
                .as_deref_mut()
                .and_then(|work| work.0.get_mut(&(query_heads, head_dim)))
                .ok_or_else(|| error("pooled attention scratch"))?;
            let parts = work.parts;
            scope.record(
                kernels::partial_gqa_lanes(
                    (&mut work.numerator).partition([1, GROUPED_ROWS, head_dim]),
                    (&mut work.maxima).partition([1, GROUPED_ROWS]),
                    (&mut work.sums).partition([1, GROUPED_ROWS]),
                    &arena.get(node.inputs[0]).map_err(error)?.view(&[
                        lanes * kv_heads,
                        query_heads / kv_heads,
                        head_dim,
                    ])?,
                    metadata,
                    &keys,
                    &values,
                    table,
                    i32::try_from(window.unwrap_or(0)).map_err(error)?,
                    scales[0],
                    scales[1],
                )
                .generics(vec![
                    E::DTYPE.as_str().into(),
                    head_dim.to_string(),
                    parts.to_string(),
                    kv_heads.to_string(),
                    flag.to_string(),
                ])
                .compile_options(super::attention_decode::grouped_options()),
            )?;
            self.record_lanes_merge(scope, node, arena)?;
            Ok(())
        })();
        if per_lane {
            return Ok((
                keys.reshape(&original_kv)?,
                values.reshape(&original_kv)?,
                recorded,
            ));
        }
        Ok((keys, values, recorded))
    }

    fn record_lanes_append<E: DType>(
        scope: &Scope,
        node: &TensorNode,
        arena: &ActivationArena,
        (keys, values): (&mut Tensor<E>, &mut Tensor<E>),
        (table, metadata): (&Tensor<i32>, &Tensor<i32>),
        (scales, per_lane, rows): ([f32; 2], i32, usize),
    ) -> StdResult<(), DeviceError> {
        let TensorOp::Attention {
            kv_heads, head_dim, ..
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let lanes = arena.lanes();
        let quantized = i32::from(E::DTYPE != f32::DTYPE);
        scope.record(
            attention::append_lanes(
                keys.partition([1, rows, head_dim]),
                values.partition([1, rows, head_dim]),
                &arena
                    .get(node.inputs[1])
                    .map_err(error)?
                    .view(&[lanes * kv_heads, head_dim])?,
                &arena
                    .get(node.inputs[2])
                    .map_err(error)?
                    .view(&[lanes * kv_heads, head_dim])?,
                metadata,
                table,
                scales[0],
                scales[1],
            )
            .generics(vec![
                E::DTYPE.as_str().into(),
                head_dim.to_string(),
                rows.to_string(),
                quantized.to_string(),
                kv_heads.to_string(),
                lanes.to_string(),
                per_lane.to_string(),
            ]),
        )?;
        Ok(())
    }

    fn record_lanes_merge(
        &self,
        scope: &Scope,
        node: &TensorNode,
        arena: &mut ActivationArena,
    ) -> StdResult<(), DeviceError> {
        let TensorOp::Attention {
            query_heads,
            kv_heads,
            head_dim,
            ..
        } = node.op
        else {
            return Err(error("expected attention"));
        };
        let work = self
            .lanes_attention
            .as_deref()
            .and_then(|work| work.0.get(&(query_heads, head_dim)))
            .ok_or_else(|| error("pooled attention scratch"))?;
        let parts = work.parts;
        let lanes = arena.lanes();
        let slot = arena.slot(node.outputs[0]).map_err(error)?;
        let output = arena.buffers[slot]
            .take()
            .ok_or_else(|| error("pooled attention output"))?;
        let original = output
            .shape()
            .iter()
            .map(|&dim| usize::try_from(dim).map_err(error))
            .collect::<StdResult<Vec<_>, _>>()?;
        let mut output = output.reshape(&[lanes * query_heads, head_dim])?;
        let blocks = lanes * kv_heads;
        scope.record(
            kernels::merge_gqa(
                (&mut output).partition([1, head_dim]),
                &work
                    .numerator
                    .view(&[blocks, parts, GROUPED_ROWS, head_dim])?,
                &work.maxima.view(&[blocks, parts, GROUPED_ROWS])?,
                &work.sums.view(&[blocks, parts, GROUPED_ROWS])?,
                0,
            )
            .generics(vec![
                head_dim.to_string(),
                parts.to_string(),
                (query_heads / kv_heads).to_string(),
                GROUPED_ROWS.to_string(),
            ]),
        )?;
        arena.buffers[slot] = Some(output.reshape(&original)?);
        Ok(())
    }
}
