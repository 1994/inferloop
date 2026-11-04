use crate::device::{CudaDevice, device_error};
use cuda_core::f8e4m3fn;
use cutile::prelude::*;
use infer_core::{Error, Result, TensorId};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{DataflowGraph, StateKind, TensorStorage};
use std::collections::BTreeMap;
use std::sync::Arc;

pub type Fp8Caches = BTreeMap<TensorId, (Tensor<f8e4m3fn>, Tensor<f8e4m3fn>)>;

/// The allocations behind a set of FP8 KV caches. Views cut from it hold the memory, so an owner
/// that keeps it can hand out caches without aliasing tensors it does not own.
#[derive(Clone, Default)]
pub(super) struct Fp8Memory(BTreeMap<TensorId, (Allocation<f8e4m3fn>, Allocation<f8e4m3fn>)>);

impl Fp8Memory {
    /// Another view of every cache, sharing its memory.
    /// # Errors
    /// Propagates view failures.
    pub(super) fn views(&self) -> Result<Fp8Caches> {
        let whole = |memory: &Allocation<f8e4m3fn>| memory.view(0..memory.len(), &[memory.len()]);
        self.0
            .iter()
            .map(|(id, (keys, values))| Ok((*id, (whole(keys)?, whole(values)?))))
            .collect()
    }

    /// Device bytes of every cache: one per FP8 element.
    pub(super) fn bytes(&self) -> u64 {
        self.0
            .values()
            .map(|(keys, values)| (keys.len() + values.len()) as u64)
            .sum()
    }
}

/// FP8 KV caches of `capacity` positions for every scaled attention state.
/// # Errors
/// As [`allocate_memory`].
pub(super) fn allocate(
    device: &CudaDevice,
    graph: &DataflowGraph,
    capacity: usize,
    scales: &BTreeMap<TensorId, [f32; 2]>,
) -> Result<Fp8Caches> {
    allocate_memory(device, graph, capacity, scales)?.views()
}

/// The allocations for [`allocate`].
/// # Errors
/// Rejects invalid bindings or scales, budgets past the device, and failed allocation.
pub(super) fn allocate_memory(
    device: &CudaDevice,
    graph: &DataflowGraph,
    capacity: usize,
    scales: &BTreeMap<TensorId, [f32; 2]>,
) -> Result<Fp8Memory> {
    let mut plan = Vec::new();
    let mut bytes = 0usize;
    let budget = device
        .allocatable_memory_bytes()?
        .saturating_sub(device.profile()?.device_headroom_bytes());
    for spec in &graph.tensors {
        let Some(scale) = scales.get(&spec.id) else {
            continue;
        };
        if !matches!(
            spec.storage,
            TensorStorage::State {
                kind: StateKind::AttentionKv,
                ..
            }
        ) || spec.shape.len() != 2
            || scale.iter().any(|s| !s.is_finite() || *s <= 0.0)
        {
            return Err(Error::invalid("FP8 KV binding/scales"));
        }
        let size = capacity
            .checked_mul(spec.shape[1])
            .ok_or_else(|| Error::invalid("FP8 KV size overflow"))?;
        bytes = size
            .checked_mul(2)
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| Error::invalid("FP8 KV budget overflow"))?;
        if bytes as u64 > budget {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "FP8 KV exceeds available device memory minus 1 GiB headroom",
            ));
        }
        plan.push((spec.id, size));
    }
    if plan.len() != scales.len() {
        return Err(Error::invalid("unknown FP8 KV state"));
    }
    let mut result = BTreeMap::new();
    for (id, size) in plan {
        let zeros = || -> Result<Allocation<f8e4m3fn>> {
            Allocation::adopt(Arc::new(
                api::zeros::<f8e4m3fn>(&[size])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            ))
        };
        result.insert(id, (zeros()?, zeros()?));
    }
    Ok(Fp8Memory(result))
}
