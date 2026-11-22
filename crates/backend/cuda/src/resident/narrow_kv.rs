use crate::constants::KV_CACHE_RANK;
use crate::device::CudaDevice;
use infer_core::{Error, Result, TensorId};
use infer_cuda_kernels::contract::{KvCache, KvElement, Width};
use infer_cuda_kernels::runtime::DType;
use infer_cuda_kernels::runtime::{
    Allocation, DeviceCopy, Element, HeadMajor, Transfers, f8e4m3fn, f16,
};
use infer_ir::{DataflowGraph, StateKind, TensorOp, TensorStorage};
use std::collections::BTreeMap;

/// Every narrow KV cache of a program or slot, by state.
pub type NarrowCaches = BTreeMap<TensorId, NarrowState>;

/// What every scaled ("narrow") attention state of a program stores. A state is narrow when the
/// program weights hold scales for it; F32 states are plain state tensors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NarrowElement {
    /// E4M3 with the checkpoint's static scales.
    #[default]
    Fp8,
    /// IEEE half precision, written and read through unit scales.
    F16,
}

impl NarrowElement {
    /// Device bytes of one element.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::Fp8 => 1,
            Self::F16 => 2,
        }
    }
}

/// One narrow attention state: FP8 or half-precision keys and values.
#[derive(Clone)]
pub enum NarrowState {
    Fp8(KvState<f8e4m3fn>),
    F16(KvState<f16>),
}

impl NarrowState {
    /// Zeroed caches of `layout` holding `element`.
    /// # Errors
    /// Reports a failed allocation.
    pub fn zeros(device: &CudaDevice, layout: KvLayout, element: NarrowElement) -> Result<Self> {
        Ok(match element {
            NarrowElement::Fp8 => Self::Fp8(KvState::zeros(device, layout)?),
            NarrowElement::F16 => Self::F16(KvState::zeros(device, layout)?),
        })
    }

    /// The FP8 caches, when these hold FP8.
    #[must_use]
    pub const fn as_fp8(&self) -> Option<&KvState<f8e4m3fn>> {
        match self {
            Self::Fp8(state) => Some(state),
            Self::F16(_) => None,
        }
    }

    /// The element these caches hold.
    #[must_use]
    pub const fn element(&self) -> NarrowElement {
        match self {
            Self::Fp8(_) => NarrowElement::Fp8,
            Self::F16(_) => NarrowElement::F16,
        }
    }

    /// The layout launches address these caches in.
    #[must_use]
    pub const fn layout(&self) -> KvLayout {
        match self {
            Self::Fp8(state) => state.layout,
            Self::F16(state) => state.layout,
        }
    }

    /// The caches as one launch addresses them, read with `scales`.
    #[must_use]
    pub fn cache(&self, scales: [f32; 2]) -> KvCache<'_> {
        match self {
            Self::Fp8(state) => state.cache(scales),
            Self::F16(state) => state.cache(scales),
        }
    }

    /// Device bytes of the key and the value cache.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        let layout = self.layout();
        (2 * layout.len() * self.element().bytes()) as u64
    }

    /// Whether `other` addresses the same memory.
    #[must_use]
    pub fn same_memory(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Fp8(a), Self::Fp8(b)) => a.same_memory(b),
            (Self::F16(a), Self::F16(b)) => a.same_memory(b),
            _ => false,
        }
    }

    /// [`KvState::copy_prefix_from`] between caches of the same element.
    /// # Errors
    /// Rejects caches of different elements, and as [`KvState::copy_prefix_from`].
    pub fn copy_prefix_from(&self, device: &CudaDevice, source: &Self, rows: usize) -> Result<()> {
        device
            .runtime()
            .enqueue(|transfers| self.copy_prefix_in(transfers, source, rows))
    }

    /// [`KvState::copy_prefix_in`] between caches of the same element.
    /// # Errors
    /// Rejects caches of different elements, and as [`KvState::copy_prefix_in`].
    pub fn copy_prefix_in(
        &self,
        transfers: &Transfers<'_>,
        source: &Self,
        rows: usize,
    ) -> Result<()> {
        match (self, source) {
            (Self::Fp8(to), Self::Fp8(from)) => to.copy_prefix_in(transfers, from, rows),
            (Self::F16(to), Self::F16(from)) => to.copy_prefix_in(transfers, from, rows),
            _ => Err(Error::invariant("KV prefix copy between cache elements")),
        }
    }
}

/// Head-major layout of one KV cache: `heads` groups of `rows` positions of `dim` values. A
/// paged slot's cache holds the pool's physical rows, not one sequence's capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    pub heads: usize,
    pub rows: usize,
    pub dim: usize,
}

impl KvLayout {
    /// Elements of one of the two caches.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.heads * self.rows * self.dim
    }

    /// The layout as `[heads, rows, dim]`.
    #[must_use]
    pub const fn dims(&self) -> [usize; KV_CACHE_RANK] {
        [self.heads, self.rows, self.dim]
    }
}

/// Keys and values of one attention state with the layout launches address them in, fixed
/// when the memory is allocated. A clone shares the memory, so a slot's view of a pool cache is
/// a clone and nothing reshapes a cache to carry its layout.
#[derive(Clone)]
pub struct KvState<T: DType> {
    pub keys: Allocation<T>,
    pub values: Allocation<T>,
    pub layout: KvLayout,
}

impl<T: DType + Element<Marker: KvElement>> KvState<T> {
    /// Zeroed caches of `layout`.
    /// # Errors
    /// Reports a failed allocation.
    pub fn zeros(device: &CudaDevice, layout: KvLayout) -> Result<Self>
    where
        T: DeviceCopy,
    {
        let runtime = device.runtime();
        Ok(Self {
            keys: runtime.zeros(layout.len())?,
            values: runtime.zeros(layout.len())?,
            layout,
        })
    }

    /// The caches as one launch addresses them, read with `scales`.
    #[must_use]
    pub fn cache(&self, scales: [f32; 2]) -> KvCache<'_> {
        KvCache {
            buffers: T::Marker::kv(self.keys.operand(), self.values.operand()),
            groups: self.layout.heads,
            dim: self.layout.dim,
            scales,
        }
    }

    /// Copy the first `rows` positions of every head of `source` into these caches on the
    /// device stream; the two may hold different row counts. Copying a cache onto itself moves
    /// nothing.
    /// # Errors
    /// Rejects another head count or dimension, a prefix past either cache, and a copy the
    /// driver does not enqueue.
    pub fn copy_prefix_from(&self, device: &CudaDevice, source: &Self, rows: usize) -> Result<()>
    where
        T::Marker: Width,
    {
        device
            .runtime()
            .enqueue(|transfers| self.copy_prefix_in(transfers, source, rows))
    }

    /// [`Self::copy_prefix_from`] as part of a larger transfer batch.
    /// # Errors
    /// As [`Self::copy_prefix_from`].
    pub fn copy_prefix_in(
        &self,
        transfers: &Transfers<'_>,
        source: &Self,
        rows: usize,
    ) -> Result<()>
    where
        T::Marker: Width,
    {
        let (to, from) = (self.layout, source.layout);
        if (to.heads, to.dim) != (from.heads, from.dim) || rows > to.rows || rows > from.rows {
            return Err(Error::invariant("KV prefix copy layout"));
        }
        if self.same_memory(source) {
            return Ok(());
        }
        for (dst, src) in [(&self.keys, &source.keys), (&self.values, &source.values)] {
            let dst = HeadMajor::new(dst.operand(), to.dims())?;
            let src = HeadMajor::new(src.operand(), from.dims())?;
            transfers.copy_heads(dst, src, rows)?;
        }
        Ok(())
    }

    /// Whether `other` addresses the same memory.
    #[must_use]
    pub fn same_memory(&self, other: &Self) -> bool {
        self.keys.buffer().address() == other.keys.buffer().address()
    }
}

/// Device bytes of every cache.
pub(super) fn bytes(caches: &NarrowCaches) -> u64 {
    caches.values().map(NarrowState::bytes).sum()
}

/// KV heads and head dimension of the attention reading `state`, as its layout's groups.
/// # Errors
/// Rejects a state no attention binds.
pub fn kv_geometry(graph: &DataflowGraph, state: TensorId) -> Result<(usize, usize)> {
    graph
        .nodes
        .iter()
        .find_map(|node| match node.op {
            TensorOp::Attention {
                kv_heads, head_dim, ..
            } if node.states.first() == Some(&state) => Some((kv_heads, head_dim)),
            _ => None,
        })
        .ok_or_else(|| Error::invalid("KV state without attention"))
}

/// `element` KV caches of `capacity` positions for every scaled attention state.
/// # Errors
/// Rejects invalid bindings or scales, budgets past the device, and failed allocation.
pub(super) fn allocate(
    device: &CudaDevice,
    graph: &DataflowGraph,
    capacity: usize,
    (scales, element): (&BTreeMap<TensorId, [f32; 2]>, NarrowElement),
) -> Result<NarrowCaches> {
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
            return Err(Error::invalid("narrow KV binding/scales"));
        }
        let (heads, dim) = kv_geometry(graph, spec.id)?;
        if heads.checked_mul(dim) != Some(spec.shape[1]) {
            return Err(Error::invalid("narrow KV width differs from its attention"));
        }
        let size = capacity
            .checked_mul(spec.shape[1])
            .ok_or_else(|| Error::invalid("narrow KV size overflow"))?;
        let layout = KvLayout {
            heads,
            rows: capacity,
            dim,
        };
        bytes = size
            .checked_mul(2 * element.bytes())
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| Error::invalid("narrow KV budget overflow"))?;
        if bytes as u64 > budget {
            return Err(Error::new(
                infer_core::ErrorCode::Capacity,
                "narrow KV exceeds available device memory minus 1 GiB headroom",
            ));
        }
        plan.push((spec.id, layout));
    }
    if plan.len() != scales.len() {
        return Err(Error::invalid("unknown narrow KV state"));
    }
    plan.into_iter()
        .map(|(id, layout)| Ok((id, NarrowState::zeros(device, layout, element)?)))
        .collect()
}
