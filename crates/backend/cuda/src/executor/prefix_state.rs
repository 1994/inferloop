//! Prefix snapshots preserve head-major KV strides and whole recurrent state.
use crate::constants::KV_CACHE_RANK;
use crate::device::asynchronous::block_on;
use crate::{
    device::CudaDevice,
    resident::{
        DeviceProgram, attention_states,
        fp8_cache::{Fp8Caches, KvLayout, KvState},
        shaped::Shaped,
        slot_batch::SlotPool,
    },
};
use infer_core::{Error, Result, TensorId};
use infer_cuda_kernels::runtime::f8e4m3fn;
use infer_ir::DataflowGraph;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::mem::forget;

/// Unquantized state by id, as a program holds it.
type States = BTreeMap<TensorId, Vec<Shaped<f32>>>;

#[derive(Default)]
pub struct ProgramSnapshot {
    states: States,
    kv: BTreeSet<TensorId>,
    fp8: Fp8Caches,
    /// The paged KV stays in the slot pool's pages, published in its ledger under the
    /// snapshot's tokens; only a slot-resident sequence of that pool can restore it.
    pages: bool,
    position: usize,
    pub bytes: u64,
}

impl ProgramSnapshot {
    #[cfg(test)]
    pub(super) fn capture(
        device: &CudaDevice,
        program: &DeviceProgram,
        graph: &DataflowGraph,
        covered: usize,
    ) -> Result<Self> {
        block_on(Self::capture_async(device, program, graph, covered))
    }

    #[expect(
        clippy::mem_forget,
        reason = "Unknown prefix-copy completion must retain unpublished destination allocations"
    )]
    pub(super) fn capture_async<'a>(
        device: &'a CudaDevice,
        program: &'a DeviceProgram,
        graph: &'a DataflowGraph,
        covered: usize,
    ) -> impl Future<Output = Result<Self>> + Send + 'a {
        let live_states = program.states();
        let live_fp8 = program.fp8_states();
        let position = program.position();
        async move {
            let kv = attention_states(graph);
            let mut states = BTreeMap::new();
            let mut bytes = 0_u64;
            for (id, tensors) in live_states {
                let mut copies = Vec::with_capacity(tensors.len());
                for tensor in tensors {
                    let mut shape = tensor.dims().to_vec();
                    if kv.contains(id) {
                        if shape.len() != KV_CACHE_RANK || shape[1] < covered {
                            return Err(Error::invariant("prefix KV shape"));
                        }
                        shape[1] = covered;
                    }
                    let len = shape.iter().product::<usize>();
                    let memory = device.tile().zeroed::<f32>(len).await?;
                    let copy = Shaped::new(memory, shape)?;
                    bytes = bytes
                        .checked_add((copy.len() * size_of::<f32>()) as u64)
                        .ok_or_else(|| Error::invalid("prefix byte overflow"))?;
                    copies.push(copy);
                }
                states.insert(*id, copies);
            }
            let mut fp8 = BTreeMap::new();
            for (id, source) in live_fp8 {
                if source.layout.rows < covered {
                    return Err(Error::invariant("prefix FP8 KV shape"));
                }
                let layout = KvLayout {
                    rows: covered,
                    ..source.layout
                };
                let allocate = async || device.tile().zeroed::<f8e4m3fn>(layout.len()).await;
                let keys = allocate().await?;
                let values = allocate().await?;
                bytes = bytes
                    .checked_add((keys.len() + values.len()) as u64)
                    .ok_or_else(|| Error::invalid("prefix byte overflow"))?;
                fp8.insert(
                    *id,
                    KvState {
                        keys,
                        values,
                        layout,
                    },
                );
            }
            // Allocate and validate every destination before queuing copies. Keep the whole
            // destination set alive through the drain even when a later copy fails.
            let mut copied = Ok(());
            'copies: {
                for (id, copies) in &mut states {
                    for (dst, src) in copies.iter().zip(&live_states[id]) {
                        if let Err(error) =
                            copy_state(device, dst, src, kv.contains(id), covered).await
                        {
                            copied = Err(error);
                            break 'copies;
                        }
                    }
                }
                for (id, dst) in &fp8 {
                    if let Err(error) = copy_kv(device, dst, &live_fp8[id], covered).await {
                        copied = Err(error);
                        break 'copies;
                    }
                }
            }
            if let Err(error) = device.fence().await {
                // These temporary destinations have not entered the cache yet. Unknown
                // completion must retain them, just as it retains the source executor.
                forget((states, fp8));
                return Err(error);
            }
            copied?;
            Ok(Self {
                states,
                kv,
                fp8,
                pages: false,
                position,
                bytes,
            })
        }
    }

    /// A snapshot of state already copied out in a private program's layout, such as a slot
    /// export; `graph` names which states are KV.
    pub(super) fn from_parts(
        states: States,
        fp8: Fp8Caches,
        graph: &DataflowGraph,
        position: usize,
    ) -> Result<Self> {
        let kv = attention_states(graph);
        let float = states.values().flatten().try_fold(0_u64, |sum, tensor| {
            sum.checked_add((tensor.len() * size_of::<f32>()) as u64)
        });
        let packed = fp8.values().try_fold(0_u64, |sum, state| {
            sum.checked_add((state.keys.len() + state.values.len()) as u64)
        });
        let bytes = float
            .zip(packed)
            .and_then(|(float, packed)| float.checked_add(packed))
            .ok_or_else(|| Error::invalid("prefix byte overflow"))?;
        Ok(Self {
            states,
            kv,
            fp8,
            pages: false,
            position,
            bytes,
        })
    }

    /// A snapshot of a slot's unpaged state whose KV stays published in the pool's pages.
    pub(super) fn with_pages(
        states: States,
        graph: &DataflowGraph,
        position: usize,
    ) -> Result<Self> {
        Ok(Self {
            pages: true,
            ..Self::from_parts(states, Fp8Caches::new(), graph, position)?
        })
    }

    /// Whether the KV lives in a slot pool's published pages rather than in the snapshot.
    pub(super) const fn pages(&self) -> bool {
        self.pages
    }

    /// Bind the snapshot into a slot-resident program's slot: the slot maps the published
    /// pages or scatters KV into its blocks, copies recurrent state, and the program resumes
    /// after the prefix.
    pub(super) fn bind_slot(
        &self,
        device: &CudaDevice,
        pool: &mut SlotPool,
        (slot, program): (usize, &mut DeviceProgram),
        (tokens, covered): (&[u32], usize),
    ) -> Result<()> {
        if self.pages {
            if !pool.attach_pages(slot, tokens, device)? {
                return Err(Error::invariant("published prefix pages were reclaimed"));
            }
            pool.bind_attached(slot, &self.states, covered, device)?;
        } else {
            pool.bind(slot, &self.states, &self.fp8, covered, device)?;
        }
        device.drain()?;
        program.set_position(self.position);
        Ok(())
    }

    pub(super) fn restore(
        &self,
        device: &CudaDevice,
        program: &mut DeviceProgram,
        covered: usize,
    ) -> Result<()> {
        block_on(self.restore_async(device, program, covered))
    }

    pub(super) async fn restore_async(
        &self,
        device: &CudaDevice,
        program: &mut DeviceProgram,
        covered: usize,
    ) -> Result<()> {
        if self.pages {
            return Err(Error::invariant(
                "paged prefix restored into a private program",
            ));
        }
        let live = program.states();
        if live.len() != self.states.len() {
            return Err(Error::invariant("prefix state set mismatch"));
        }
        for (id, cached) in &self.states {
            let tensors = live
                .get(id)
                .ok_or_else(|| Error::invariant("prefix state missing"))?;
            if tensors.len() != cached.len() {
                return Err(Error::invariant("prefix state arity"));
            }
            for (dst, src) in tensors.iter().zip(cached) {
                copy_state(device, dst, src, self.kv.contains(id), covered).await?;
            }
        }
        let live = program.fp8_states();
        if live.len() != self.fp8.len() {
            return Err(Error::invariant("prefix FP8 state set mismatch"));
        }
        for (id, cached) in &self.fp8 {
            let dst = live
                .get(id)
                .ok_or_else(|| Error::invariant("prefix FP8 state missing"))?;
            copy_kv(device, dst, cached, covered).await?;
        }
        device.fence().await?;
        program.set_position(self.position);
        Ok(())
    }
}

async fn copy_state(
    device: &CudaDevice,
    dst: &Shaped<f32>,
    src: &Shaped<f32>,
    kv: bool,
    covered: usize,
) -> Result<()> {
    if !kv {
        if dst.dims() != src.dims() {
            return Err(Error::invariant("prefix recurrent shape"));
        }
        return dst.copy_from(device, src);
    }
    if dst.dims().len() != KV_CACHE_RANK || src.dims().len() != KV_CACHE_RANK {
        return Err(Error::invariant("prefix KV stride mismatch"));
    }
    if let Err(error) = dst.copy_head_prefix(device, src, covered) {
        device.fence().await?;
        return Err(error);
    }
    Ok(())
}

/// Copy the first `covered` positions of an FP8 cache; a failed enqueue fences first, since
/// earlier copies of the batch may still be running.
async fn copy_kv(
    device: &CudaDevice,
    dst: &KvState<f8e4m3fn>,
    src: &KvState<f8e4m3fn>,
    covered: usize,
) -> Result<()> {
    if let Err(error) = dst.copy_prefix_from(device, src, covered) {
        device.fence().await?;
        return Err(error);
    }
    Ok(())
}

pub struct DraftSnapshot {
    pub program: ProgramSnapshot,
    pub hidden: Vec<f32>,
}

#[cfg(test)]
#[path = "../../tests/unit/executor_prefix_state.rs"]
mod tests;
