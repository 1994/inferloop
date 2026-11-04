//! Prompt-graph scratch shared by every sequence of one program.
//!
//! A sequence's prompt graphs need a full-width activation arena, NVFP4 split partials, the
//! chunked recurrent workspace and the draft fusion workspace: about 1 GB at 2048 lanes for the
//! target and as much again for the draft. Sequences replay their prompt graphs one at a time on
//! the device stream and read nothing from this scratch after their own replay, so one
//! allocation per prompt width serves every admitted sequence instead of one per sequence.
use super::ActivationArena;
use crate::constants::PREFILL_SPLIT_K;
use crate::device::{CudaDevice, device_error};
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::Buffer;
use infer_cuda_kernels::tile::runtime::Allocation;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// NVFP4 split-K partials the prompt GEMMs accumulate into: one `[PREFILL_SPLIT_K * lanes, rows]`
/// window per projection output width. A narrower prompt rung shares a wider rung's memory and
/// addresses only its leading windows.
#[derive(Clone, Default)]
pub(super) struct Partials {
    memory: BTreeMap<usize, Allocation<f32>>,
    lanes: usize,
}

impl Partials {
    /// Zeroed partials of `lanes` prompt lanes for every output width in `widths`.
    /// # Errors
    /// Rejects overflowing sizes and failed allocation.
    pub(super) fn allocate(
        device: &CudaDevice,
        widths: impl IntoIterator<Item = usize>,
        lanes: usize,
    ) -> Result<Self> {
        let mut memory = BTreeMap::new();
        for rows in widths {
            let len = PREFILL_SPLIT_K
                .checked_mul(lanes)
                .and_then(|split| split.checked_mul(rows))
                .ok_or_else(|| Error::invalid("prefill split partials overflow"))?;
            let zeros = api::zeros::<f32>(&[len])
                .sync_on(&device.stream)
                .map_err(device_error)?;
            memory.insert(rows, Allocation::adopt(Arc::new(zeros))?);
        }
        Ok(Self { memory, lanes })
    }

    /// The leading windows of `lanes` prompt lanes; the memory is shared.
    /// # Errors
    /// Rejects more lanes than these partials hold.
    pub(super) fn narrowed(&self, lanes: usize) -> Result<Self> {
        if lanes > self.lanes {
            return Err(Error::invalid("narrowed prefill split partials"));
        }
        Ok(Self {
            memory: self.memory.clone(),
            lanes,
        })
    }

    /// The partials for an output `rows` wide.
    /// # Errors
    /// Rejects a width the graph has no split projection for.
    pub(super) fn buffer(&self, rows: usize) -> Result<Buffer<'_, f32>> {
        let memory = self
            .memory
            .get(&rows)
            .ok_or_else(|| Error::invalid("prefill split partials"))?;
        memory
            .buffer()
            .slice(0..PREFILL_SPLIT_K * self.lanes * rows)
    }
}

/// The scratch one prompt width's graphs capture against.
pub(super) struct PromptScratch {
    /// Master arena; each program captures through [`ActivationArena::shared_view`].
    pub(super) arena: ActivationArena,
    /// Final prompt row the one-row logits projection reads, when the vocabulary projection
    /// supports it.
    pub(super) row: Option<Tensor<f32>>,
    pub(super) partials: Partials,
    pub(super) recurrent: super::recurrent_prefill::Workspace,
    pub(super) fusion: Option<super::fusion_batch::Workspace>,
    /// The wider scratch whose storage this one's arena, partials and fusion rows view; kept
    /// alive with them and declared last so the views drop first.
    _wider: Option<Arc<Mutex<Self>>>,
}

impl PromptScratch {
    pub(super) const fn new(
        arena: ActivationArena,
        row: Option<Tensor<f32>>,
        partials: Partials,
        recurrent: super::recurrent_prefill::Workspace,
        fusion: Option<super::fusion_batch::Workspace>,
    ) -> Self {
        Self {
            arena,
            row,
            partials,
            recurrent,
            fusion,
            _wider: None,
        }
    }

    /// Scratch for a narrower prompt rung whose arena, partials and fusion rows view `wider`'s
    /// storage; holding `wider` keeps that storage alive for as long as the views exist.
    pub(super) fn viewing(
        (arena, row, partials): (ActivationArena, Option<Tensor<f32>>, Partials),
        recurrent: super::recurrent_prefill::Workspace,
        fusion: Option<super::fusion_batch::Workspace>,
        wider: Arc<Mutex<Self>>,
    ) -> Self {
        Self {
            arena,
            row,
            partials,
            recurrent,
            fusion,
            _wider: Some(wider),
        }
    }
}

/// Shared prompt scratch of one program, keyed by prompt width.
#[derive(Default)]
pub struct Registry(Mutex<BTreeMap<usize, Arc<Mutex<PromptScratch>>>>);

impl Registry {
    /// Whether a width's storage is already physically allocated, including idle cache entries.
    pub(crate) fn is_allocated(&self, width: usize) -> Result<bool> {
        let entries = self
            .0
            .lock()
            .map_err(|_| Error::invariant("prompt scratch registry poisoned"))?;
        Ok(entries.contains_key(&width))
    }

    /// Release only entries no captured program owns, after draining detached replays.
    /// # Errors
    /// Reports a poisoned registry or failure to synchronize deferred device work.
    pub(crate) fn reclaim_unused(&self, device: &CudaDevice) -> Result<usize> {
        device.reclaim_barrier()?;
        let mut entries = self
            .0
            .lock()
            .map_err(|_| Error::invariant("prompt scratch registry poisoned"))?;
        let before = entries.len();
        entries.retain(|_, entry| Arc::strong_count(entry) > 1);
        let released = before - entries.len();
        drop(entries);
        if released != 0 {
            device.reclaim_barrier()?;
        }
        Ok(released)
    }

    /// The widest allocated scratch of at least `rows` lanes, which another graph may view.
    /// # Errors
    /// Reports a poisoned registry.
    pub(super) fn covering(&self, rows: usize) -> Result<Option<Arc<Mutex<PromptScratch>>>> {
        let entries = self
            .0
            .lock()
            .map_err(|_| Error::invariant("prompt scratch registry poisoned"))?;
        Ok(entries
            .range(rows..)
            .next_back()
            .map(|(_, s)| Arc::clone(s)))
    }

    /// The scratch for `width`, allocated by `allocate` on first use. `allocate` receives the
    /// widest allocated scratch wider than `width`, which a narrower ladder rung views.
    /// # Errors
    /// Propagates allocation failures; a failed allocation is retried by the next caller.
    pub(super) fn get_or_allocate(
        &self,
        width: usize,
        allocate: impl FnOnce(Option<&Arc<Mutex<PromptScratch>>>) -> Result<PromptScratch>,
    ) -> Result<Arc<Mutex<PromptScratch>>> {
        let mut entries = self
            .0
            .lock()
            .map_err(|_| Error::invariant("prompt scratch registry poisoned"))?;
        let scratch = if let Some(scratch) = entries.get(&width) {
            Arc::clone(scratch)
        } else {
            let wider = entries
                .range(width + 1..)
                .next_back()
                .map(|(_, wider)| wider);
            let scratch = Arc::new(Mutex::new(allocate(wider)?));
            entries.insert(width, Arc::clone(&scratch));
            scratch
        };
        drop(entries);
        Ok(scratch)
    }
}
