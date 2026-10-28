//! CUDA payload and reuse adapter for the shared prefix lifecycle.
use super::{
    Sequence,
    prefix_state::{DraftSnapshot, ProgramSnapshot},
};
use crate::{constants::MTP_KV_OFFSET, device::CudaDevice, resident::slot_batch::SlotPool};
use infer_core::{Error, Result};
use infer_ir::OutputReadout;
use infer_state::cache::{
    PrefixCache as SharedPrefixCache, PrefixReuse, PrefixSnapshot, SnapshotScan,
};

/// Prompt positions between prefix-cache snapshots, where a prefill state becomes a reusable
/// prompt prefix. Each snapshot copies the covered KV and
/// the recurrent state and synchronizes, so snapshotting every 2048-token chunk cost 0.12 s
/// of a 32K prompt's TTFT; at 4096 a later turn recomputes at most one interval.
pub(super) const PREFIX_GRANULARITY: usize = 4096;

pub struct Snapshot {
    pub target: ProgramSnapshot,
    pub draft: Option<DraftSnapshot>,
}
pub type CachedPrefix = PrefixSnapshot<Snapshot>;
pub type PrefixCache = SharedPrefixCache<Snapshot, SnapshotScan>;

pub(super) struct Restore<'a> {
    pub device: &'a CudaDevice,
    pub sequence: &'a mut Sequence,
    /// The target and draft slot pools a slot-resident sequence binds the snapshot into.
    pub pools: Option<(&'a mut SlotPool, Option<&'a mut SlotPool>)>,
}
impl Restore<'_> {
    pub(super) fn supports(sequence: &Sequence) -> bool {
        // A leased slot's state already belongs to the pooled timeline; a slot-resident
        // sequence (`home`) owns its slot and binds the snapshot into it instead.
        !sequence.poisoned
            && sequence.readout == OutputReadout::Logits
            && sequence.slot.is_none()
            // A snapshot does not carry the draft ring the restored prefix would need.
            && sequence.draft.is_none()
    }
}
impl PrefixReuse<Snapshot> for Restore<'_> {
    type Output = usize;
    type Error = Error;
    fn compatible(&self, tokens: &[u32], snapshot: &Snapshot) -> bool {
        tokens.len() <= self.sequence.capacity
            && self.sequence.speculation.is_some() == snapshot.draft.is_some()
    }
    fn reuse(&mut self, tokens: &[u32], snapshot: &Snapshot) -> Result<usize> {
        let matched = tokens.len();
        let draft_covered = matched.saturating_sub(MTP_KV_OFFSET);
        let restored = (|| {
            match (self.sequence.home, self.pools.as_mut()) {
                (Some(slot), Some((target, draft_pool))) => {
                    snapshot.target.bind_slot(
                        self.device,
                        target,
                        (slot, &mut self.sequence.program),
                        matched,
                    )?;
                    if let (Some(spec), Some(draft)) =
                        (&mut self.sequence.speculation, &snapshot.draft)
                    {
                        let pool = draft_pool
                            .as_deref_mut()
                            .ok_or_else(|| Error::invariant("slot draft pool"))?;
                        draft.program.bind_slot(
                            self.device,
                            pool,
                            (slot, &mut spec.program),
                            draft_covered,
                        )?;
                    }
                }
                (Some(_), None) => return Err(Error::invariant("slot-resident restore pool")),
                (None, _) => {
                    snapshot
                        .target
                        .restore(self.device, &mut self.sequence.program, matched)?;
                    if let (Some(spec), Some(draft)) =
                        (&mut self.sequence.speculation, &snapshot.draft)
                    {
                        draft
                            .program
                            .restore(self.device, &mut spec.program, draft_covered)?;
                    }
                }
            }
            if let (Some(spec), Some(draft)) = (&mut self.sequence.speculation, &snapshot.draft) {
                spec.last_hidden.clone_from(&draft.hidden);
                spec.prompt_len = None;
            }
            Ok(())
        })();
        if let Err(error) = restored {
            self.sequence.poisoned = true;
            // Even a validation error after an earlier copy must settle source/destination reads.
            self.device.drain()?;
            return Err(error);
        }
        self.sequence.history.clear();
        self.sequence.history.extend_from_slice(tokens);
        Ok(matched)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/executor_prefix.rs"]
mod tests;
