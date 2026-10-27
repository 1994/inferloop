//! CUDA payload and reuse adapter for the shared prefix lifecycle.
use super::{
    Sequence,
    prefix_state::{DraftSnapshot, ProgramSnapshot},
};
use crate::{constants::MTP_KV_OFFSET, device::CudaDevice};
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
}
impl Restore<'_> {
    pub(super) fn supports(sequence: &Sequence) -> bool {
        // Slot/home KV lives in a shared pool; a contiguous snapshot cannot describe it.
        !sequence.poisoned
            && sequence.readout == OutputReadout::Logits
            && sequence.slot.is_none()
            && sequence.home.is_none()
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
        let restored = (|| {
            snapshot
                .target
                .restore(self.device, &mut self.sequence.program, matched)?;
            if let (Some(spec), Some(draft)) = (&mut self.sequence.speculation, &snapshot.draft) {
                draft.program.restore(
                    self.device,
                    &mut spec.program,
                    matched.saturating_sub(MTP_KV_OFFSET),
                )?;
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
