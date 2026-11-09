//! Complete paired evidence; new choices preserve the private reduction tree.
use super::capture::tile;
use super::{Choice, Key, baseline, candidates_for, prompt_searchable, searchable};
use crate::tuning::search::{self, Protocol, Record};

pub type Trial = search::Trial<Choice>;
pub type Entry = Record<Key, Choice>;

/// The search a key runs: its baseline's candidates.
pub(super) fn protocol(key: &Key) -> Protocol<Choice> {
    let shape = [key.native.m, key.native.n, key.native.k];
    Protocol {
        baseline: key.baseline,
        candidates: candidates_for(key.baseline, shape),
    }
}

impl Entry {
    /// The evidence is a complete run of the protocol the key's geometry selects. The legacy
    /// baseline retains its own already-admitted arithmetic when the row count selects tensor
    /// cores, so no other choice may newly promote tensor-core arithmetic.
    pub(crate) fn valid(&self) -> bool {
        let shape = [self.key.native.m, self.key.native.n, self.key.native.k];
        let baseline = if self.key.baseline.prompt_family() {
            Choice::PromptBase
        } else {
            baseline(shape)
        };
        let prompt = baseline.prompt_family();
        let allowed = |choice: Choice| {
            choice.prompt_family() == prompt
                && (!choice.native_family() || baseline.native_family())
        };
        (if prompt {
            prompt_searchable(shape)
        } else {
            searchable(shape)
        }) && tile(&self.key).is_ok()
            && self.key.baseline == baseline
            && !self.key.native.kernel_fingerprint.is_empty()
            && self
                .key
                .native
                .driver_build
                .as_ref()
                .is_none_or(|build| !build.is_empty())
            && self.evidence.valid(&protocol(&self.key), allowed)
    }
}
