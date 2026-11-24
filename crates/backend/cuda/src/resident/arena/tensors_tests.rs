//! Arena slots as cuTile tensors, for tests that fill them from the host or launch reference
//! kernels directly.
use super::ActivationArena;
use cutile::prelude::Tensor;
use infer_core::Result;

impl ActivationArena {
    /// Slot `index` as a tensor holding a share of its memory.
    /// # Errors
    /// Rejects an index outside the arena.
    pub(crate) fn output(&self, index: usize) -> Result<Tensor<f32>> {
        let slot = self.at(index)?;
        slot.memory
            .view(slot.offset..slot.offset + slot.len, &[slot.len])
    }
}
