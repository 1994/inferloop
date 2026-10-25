//! Batched external-hidden execution and attention state transfer.
use super::{BatchOutput, SlotCopy, SlotLane, SlotPool};
use crate::device::CudaDevice;
#[cfg(test)]
use crate::device::asynchronous::block_on;
use crate::resident::batch::BatchReadout;
use infer_core::{Error, Result};

impl SlotPool {
    /// Discard an unaccepted attention-only draft suffix without copying KV data.
    pub(crate) fn rewind_external(&mut self, slot: usize, position: usize) -> Result<()> {
        if self
            .plan
            .iter()
            .any(|(_, copy)| matches!(copy, SlotCopy::Whole))
            || self
                .cursors()
                .get(slot)
                .is_none_or(|&cursor| position > cursor)
        {
            return Err(Error::invalid("draft rewind range or recurrent state"));
        }
        let graph = self
            .decode
            .as_mut()
            .filter(|graph| graph.external_hidden)
            .ok_or_else(|| Error::invalid("draft rewind requires external decode"))?;
        graph.set_cursor(slot, position)
    }

    /// Stage one lane's external hidden row through its pinned ring and enqueue the H2D
    /// copy; never syncs: ordering comes from the shared stream, completion from the
    /// next readback or the pool drop barrier.
    fn upload_external(&mut self, device: &CudaDevice, slot: usize, values: &[f32]) -> Result<()> {
        let ring = self
            .pinned_external
            .get_mut(slot)
            .filter(|ring| !ring.is_empty())
            .ok_or_else(|| Error::invalid("external hidden staging"))?;
        let cursor = self
            .pinned_cursor
            .get_mut(slot)
            .ok_or_else(|| Error::invariant("external staging cursor"))?;
        let index = *cursor % ring.len();
        let pinned = ring
            .get_mut(index)
            .ok_or_else(|| Error::invariant("external staging ring"))?;
        *cursor = cursor.wrapping_add(1);
        if values.len() != pinned.len() {
            return Err(Error::invalid("external hidden shape"));
        }
        pinned.as_mut_slice().copy_from_slice(values);
        let buffer = self
            .lane_external
            .get_mut(slot)
            .ok_or_else(|| Error::invalid("external hidden slot"))?;
        device.copy_h2d_pinned(buffer, pinned, values.len())?;
        self.external_uploaded[slot] = true;
        Ok(())
    }

    /// Supply one external hidden row per active lane through the pinned rings.
    fn upload_lanes(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
    ) -> Result<()> {
        let external = self
            .decode
            .as_ref()
            .ok_or_else(|| Error::invalid("external hidden requires a decode pool"))?
            .external_hidden;
        if !external || lanes.len() != hidden.len() {
            return Err(Error::invalid("external hidden lane count or graph"));
        }
        for (&(slot, ..), values) in lanes.iter().zip(hidden) {
            self.upload_external(device, slot, values)?;
        }
        Ok(())
    }

    /// Supply one external hidden row per active lane, retaining upload owners for every
    /// graph update (including inactive slots on later replays).
    #[cfg(test)]
    pub(crate) fn run_external(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
    ) -> Result<BatchOutput> {
        block_on(self.run_external_async(device, lanes, hidden))
    }

    pub(crate) async fn run_external_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
    ) -> Result<BatchOutput> {
        self.run_external_readout_async(device, lanes, hidden, true)
            .await
    }

    #[cfg(test)]
    pub(crate) fn run_external_readout(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
        read_logits: bool,
    ) -> Result<BatchOutput> {
        block_on(self.run_external_readout_async(device, lanes, hidden, read_logits))
    }

    pub(crate) async fn run_external_readout_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
        read_logits: bool,
    ) -> Result<BatchOutput> {
        self.upload_lanes(device, lanes, hidden)?;
        self.decode
            .as_mut()
            .ok_or_else(|| Error::invariant("external decode graph"))?
            .run_lanes_async(device, lanes, self.capacity, self.vocabulary, read_logits)
            .await
    }

    pub(crate) async fn run_external_selected_async(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
        plain_greedy: bool,
    ) -> Result<BatchReadout> {
        if !plain_greedy {
            return self
                .run_external_async(device, lanes, hidden)
                .await
                .map(BatchReadout::host);
        }
        self.upload_lanes(device, lanes, hidden)?;
        self.decode
            .as_mut()
            .ok_or_else(|| Error::invariant("external decode graph"))?
            .run_lanes_readout_async(
                device,
                lanes,
                self.capacity,
                self.vocabulary,
                true,
                plain_greedy,
            )
            .await
    }

    /// Draft catch-up replay: the state-only graph's outputs are never read, so the
    /// launch detaches — no readback, no sync. Completion rides the next pool readback
    /// or the drop barrier; stream order keeps it ahead of any later replay.
    pub(crate) fn run_external_detached(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        hidden: &[&[f32]],
    ) -> Result<()> {
        self.upload_lanes(device, lanes, hidden)?;
        let (capacity, vocabulary) = (self.capacity, self.vocabulary);
        self.decode
            .as_mut()
            .ok_or_else(|| Error::invariant("external decode graph"))?
            .replay_detached(device, lanes, capacity, vocabulary)
    }

    /// Copy a draft slot back to its private program before acceptance/rejection replay.
    /// Recurrent state requires checkpoints and is deliberately rejected here.
    pub(crate) fn export_attention(
        &self,
        slot: usize,
        target: &mut super::super::DeviceProgram,
        device: &CudaDevice,
    ) -> Result<()> {
        let rows = *self
            .cursors()
            .get(slot)
            .ok_or_else(|| Error::invalid("export slot"))?;
        let capacity = target.external_attention_capacity(rows)?;
        for (id, copy) in &self.plan {
            match copy {
                SlotCopy::Kv => Self::copy_kv_states(
                    target.states_mut(),
                    &self.states[slot],
                    *id,
                    rows,
                    capacity,
                    device,
                )?,
                SlotCopy::KvFp8 => Self::copy_kv_fp8(
                    target.fp8_states_mut(),
                    &self.fp8[slot],
                    *id,
                    rows,
                    capacity,
                    device,
                )?,
                SlotCopy::Whole => {
                    return Err(Error::unsupported(
                        "recurrent slot export requires checkpoints",
                    ));
                }
            }
        }
        target.adopt_attention_prefix(rows)
    }
}
