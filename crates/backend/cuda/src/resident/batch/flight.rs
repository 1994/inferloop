//! Completion-owned pooled decode. Cursors remain committed until readback completes.
use super::{BatchReadout, SlotDecodeGraph, SlotLane};
use crate::device::{CudaDevice, ReadbackSource};
use infer_core::{Error, Result};
use std::sync::Arc;

pub(super) struct DecodeFlight {
    slots: Vec<usize>,
    device_tokens: bool,
}

impl SlotDecodeGraph {
    pub(crate) fn submit_lanes(
        &mut self,
        device: &CudaDevice,
        lanes: &[SlotLane],
        capacity: usize,
        vocabulary: usize,
        plain_greedy: bool,
    ) -> Result<()> {
        if self.pending.is_some() || self.external_hidden {
            return Err(Error::invalid("busy or external-input decode graph"));
        }
        Self::stage_lanes(
            &mut self.metadata_upload,
            &self.cursors,
            self.width,
            false,
            lanes,
            capacity,
            vocabulary,
        )?;
        device.copy_host_slice(&mut self.metadata_owner, &self.metadata_upload)?;
        let device_tokens = plain_greedy && self.token_readout.is_some();
        let vocabulary = self.logits.size() / self.width;
        let mut sources = Vec::with_capacity(2 * lanes.len());
        let slots: Vec<_> = lanes.iter().map(|&(slot, ..)| slot).collect();
        for &slot in &slots {
            sources.push(None);
            sources.push((!device_tokens).then(|| ReadbackSource {
                tensor: Arc::clone(&self.logits),
                skip: slot * vocabulary,
                len: vocabulary,
            }));
        }
        self.pending = Some(DecodeFlight {
            slots,
            device_tokens,
        });
        if device_tokens {
            self.token_readout
                .as_mut()
                .ok_or_else(|| Error::invariant("token readout"))?
                .submit(
                    device,
                    &self.graph,
                    &mut self.readbacks,
                    &sources,
                    &self
                        .pending
                        .as_ref()
                        .ok_or_else(|| Error::invariant("decode flight"))?
                        .slots,
                )
        } else {
            self.readbacks.submit(device, &self.graph, &sources)
        }
    }

    pub(crate) fn poll_lanes(&mut self) -> Result<Option<BatchReadout>> {
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| Error::invalid("no decode flight"))?;
        let readout = if pending.device_tokens {
            let Some((rows, tokens)) = self
                .token_readout
                .as_mut()
                .ok_or_else(|| Error::invariant("token readout"))?
                .poll(&mut self.readbacks, &pending.slots)?
            else {
                return Ok(None);
            };
            BatchReadout {
                rows,
                tokens: Some(tokens.into_iter().map(Some).collect()),
            }
        } else {
            let Some(rows) = self.readbacks.poll()? else {
                return Ok(None);
            };
            let mut rows = rows.into_iter();
            let output = pending
                .slots
                .iter()
                .map(|_| {
                    Ok((
                        rows.next()
                            .ok_or_else(|| Error::invariant("decode hidden readback"))?,
                        rows.next()
                            .ok_or_else(|| Error::invariant("decode logits readback"))?,
                    ))
                })
                .collect::<Result<_>>()?;
            BatchReadout::host(output)
        };
        for &slot in &pending.slots {
            self.cursors[slot] += 1;
        }
        self.pending = None;
        Ok(Some(readout))
    }
}
