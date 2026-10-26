//! Draft readout and external-hidden reuse on the program's owned CUDA stream.
use super::{CudaDevice, DeviceProgram, ExternalInput, ProgramWeights, device_error};
#[cfg(test)]
use crate::device::asynchronous::block_on;
use crate::resident::ActivationArena;
use crate::resident::batch::take_result;
use crate::resident::greedy::Greedy;
use crate::resident::token_readout::TokenReadout;
use crate::tuning::greedy::select;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_models::mrope::ROTARY_AXES;

pub(super) struct ProgramReadouts {
    pub greedy: Option<DeviceGreedy>,
    pub token_readout: Option<TokenReadout>,
    pub hidden: Arc<Tensor<f32>>,
    pub logits: Arc<Tensor<f32>>,
}
impl ProgramReadouts {
    pub(super) fn new(
        device: &CudaDevice,
        arena: &mut ActivationArena,
        graph: &infer_ir::DataflowGraph,
        weights: &ProgramWeights,
        vocabulary: usize,
        forward_captured: bool,
    ) -> Result<Self> {
        if !forward_captured {
            return Ok(Self {
                hidden: take_result(arena, graph.hidden)?,
                logits: take_result(arena, graph.logits)?,
                greedy: None,
                token_readout: None,
            });
        }
        let hidden = take_result(arena, graph.hidden)?;
        let logits = take_result(arena, graph.logits)?;
        let greedy = DeviceGreedy::new(device, &logits, vocabulary, weights)?;
        let token_readout = if logits.size() == vocabulary {
            TokenReadout::contiguous(device, &logits, 1, weights)?
        } else {
            None
        };
        Ok(Self {
            greedy,
            token_readout,
            hidden,
            logits,
        })
    }
}

pub(super) struct DeviceGreedy {
    graph: CudaGraph<()>,
    selector: Greedy,
}
impl DeviceGreedy {
    pub(super) fn new(
        device: &CudaDevice,
        logits: &Tensor<f32>,
        vocabulary: usize,
        weights: &ProgramWeights,
    ) -> Result<Option<Self>> {
        if weights.fusion.is_none() || logits.size() != vocabulary {
            return Ok(None);
        }
        let tile = select(
            device,
            1,
            vocabulary,
            weights.autotune,
            &weights.resident_tuning,
        );
        let mut selector = Greedy::new(device, 1, vocabulary, tile)?;
        let graph = CudaGraph::scope(&device.stream, |scope| selector.record(scope, logits))
            .map_err(device_error)?;
        tracing::info!(target:"infer::capture", device_greedy=true, rows=1, vocabulary, tile, "draft GPU token readout captured");
        Ok(Some(Self { graph, selector }))
    }
}
impl DeviceProgram {
    pub(crate) const fn has_device_greedy(&self) -> bool {
        self.greedy.is_some()
    }
    /// The first draft consumes a target hidden row; later drafts reuse their own resident output.
    #[cfg(test)]
    pub(crate) fn step_greedy(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
    ) -> Result<u32> {
        block_on(self.step_greedy_async(token, position, kv_position, external))
    }

    pub(crate) async fn step_greedy_async(
        &mut self,
        token: u32,
        position: usize,
        kv_position: usize,
        external: Option<&[f32]>,
    ) -> Result<u32> {
        if self.greedy.is_none() {
            return Err(Error::unsupported("device greedy readout unavailable"));
        }
        let previous = Arc::clone(&self.hidden);
        let external =
            external.map_or_else(|| ExternalInput::Device(&previous), ExternalInput::Host);
        self.prepare_step_async(
            token,
            [position; ROTARY_AXES],
            kv_position,
            Some(external),
            true,
        )
        .await?;
        let greedy = self
            .greedy
            .as_mut()
            .ok_or_else(|| Error::invariant("device greedy readout"))?;
        self.device
            .run_op(self.graph.launch().then(|()| greedy.graph.launch()))
            .await?;
        let tokens = greedy.selector.read_async(&self.device).await?;
        let token = *tokens
            .first()
            .ok_or_else(|| Error::invariant("device greedy token readback"))?;
        self.last_batch = None;
        self.next_position = kv_position + 1;
        Ok(token)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_program_greedy.rs"]
mod tests;
