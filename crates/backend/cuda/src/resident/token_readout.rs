//! Exact plain-greedy output for contiguous or separately allocated resident logits.
use super::{ProgramWeights, greedy::Greedy};
use crate::device::{CudaDevice, ReadbackSource, Readbacks, device_error};
use crate::tuning::greedy::select;
use cutile::prelude::*;
use infer_core::{Error, Result};
use std::slice::from_ref;

pub(super) struct TokenReadout {
    // The graph is destroyed before every buffer it references.
    graph: CudaGraph<()>,
    tokens: Arc<Tensor<i32>>,
    readbacks: Readbacks<i32>,
    _selectors: Vec<Greedy>,
    vocabulary: usize,
}
impl TokenReadout {
    pub(super) fn contiguous(
        device: &CudaDevice,
        logits: &Arc<Tensor<f32>>,
        rows: usize,
        weights: &ProgramWeights,
    ) -> Result<Option<Self>> {
        if rows == 0 {
            return Err(Error::invalid("token readout row count"));
        }
        Self::new(
            device,
            from_ref(logits),
            rows,
            logits.size() / rows,
            weights,
        )
    }

    pub(super) fn new(
        device: &CudaDevice,
        logits: &[Arc<Tensor<f32>>],
        rows: usize,
        vocabulary: usize,
        weights: &ProgramWeights,
    ) -> Result<Option<Self>> {
        if rows == 0 || vocabulary == 0 || logits.is_empty() {
            return Err(Error::invalid("token readout geometry"));
        }
        let per_input = if logits.len() == 1 { rows } else { 1 };
        if logits.len() * per_input != rows
            || logits
                .iter()
                .any(|input| input.size() != per_input * vocabulary)
        {
            return Err(Error::invalid("token readout logits layout"));
        }
        let tile = select(
            device,
            per_input,
            vocabulary,
            weights.autotune,
            &weights.resident_tuning,
        );
        let mut selectors = logits
            .iter()
            .map(|_| Greedy::new(device, per_input, vocabulary, tile))
            .collect::<Result<Vec<_>>>()?;
        let mut tokens = api::zeros::<i32>(&[rows, 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let graph = CudaGraph::scope(&device.stream, |scope| {
            for (index, (selector, input)) in selectors.iter_mut().zip(logits).enumerate() {
                selector.record_into(scope, input, &mut tokens, index * per_input)?;
            }
            Ok(())
        })
        .map_err(device_error)?;
        tracing::info!(target: "infer::capture", device_token_readout=true,
            rows, vocabulary, tile, "GPU token readout captured");
        Ok(Some(Self {
            graph,
            tokens: Arc::new(tokens),
            readbacks: Readbacks::allocated(device, rows * 2)?,
            _selectors: selectors,
            vocabulary,
        }))
    }

    pub(super) async fn run_async(
        &mut self,
        device: &CudaDevice,
        graph: &CudaGraph<()>,
        readbacks: &mut Readbacks,
        sources: &[Option<ReadbackSource>],
        active: &[usize],
    ) -> Result<(super::batch::BatchOutput, Vec<u32>)> {
        if active.iter().any(|&row| row >= self.tokens.size() / 2) {
            return Err(Error::invalid("token readout active row"));
        }
        let (rows, pairs) = readbacks
            .run_with_tokens_async(
                device,
                graph,
                sources,
                (&self.graph, &mut self.readbacks, Arc::clone(&self.tokens)),
            )
            .await?;
        self.decode(rows, &pairs, active, sources.len() / 2)
    }

    fn decode(
        &self,
        rows: Vec<Vec<f32>>,
        pairs: &[Vec<i32>],
        active: &[usize],
        count: usize,
    ) -> Result<(super::batch::BatchOutput, Vec<u32>)> {
        let pairs = pairs
            .first()
            .ok_or_else(|| Error::invariant("token readout pairs"))?;
        let tokens = active
            .iter()
            .map(|&row| Greedy::decode(&pairs[row * 2..row * 2 + 2], self.vocabulary))
            .collect::<Result<Vec<_>>>()?;
        let mut rows = rows.into_iter();
        let output = (0..count)
            .map(|_| {
                Ok((
                    rows.next()
                        .ok_or_else(|| Error::invariant("token hidden readback"))?,
                    rows.next()
                        .ok_or_else(|| Error::invariant("token logits readback"))?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok((output, tokens))
    }
}
