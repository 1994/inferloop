//! Numerical validation, sampling and projection execute outside the scheduler owner.
use infer_core::{Error, Result};
use infer_ir::{CanonicalRequest, ModelOutput, WorkloadOutput};
use infer_spi::WorkloadProvider;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct OutputShape {
    pub logits: usize,
    pub rows: usize,
    pub width: usize,
}
pub struct OutputJob {
    pub constraint: Option<infer_spi::ConstraintHandle>,
    pub request: Arc<CanonicalRequest>,
    pub output: ModelOutput,
    pub shape: OutputShape,
    pub sample: Option<usize>,
    pub generated: infer_ir::TokenBuffer,
    pub project: Option<Vec<ModelOutput>>,
}
pub struct OutputAcknowledgement {
    pub result: Result<ProcessedOutput>,
    pub elapsed_us: u64,
}
/// Decided tokens a step may carry: the sampled token plus any accepted speculative prefix.
pub const DECIDED_TOKEN_CAPACITY: usize = 16;

/// Tokens decided by one step, held inline.
///
/// Speculation decides a handful of tokens per step, so a heap buffer here would allocate on
/// every decode step and break the engine's steady-state allocation invariant.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DecidedTokens {
    len: usize,
    tokens: [u32; DECIDED_TOKEN_CAPACITY],
}
impl DecidedTokens {
    pub fn push(&mut self, token: u32) {
        if let Some(slot) = self.tokens.get_mut(self.len) {
            *slot = token;
            self.len += 1;
        }
    }
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        self.tokens.get(..self.len).unwrap_or_default()
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}
pub struct ProcessedOutput {
    pub constraint: Option<infer_spi::ConstraintHandle>,
    pub output: Option<ModelOutput>,
    /// Tokens decided by this step: one from the engine sampler, or several when a backend
    /// ran speculative decoding and returned `ModelOutput::tokens`.
    pub tokens: DecidedTokens,
    pub projection: Option<WorkloadOutput>,
}
impl OutputJob {
    pub fn process(
        mut self,
        workloads: &dyn WorkloadProvider,
        scratch: &mut infer_workloads::SamplingWorkspace,
    ) -> Result<ProcessedOutput> {
        if self.constraint.is_some()
            && (self.output.next_token.is_some() || !self.output.tokens.is_empty())
        {
            return Err(Error::invariant("backend bypassed constrained sampling"));
        }
        self.validate_backend_token()?;
        self.shape.validate(&self.output)?;
        if self.output.tokens.len() + usize::from(self.sample.is_some()) > DECIDED_TOKEN_CAPACITY {
            return Err(Error::invariant("backend decided token capacity exceeded"));
        }
        let mut tokens = DecidedTokens::default();
        for token in self.output.tokens.drain(..) {
            tokens.push(token);
        }
        if let Some(position) = self.sample {
            // A speculative backend returns the tokens it already accepted plus the logits for
            // the position after them, so this sampler decides the following token. The accepted
            // prefix joins the history first, exactly as if it had been generated here.
            for token in tokens.as_slice() {
                self.generated.push(*token);
            }
            let token = match self.output.next_token.take() {
                Some(token) => token,
                None => self.sample_token(position, scratch)?,
            };
            tokens.push(token);
            return Ok(ProcessedOutput {
                constraint: self.constraint,
                output: Some(self.output),
                tokens,
                projection: None,
            });
        }
        if !tokens.is_empty() {
            return Err(Error::invariant(
                "backend decided tokens without an output sampler",
            ));
        }
        if let Some(mut outputs) = self.project.take() {
            outputs.push(self.output);
            let projection = workloads.postprocess(&self.request, &outputs)?;
            return Ok(ProcessedOutput {
                constraint: self.constraint,
                output: None,
                tokens: DecidedTokens::default(),
                projection: Some(projection),
            });
        }
        Ok(ProcessedOutput {
            constraint: self.constraint,
            output: Some(self.output),
            tokens: DecidedTokens::default(),
            projection: None,
        })
    }

    fn sample_token(
        &mut self,
        position: usize,
        scratch: &mut infer_workloads::SamplingWorkspace,
    ) -> Result<u32> {
        let mut sample = |mask: Option<infer_workloads::TokenMask<'_>>| {
            infer_workloads::sample_constrained(
                &self.output.logits,
                &self.request.sampling,
                self.request.id.get(),
                position,
                infer_workloads::SamplingHistory {
                    prompt: match &self.request.input {
                        infer_ir::RequestInput::Sequence { tokens, .. } => tokens,
                        infer_ir::RequestInput::Pairs { .. } => &[],
                    },
                    generated: &self.generated,
                    pending: &[],
                },
                mask,
                scratch,
            )
        };
        if let Some(constraint) = &mut self.constraint {
            constraint.select(position, |words| {
                sample(Some(infer_workloads::TokenMask::new(
                    words,
                    self.output.logits.len(),
                )?))
            })
        } else {
            sample(None)
        }
    }

    fn validate_backend_token(&self) -> Result<()> {
        if self.output.next_token.is_none() {
            return Ok(());
        }
        self.request.sampling.validate()?;
        if self.sample.is_none()
            || self.project.is_some()
            || !matches!(self.request.workload, infer_ir::Workload::Generate { .. })
            || !self.request.sampling.uses_plain_greedy()
            || self.shape.rows != 0
        {
            return Err(Error::invariant(
                "backend token outside plain-greedy generation",
            ));
        }
        Ok(())
    }
}
impl OutputShape {
    fn validate(self, output: &ModelOutput) -> Result<()> {
        let logits_valid = output
            .next_token
            .map_or(output.logits.len() == self.logits, |token| {
                output.logits.is_empty()
                    && self.logits > 0
                    && usize::try_from(token).is_ok_and(|token| token < self.logits)
            });
        if !logits_valid
            || output
                .tokens
                .iter()
                .any(|&token| usize::try_from(token).map_or(true, |token| token >= self.logits))
            || output.hidden.len() != self.rows
            || output
                .hidden
                .iter()
                .any(|row| row.len() != self.width || row.iter().any(|value| !value.is_finite()))
            || output.logits.iter().any(|value| !value.is_finite())
        {
            return Err(Error::invariant("backend output shape/numerics mismatch"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod backend_token_tests;
