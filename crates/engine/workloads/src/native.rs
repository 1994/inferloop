//! Native workload responsibilities.
use super::{decisions, embedding, ranking};
use infer_core::{Error, ProgramId, Result};
use infer_ir::{
    CanonicalRequest, DecisionQuestion, ModelIr, ModelOutput, RequestInput, Workload,
    WorkloadOutput, WorkloadPlan,
};
use infer_spi::WorkloadProvider;

pub struct NativeWorkloads;
impl WorkloadProvider for NativeWorkloads {
    fn fork(&self) -> Option<Box<dyn WorkloadProvider + Send + Sync>> {
        Some(Box::new(Self))
    }
    fn identity(&self) -> &'static str {
        "native-workloads-v1"
    }
    fn supports(&self, workload: &Workload) -> bool {
        matches!(
            workload,
            Workload::Generate { .. }
                | Workload::Embed { .. }
                | Workload::Rerank { .. }
                | Workload::Decision(_)
        )
    }
    fn plan(
        &self,
        request: &CanonicalRequest,
        model: &ModelIr,
        program: ProgramId,
    ) -> Result<WorkloadPlan> {
        request.validate()?;
        model.validate()?;
        if request.model != model.id {
            return Err(Error::invalid("request/model mismatch"));
        }
        if !self.supports(&request.workload)
            || !request
                .workload
                .head()
                .is_some_and(|h| model.heads.contains(&h))
        {
            return Err(Error::unsupported("workload/head is not installed"));
        }
        let units = match &request.input {
            RequestInput::Sequence { tokens, .. } => vec![tokens.clone()],
            RequestInput::Pairs { query, documents } => documents
                .iter()
                .map(|doc| query.iter().chain(doc.iter()).copied().collect())
                .collect(),
        };
        for unit in &units {
            if unit.is_empty()
                || unit.len() > model.max_sequence
                || unit.iter().any(|token| *token as usize >= model.vocab_size)
            {
                return Err(Error::invalid(
                    "token input exceeds vocabulary/context limits",
                ));
            }
        }
        if request
            .sampling
            .eos_token
            .is_some_and(|t| t as usize >= model.vocab_size)
            || request
                .sampling
                .eos_tokens
                .iter()
                .any(|t| *t as usize >= model.vocab_size)
            || request.sampling.top_k.is_some_and(|k| k > model.vocab_size)
        {
            return Err(Error::invalid("sampling exceeds model vocabulary"));
        }
        if let Workload::Embed {
            dimensions: Some(d),
            ..
        } = request.workload
            && d > model.hidden_size
        {
            return Err(Error::invalid("embedding dimensions exceed hidden width"));
        }
        if let Workload::Decision(schema) = &request.workload {
            for q in &schema.questions {
                if !decision_tokens_fit(q, model.vocab_size) {
                    return Err(Error::invalid("decision option exceeds vocabulary"));
                }
            }
        }
        let extra = match request.workload {
            Workload::Generate { max_new_tokens } => max_new_tokens,
            _ => 0,
        };
        let reserved_tokens = units
            .iter()
            .map(|unit| unit.len())
            .max()
            .ok_or_else(|| Error::invariant("nonempty units"))?
            .checked_add(extra)
            .ok_or_else(|| Error::invalid("request token count overflow"))?;
        if reserved_tokens > model.max_sequence {
            return Err(Error::invalid("input + output exceeds context window"));
        }
        Ok(WorkloadPlan {
            request: request.id,
            model: model.id,
            program,
            units,
            reserved_tokens,
        })
    }
    fn postprocess(
        &self,
        request: &CanonicalRequest,
        outputs: &[ModelOutput],
    ) -> Result<WorkloadOutput> {
        request.validate()?;
        let first = outputs
            .first()
            .ok_or_else(|| Error::invariant("empty workload outputs"))?;
        match &request.workload {
            Workload::Embed {
                pooling,
                dimensions,
                normalize,
            } => embedding(first, outputs.len(), *pooling, *dimensions, *normalize),
            Workload::Rerank { top_k } => ranking(outputs, *top_k),
            Workload::Decision(schema) => decisions(first, outputs.len(), schema),
            _ => Err(Error::unsupported(
                "postprocessor unavailable for this workload",
            )),
        }
    }
}

pub fn decision_tokens_fit(question: &DecisionQuestion, limit: usize) -> bool {
    let fits = |token: u32| (token as usize) < limit;
    match question {
        DecisionQuestion::Binary {
            negative_token,
            positive_token,
        } => fits(*negative_token) && fits(*positive_token),
        DecisionQuestion::Categorical { options } | DecisionQuestion::Ordinal { options, .. } => {
            options.iter().copied().all(fits)
        }
        DecisionQuestion::Continuous { token, .. } => fits(*token),
    }
}
