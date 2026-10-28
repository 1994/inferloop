//! Immutable request preparation runs outside the scheduler owner.
mod memory;
use crate::RuntimeConfig;
use infer_core::{Error, ProgramId, Result};
use infer_ir::{CanonicalRequest, ModelIr, RequestInput, WorkloadPlan};
use infer_spi::WorkloadProvider;
use std::sync::Arc;

pub use memory::request_bytes;

#[cfg(test)]
#[path = "../../tests/unit/media_plan.rs"]
mod media_plan_tests;

/// Fixed retained overhead beyond token, page and projection storage.
const RETAINED_OVERHEAD_BYTES: usize = 4096;

pub struct PreparedRequest {
    pub(crate) constraint: Option<infer_spi::ConstraintHandle>,
    pub(crate) request: Arc<CanonicalRequest>,
    pub(crate) plan: WorkloadPlan,
    pub(crate) tenant: Arc<str>,
    pub(crate) generated: infer_ir::TokenBuffer,
    pub(crate) media: Option<Arc<infer_ir::MediaPlan>>,
    pub(crate) model: Arc<ModelIr>,
    pub(crate) provider: String,
}
impl PreparedRequest {
    /// Attach a matcher built on the bounded preparation pool.
    /// # Errors
    /// Rejects mismatched vocabularies or non-generation workloads.
    pub fn with_constraint(
        mut self,
        constraint: Option<infer_spi::ConstraintHandle>,
    ) -> Result<Self> {
        if constraint
            .as_ref()
            .is_some_and(|c| c.vocabulary() != self.model.vocab_size)
            || (constraint.is_some()
                && !matches!(self.request.workload, infer_ir::Workload::Generate { .. }))
        {
            return Err(Error::invalid(
                "constraint requires generation with matching vocabulary",
            ));
        }
        self.constraint = constraint;
        Ok(self)
    }

    #[must_use]
    pub fn id(&self) -> infer_core::RequestId {
        self.request.id
    }
}
#[derive(Clone)]
pub struct RequestPreparer {
    model: Arc<ModelIr>,
    program: ProgramId,
    workloads: Arc<dyn WorkloadProvider + Send + Sync>,
    config: RuntimeConfig,
    limits: crate::ResolvedLengthLimits,
}
impl RequestPreparer {
    #[must_use]
    pub fn vocabulary(&self) -> usize {
        self.model.vocab_size
    }
    pub(crate) fn new(
        model: ModelIr,
        program: ProgramId,
        workloads: Box<dyn WorkloadProvider + Send + Sync>,
        config: RuntimeConfig,
        limits: crate::ResolvedLengthLimits,
    ) -> Self {
        Self {
            model: Arc::new(model),
            program,
            workloads: workloads.into(),
            config,
            limits,
        }
    }
    /// # Errors
    /// Rejects invalid schema, input/plan budgets, unsupported heads or invalid tokens.
    pub fn prepare(&self, request: CanonicalRequest) -> Result<PreparedRequest> {
        request.validate()?;
        validate_input(&request.input, &self.config, &self.limits)?;
        let plan = self.workloads.plan(&request, &self.model, self.program)?;
        validate_plan(
            &request,
            &plan,
            &self.model,
            self.program,
            &self.config,
            &self.limits,
        )?;
        let generated = generation_storage(&request)?;
        let media = match &request.input {
            RequestInput::Sequence { tokens, media } => {
                infer_ir::MediaPlan::from_sequence(tokens, media)?.map(Arc::new)
            }
            RequestInput::Pairs { .. } => None,
        };
        let tenant = Arc::from(request.qos.tenant.as_str());
        Ok(PreparedRequest {
            constraint: None,
            generated,
            media,
            tenant,
            request: request.into(),
            plan,
            model: self.model.clone(),
            provider: self.workloads.identity().into(),
        })
    }
}
pub fn validate_input(
    input: &RequestInput,
    config: &RuntimeConfig,
    limits: &crate::ResolvedLengthLimits,
) -> Result<()> {
    let (units, tokens) = match input {
        RequestInput::Sequence { tokens, .. } => (1, tokens.len()),
        RequestInput::Pairs { query, documents } => (
            documents.len(),
            documents
                .iter()
                .try_fold(0usize, |sum, doc| {
                    sum.checked_add(query.len())
                        .and_then(|n| n.checked_add(doc.len()))
                })
                .ok_or_else(|| Error::invalid("pair token count overflow"))?,
        ),
    };
    if units > config.max_request_units || tokens > limits.input_cap {
        return Err(Error::new(
            infer_core::ErrorCode::Capacity,
            "request input exceeds admission budget",
        ));
    }
    Ok(())
}
pub fn validate_plan(
    request: &CanonicalRequest,
    plan: &WorkloadPlan,
    model: &ModelIr,
    program: ProgramId,
    config: &RuntimeConfig,
    limits: &crate::ResolvedLengthLimits,
) -> Result<()> {
    let tokens = plan
        .units
        .iter()
        .try_fold(0usize, |sum, unit| sum.checked_add(unit.len()));
    let requested_output = match request.workload {
        infer_ir::Workload::Generate { max_new_tokens } => max_new_tokens,
        _ => 0,
    };
    // Structural integrity stays a single provider-fault error; the length budgets below get
    // their own messages so a client can tell an explicit over-budget request from a bad plan.
    if plan.request != request.id
        || plan.model != model.id
        || plan.program != program
        || plan.units.is_empty()
        || plan.units.len() > config.max_request_units
        || plan.reserved_tokens == 0
        || plan.units.iter().any(|unit| {
            unit.is_empty()
                || unit.len() > plan.reserved_tokens
                || unit.iter().any(|token| *token as usize >= model.vocab_size)
        })
    {
        return Err(Error::invalid(
            "workload provider returned an invalid/budget-exceeding plan",
        ));
    }
    let tokens = tokens.ok_or_else(|| Error::invalid("planned token count overflow"))?;
    if tokens > limits.input_cap {
        return Err(Error::invalid(format!(
            "encoded prompt of {tokens} tokens exceeds the input cap {}",
            limits.input_cap
        )));
    }
    if requested_output > limits.output_cap {
        return Err(Error::invalid(format!(
            "requested output of {requested_output} tokens exceeds the output cap {}",
            limits.output_cap
        )));
    }
    if plan.reserved_tokens > limits.total {
        return Err(Error::invalid(format!(
            "prompt {tokens} + output {requested_output} exceeds the total context {}",
            limits.total
        )));
    }
    Ok(())
}

pub fn generation_storage(request: &CanonicalRequest) -> Result<infer_ir::TokenBuffer> {
    let capacity = if let infer_ir::Workload::Generate { max_new_tokens } = request.workload {
        max_new_tokens
    } else {
        0
    };
    infer_ir::TokenBuffer::with_capacity(capacity)
}

pub fn retained_tokens(plan: &WorkloadPlan, generated: usize) -> Result<usize> {
    plan.units
        .iter()
        .try_fold(generated, |sum, unit| {
            sum.checked_add(unit.len().saturating_mul(2))
        })
        .ok_or_else(|| Error::invalid("host token reservation overflow"))
}

pub const fn generation_capacity(request: &CanonicalRequest) -> usize {
    if let infer_ir::Workload::Generate { max_new_tokens } = request.workload {
        max_new_tokens
    } else {
        0
    }
}

/// Conservative peak reservation for request tokens, full readouts and retained projections.
/// # Errors
/// Rejects overflow before any backend reservation or native output allocation.
pub fn retained_bytes(
    request: &CanonicalRequest,
    plan: &WorkloadPlan,
    model: &ModelIr,
    block_size: usize,
) -> Result<usize> {
    let tokens = retained_tokens(plan, generation_capacity(request))?;
    let mut total = tokens
        .checked_mul(size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(RETAINED_OVERHEAD_BYTES))
        .and_then(|bytes| bytes.checked_add(request_bytes(request)))
        .and_then(|bytes| bytes.checked_add(memory::projection_bytes(request)))
        .and_then(|bytes| {
            plan.reserved_tokens
                .div_ceil(block_size)
                .checked_mul(size_of::<infer_core::StatePageId>())
                .and_then(|pages| bytes.checked_add(pages))
        })
        .and_then(|bytes| {
            plan.units
                .capacity()
                .checked_mul(size_of::<infer_ir::TokenBuffer>())
                .and_then(|units| bytes.checked_add(units))
        })
        .ok_or_else(|| Error::invalid("host request bytes overflow"))?;
    // Forward readouts can retain all pair units. Include vector headers and both projection/input lifetimes.
    let hidden_rows = request.workload.final_readout() == infer_ir::OutputReadout::Full;
    for unit in &plan.units {
        let rows = if hidden_rows { unit.len() } else { 0 };
        let row_bytes = model
            .hidden_size
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(size_of::<Vec<f32>>()))
            .ok_or_else(|| Error::invalid("host readout width overflow"))?;
        total = rows
            .checked_mul(row_bytes)
            .and_then(|bytes| {
                model
                    .vocab_size
                    .checked_mul(size_of::<f32>())
                    .and_then(|logits| bytes.checked_add(logits))
            })
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| Error::invalid("host readout bytes overflow"))?;
    }
    Ok(total)
}
