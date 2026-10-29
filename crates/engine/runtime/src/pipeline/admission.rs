//! Validate and prepare the entire request before reserving execution state.
use crate::preparation::HostCreditQuote;
use crate::preparation::generation_capacity;
use crate::preparation::generation_storage;
use crate::preparation::retained_bytes;
use crate::preparation::retained_tokens;
use crate::preparation::validate_input;
use crate::preparation::validate_plan;
use crate::resource::PendingResource;
use crate::resource::ResourcePhase;
use crate::stages::prefill::retained_readout;
use crate::{Engine, RequestRecord, TenantService};
use infer_core::credits::CreditLease;
use infer_core::{
    Error, ErrorCode, RequestId, RequestStatus, Result, StateId, event::EventKind,
    event::ObjectKind,
};
use infer_ir::{AdmissionDecision, CanonicalRequest, RequestInput, StateKind, WorkloadPlan};
use infer_observe::{DiagnosticCode, trace::TraceContext};
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum ReservationQuote {
    Query,
    Acknowledged(Option<u64>),
}
type PreparedParts = (
    WorkloadPlan,
    infer_ir::TokenBuffer,
    Arc<str>,
    Option<infer_spi::ConstraintHandle>,
    Option<Arc<infer_ir::MediaPlan>>,
);

struct PreparedAdmission {
    constraint: Option<infer_spi::ConstraintHandle>,
    request: Arc<CanonicalRequest>,
    tenant: Arc<str>,
    media: Option<Arc<infer_ir::MediaPlan>>,
    plan: WorkloadPlan,
    admission: AdmissionDecision,
    context: infer_ir::TokenBuffer,
    generated: infer_ir::TokenBuffer,
}
impl PreparedAdmission {
    fn into_record(
        mut self,
        state: StateId,
        (now_us, page): (u64, usize),
        credit: CreditLease,
        byte_credit: CreditLease,
    ) -> RequestRecord {
        // A snapshot covers whole pages, so the boundary rounds down to one.
        let prefix_stop = self
            .request
            .prefix_boundary
            .and_then(|boundary| boundary.checked_div(page))
            .map(|pages| pages * page)
            .filter(|stop| *stop > 0);
        self.generated.attach_credit(credit.clone());
        self.generated.attach_byte_credit(byte_credit.clone());
        self.context.attach_byte_credit(byte_credit.clone());
        for unit in &mut self.plan.units {
            unit.attach_byte_credit(byte_credit.clone());
        }
        self.context.attach_credit(credit.clone());
        RequestRecord {
            constraint: self.constraint,
            request: self.request,
            tenant: self.tenant,
            media: self.media,
            host_lease: Some(credit),
            byte_lease: Some(byte_credit),
            plan: self.plan,
            state: Some(state),
            status: RequestStatus::Runnable,
            progress_epoch: 0,
            unit: 0,
            prefill_done: 0,
            prefill_target: self.context.len(),
            preemptions: 0,
            cached_tokens: 0,
            prefix_attempted: false,
            prefix_stop,
            context: crate::TokenContext::new(self.context),
            generated: self.generated,
            outputs: Vec::new(),
            completed: None,
            pending_finish: None,
            accepted_us: now_us,
            first_token_us: None,
            last_token_us: None,
            max_tpot_us: None,
            last_service_us: now_us,
            admission: self.admission,
        }
    }
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// # Errors
    /// Returns validation, capacity or execution-state admission errors.
    pub fn submit(&mut self, request: CanonicalRequest) -> Result<()> {
        self.submit_with_trace(request, None)
    }
    /// Transport trace metadata is separate from workload feature extensions.
    /// # Errors
    /// Returns request validation, capacity, conflict or quarantine errors.
    pub fn submit_with_trace(
        &mut self,
        request: CanonicalRequest,
        parent: Option<TraceContext>,
    ) -> Result<()> {
        let id = request.id;
        let result = self.submit_inner(request.into(), None);
        self.record_admission_result(id, parent, &result);
        result
    }
    /// Freeze an immutable preparation context before starting CPU workers.
    /// # Errors
    /// Returns unsupported if a custom workload provider cannot create an isolated preparer.
    pub fn request_preparer(&self) -> Result<crate::RequestPreparer> {
        Ok(crate::RequestPreparer::new(
            self.model.clone(),
            self.program.id,
            self.fork_workloads()?,
            self.config.clone(),
            self.limits,
        ))
    }
    /// Admit a CPU-prepared request after checking live resources and request identity.
    /// # Errors
    /// Rejects a foreign preparation context or dynamic admission/reservation failures.
    pub fn submit_prepared_with_trace(
        &mut self,
        prepared: crate::PreparedRequest,
        parent: Option<TraceContext>,
    ) -> Result<()> {
        if *prepared.model != self.model
            || prepared.provider != self.workloads.identity()
            || prepared.plan.program != self.program.id
        {
            return Err(Error::invalid("foreign request preparation context"));
        }
        let id = prepared.request.id;
        let result = self.submit_inner(
            prepared.request,
            Some((
                prepared.plan,
                prepared.generated,
                prepared.tenant,
                prepared.constraint,
                prepared.media,
            )),
        );
        self.record_admission_result(id, parent, &result);
        result
    }
    /// Begin resource sizing without waiting for the backend owner.
    /// # Errors
    /// Rejects foreign preparation contexts or a full/stopped resource command lane.
    pub fn admission_quote(
        &mut self,
        prepared: &crate::PreparedRequest,
    ) -> Result<infer_spi::ResourceTicket> {
        let command = self.quote_command(prepared)?;
        self.backend.begin_resource(command)
    }
    /// Begin resource sizing, or report that the backend's command lane is momentarily full.
    ///
    /// `Ok(None)` is backpressure from commands already queued for the device owner: the
    /// request itself was accepted and the same quote can be retried once the owner drains.
    /// Errors about the request (identity, buffer limits, deadline) are returned as before.
    /// # Errors
    /// Rejects foreign preparation contexts, invalid or over-limit requests and a stopped lane.
    pub fn try_admission_quote(
        &mut self,
        prepared: &crate::PreparedRequest,
    ) -> Result<Option<infer_spi::ResourceTicket>> {
        let command = self.quote_command(prepared)?;
        match self.backend.begin_resource(command) {
            Ok(ticket) => Ok(Some(ticket)),
            Err(error) if error.code == ErrorCode::Capacity => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Check whether a prepared media request can enter the runtime's host budget now.
    ///
    /// This does not reserve credit. The engine owner must still perform admission without
    /// allowing another reservation between this check and the actual admission. Non-media
    /// requests retain their existing admission behavior.
    /// # Errors
    /// Rejects foreign or invalid requests and requests that can never fit the host budget.
    pub fn media_host_credit_ready(&self, prepared: &mut crate::PreparedRequest) -> Result<bool> {
        if prepared.media.is_none() {
            return Ok(true);
        }
        if self.fault.is_some() {
            return Err(Error::new(
                ErrorCode::Backend,
                "engine is quarantined; replace it after in-flight work drains",
            ));
        }
        if *prepared.model != self.model
            || prepared.provider != self.workloads.identity()
            || prepared.plan.program != self.program.id
        {
            return Err(Error::invalid("foreign request preparation context"));
        }
        self.validate_request_admission(&prepared.request)?;
        let quote = if let Some(quote) = prepared.host_quote {
            quote
        } else {
            let quote = self.host_requirements(
                &prepared.request,
                &prepared.plan,
                prepared.constraint.as_ref(),
            )?;
            prepared.host_quote = Some(quote);
            quote
        };
        let tokens = self.host.credits.can_reserve(quote.tokens)?;
        let bytes = self.host.bytes.can_reserve(quote.bytes)?;
        Ok(tokens && bytes)
    }
    fn quote_command(
        &self,
        prepared: &crate::PreparedRequest,
    ) -> Result<infer_spi::ResourceCommand> {
        if *prepared.model != self.model
            || prepared.provider != self.workloads.identity()
            || prepared.plan.program != self.program.id
        {
            return Err(Error::invalid("foreign request preparation context"));
        }
        self.validate_request_admission(&prepared.request)?;
        let capacity = prepared.plan.reserved_tokens;
        let readout = retained_readout(&prepared.request.workload);
        Ok(prepared.media.clone().map_or_else(
            || infer_spi::ResourceCommand::ReservationBytes { capacity, readout },
            |media| infer_spi::ResourceCommand::ReservationBytesMedia {
                capacity,
                readout,
                media,
            },
        ))
    }
    /// Admit with the already acknowledged sizing quote, while rechecking live quotas/resources.
    /// # Errors
    /// Rejects a stale context, capacity, duplicate identity or failed resource reservation.
    pub fn submit_quoted_with_trace(
        &mut self,
        prepared: crate::PreparedRequest,
        bytes: Option<u64>,
        parent: Option<TraceContext>,
    ) -> Result<()> {
        if *prepared.model != self.model
            || prepared.provider != self.workloads.identity()
            || prepared.plan.program != self.program.id
        {
            return Err(Error::invalid("foreign request preparation context"));
        }
        let id = prepared.id();
        let result = self.submit_inner_quoted(
            prepared.request,
            Some((
                prepared.plan,
                prepared.generated,
                prepared.tenant,
                prepared.constraint,
                prepared.media,
            )),
            ReservationQuote::Acknowledged(bytes),
        );
        self.record_admission_result(id, parent, &result);
        result
    }
    fn record_admission_result(
        &mut self,
        id: RequestId,
        parent: Option<TraceContext>,
        result: &Result<()>,
    ) {
        match result {
            Ok(()) => {
                let workload = self.host.requests.get(id).map_or_else(
                    || "unknown".to_owned(),
                    |record| format!("{:?}", record.request.workload),
                );
                tracing::info!(
                    target: "infer::request",
                    request = id.get(),
                    workload = %workload,
                    "request accepted"
                );
                if let Some(parent) = parent.filter(TraceContext::valid)
                    && self
                        .observations
                        .collector
                        .as_mut()
                        .is_none_or(|collector| collector.parent(id.get(), parent.clone()))
                {
                    self.observations.parents.insert(id.get(), parent);
                    self.observations.parents_dirty = true;
                }
            }
            Err(error) => self.record_admission_rejection(id, error),
        }
    }
    /// Retain cold-stage rejection evidence even when no runtime record was admitted.
    pub fn record_admission_rejection(&mut self, id: RequestId, error: &Error) {
        tracing::warn!(
            target: "infer::request",
            request = id.get(),
            code = ?error.code,
            "request rejected: {}",
            error.message
        );
        self.event(
            EventKind::Rejected,
            ObjectKind::Request,
            id.get(),
            0,
            infer_observe::error_code(error.code),
            0,
        );
        self.retain_diagnostic(DiagnosticCode::AdmissionRejected, error, Some(id), None);
    }
    fn submit_inner(
        &mut self,
        request: Arc<CanonicalRequest>,
        plan: Option<PreparedParts>,
    ) -> Result<()> {
        self.submit_inner_quoted(request, plan, ReservationQuote::Query)
    }
    fn submit_inner_quoted(
        &mut self,
        request: Arc<CanonicalRequest>,
        plan: Option<PreparedParts>,
        quote: ReservationQuote,
    ) -> Result<()> {
        if self.fault.is_some() {
            return Err(Error::new(
                ErrorCode::Backend,
                "engine is quarantined; replace it after in-flight work drains",
            ));
        }
        let prepared = self.prepare_admission(request, plan, quote)?;
        let quote = self.host_requirements(
            &prepared.request,
            &prepared.plan,
            prepared.constraint.as_ref(),
        )?;
        let credit = self.host.credits.reserve(quote.tokens)?;
        let byte_credit = self.host.bytes.reserve(quote.bytes)?;
        let state = self.reserve_execution_state(&prepared)?;
        let id = prepared.request.id;
        self.retain_request_identity(id);
        self.register_tenant(&prepared.request, prepared.tenant.clone(), &prepared.plan)?;
        let record = prepared.into_record(
            state,
            (self.now_us, self.config.block_size),
            credit,
            byte_credit,
        );
        self.record_action(if record.constraint.is_some() {
            crate::ReplayAction::ConstrainedSubmit(id)
        } else {
            crate::ReplayAction::Submit(record.request.clone())
        });
        self.host.requests.insert(id, record)?;
        self.discard_prepared();
        self.enqueue_request(id)?;
        if let Some(pending) = self.resources_pending.remove(id) {
            self.park_pending(id, pending)?;
        }
        self.event(
            EventKind::Accepted,
            ObjectKind::Request,
            id.get(),
            self.program.id.get(),
            0,
            0,
        );
        self.event(
            EventKind::StateReserved,
            ObjectKind::State,
            state.get(),
            id.get(),
            self.state.get(state)?.pages.len() as u64,
            0,
        );
        self.progress(id)
    }
    fn prepare_admission(
        &mut self,
        request: Arc<CanonicalRequest>,
        plan: Option<PreparedParts>,
        quote: ReservationQuote,
    ) -> Result<PreparedAdmission> {
        self.apply_cost_feedback()?;
        self.validate_request_admission(&request)?;
        let (plan, generated, tenant, constraint, media) = if let Some(plan) = plan {
            plan
        } else {
            request.validate()?;
            self.validate_input_budget(&request.input)?;
            if !self.workloads.supports(&request.workload) {
                return Err(Error::unsupported("workload provider not installed"));
            }
            let plan = self
                .workloads
                .plan(&request, &self.model, self.program.id)?;
            self.validate_workload_plan(&request, &plan)?;
            let media = match &request.input {
                RequestInput::Sequence { tokens, media } => {
                    infer_ir::MediaPlan::from_sequence(tokens, media)?.map(Arc::new)
                }
                RequestInput::Pairs { .. } => None,
            };
            (
                plan,
                generation_storage(&request)?,
                Arc::from(request.qos.tenant.as_str()),
                None,
                media,
            )
        };
        let admission = match quote {
            ReservationQuote::Acknowledged(bytes) => {
                self.check_admission_with_bytes(&request, &plan, bytes)?
            }
            ReservationQuote::Query => {
                let readout = retained_readout(&request.workload);
                if media.is_some() {
                    self.backend
                        .prepare_state_reservation_for_media(plan.reserved_tokens, readout)?;
                } else {
                    self.backend
                        .prepare_state_reservation(plan.reserved_tokens, readout)?;
                }
                self.check_admission(&request, &plan)?
            }
        };
        let context = plan
            .units
            .first()
            .cloned()
            .ok_or_else(|| Error::invariant("validated plan has no initial unit"))?;
        Ok(PreparedAdmission {
            constraint,
            request,
            tenant,
            media,
            plan,
            admission,
            context,
            generated,
        })
    }
    fn validate_request_admission(&self, request: &CanonicalRequest) -> Result<()> {
        if self.host.requests.contains_key(request.id)
            || self.seen_requests.contains(&request.id)
            || request.id.get() <= self.retired_request_floor
        {
            return Err(Error::new(ErrorCode::Conflict, "duplicate request ID"));
        }
        if self.host.requests.len() >= self.config.max_requests {
            return Err(Error::new(
                ErrorCode::Capacity,
                "request/result buffer is full; drain completed requests",
            ));
        }
        if request.qos.deadline_us.is_some_and(|d| d <= self.now_us) {
            return Err(Error::invalid("request deadline has expired"));
        }
        if self
            .tenants
            .get(request.qos.tenant.as_str())
            .is_some_and(|t| t.weight != request.qos.weight)
        {
            return Err(Error::invalid(
                "tenant weight must be consistent across requests",
            ));
        }
        Ok(())
    }
    fn validate_input_budget(&self, input: &RequestInput) -> Result<()> {
        validate_input(input, &self.config, &self.limits)
    }
    fn validate_workload_plan(
        &self,
        request: &CanonicalRequest,
        plan: &WorkloadPlan,
    ) -> Result<()> {
        validate_plan(
            request,
            plan,
            &self.model,
            self.program.id,
            &self.config,
            &self.limits,
        )
    }
    fn reserve_execution_state(&mut self, prepared: &PreparedAdmission) -> Result<StateId> {
        let state = self.state.reserve_incremental(
            prepared.request.id,
            StateKind::AttentionKv,
            prepared.plan.reserved_tokens,
        )?;
        let command = if prepared.media.is_some() {
            infer_spi::ResourceCommand::ReserveMedia {
                state,
                capacity: prepared.plan.reserved_tokens,
                readout: retained_readout(&prepared.request.workload),
                media: prepared.media.clone().ok_or_else(|| {
                    Error::invariant("media reserve without a prepared media plan")
                })?,
            }
        } else {
            infer_spi::ResourceCommand::Reserve {
                state,
                capacity: prepared.plan.reserved_tokens,
                readout: retained_readout(&prepared.request.workload),
            }
        };
        let attempt =
            self.backend
                .begin_resource(command)
                .and_then(|mut ticket| match ticket.poll() {
                    Ok(Some(infer_spi::ResourceReply::Reserved)) => Ok(None),
                    Ok(Some(_)) => Err(Error::invariant(
                        "reservation acknowledgement type mismatch",
                    )),
                    Err(error) => Err(error),
                    Ok(None) => Ok(Some(PendingResource {
                        ticket: Some(ticket),
                        phase: ResourcePhase::Reserve,
                        retry_epoch: Some(self.resource_epoch),
                        started: self.now_us,
                    })),
                });
        let pending = match attempt {
            // A busy resident budget, whether the device owner reports it before the engine's
            // first poll or a full control lane refuses the command outright, is backpressure,
            // not a failed submission: keep the host state and retry only after ownership
            // progresses, exactly as a later retry of the same intent does.
            Err(error)
                if error.code == ErrorCode::Capacity
                    && self.backend.tracks_reservation_intent() =>
            {
                Ok(Some(PendingResource {
                    ticket: None,
                    phase: ResourcePhase::Reserve,
                    retry_epoch: Some(self.resource_epoch),
                    started: self.now_us,
                }))
            }
            attempt => attempt,
        };
        let pending = match pending {
            Ok(pending) => pending,
            Err(error) => {
                self.state.release(state)?;
                return Err(error);
            }
        };
        if let Some(pending) = pending {
            self.resources_pending
                .insert(prepared.request.id, pending)?;
        }
        Ok(state)
    }
    fn retain_request_identity(&mut self, id: RequestId) {
        self.seen_requests.insert(id);
        if self.seen_requests.len() > self.config.history_capacity
            && let Some(retired) = self.seen_requests.pop_first()
        {
            self.retired_request_floor = retired.get();
        }
    }
    fn register_tenant(
        &mut self,
        request: &CanonicalRequest,
        name: Arc<str>,
        plan: &WorkloadPlan,
    ) -> Result<()> {
        let floor = self.host.queues.virtual_time();
        if let Some(tenant) = self.tenants.get_mut(request.qos.tenant.as_str()) {
            tenant.virtual_finish = tenant.virtual_finish.max(floor);
            tenant.owners += 1;
            tenant.active += 1;
            tenant.tokens += plan.reserved_tokens;
            tenant.pages += plan.reserved_tokens.div_ceil(self.config.block_size);
        } else {
            self.tenants.insert(
                name,
                TenantService {
                    owners: 1,
                    active: 1,
                    tokens: plan.reserved_tokens,
                    pages: plan.reserved_tokens.div_ceil(self.config.block_size),
                    weight: request.qos.weight,
                    virtual_finish: floor,
                },
            )?;
        }
        Ok(())
    }
}

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    fn host_requirements(
        &self,
        request: &CanonicalRequest,
        plan: &WorkloadPlan,
        constraint: Option<&infer_spi::ConstraintHandle>,
    ) -> Result<HostCreditQuote> {
        let tokens = retained_tokens(plan, generation_capacity(request))?;
        let mut bytes = retained_bytes(request, plan, &self.model, self.config.block_size)?
            .checked_add(constraint.map_or(0, infer_spi::ConstraintHandle::retained_bytes))
            .ok_or_else(|| Error::invalid("constraint reservation overflow"))?;
        if let Some(recipe) = self.backend.state_recipe() {
            recipe.visit(
                plan.reserved_tokens,
                retained_readout(&request.workload),
                |region| {
                    // Numerical readbacks and their row headers are already charged by retained_bytes.
                    if region.region.memory == infer_ir::StateMemory::HostMirror
                        && matches!(
                            region.region.kind,
                            infer_ir::StateRegionKind::Tokens
                                | infer_ir::StateRegionKind::PageTable
                                | infer_ir::StateRegionKind::BlockLeases
                        )
                    {
                        bytes = usize::try_from(region.bytes)
                            .ok()
                            .and_then(|extra| bytes.checked_add(extra))
                            .ok_or_else(|| {
                                Error::invalid("physical state host mirrors exceed byte ABI")
                            })?;
                    }
                    Ok(())
                },
            )?;
        }
        Ok(HostCreditQuote { tokens, bytes })
    }
}
