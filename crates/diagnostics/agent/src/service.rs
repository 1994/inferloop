use crate::constants::JSONRPC_INVALID_PARAMS;
use crate::constants::JSONRPC_INVALID_REQUEST;
use crate::constants::JSONRPC_METHOD_NOT_FOUND;
use crate::constants::MAX_BATCH_REQUESTS;
use crate::constants::MAX_RECORDED_METHOD_CHARS;
use crate::{
    AgentBackend, AgentContext, CallRecord, commands, protocol::RpcError, protocol::RpcRequest,
    protocol::error_response, protocol::response, registry::CommandDescriptor,
    registry::CommandOutput, registry::CommandRegistry,
};
use infer_core::Result;
use infer_runtime::Engine;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::time::Instant;

pub struct AgentDispatcher<B: AgentBackend> {
    context: AgentContext<B>,
    registry: CommandRegistry<AgentContext<B>>,
}

impl<B: AgentBackend> AgentDispatcher<B> {
    /// # Errors
    /// Returns registration errors if built-in command names collide.
    pub fn new(engine: Engine<B>, backend_catalog: Value) -> Result<Self> {
        let registry = commands::registry()?;
        let mut context = AgentContext::new(engine, backend_catalog);
        context.descriptors = registry.descriptors();
        Ok(Self { context, registry })
    }

    #[must_use]
    pub const fn context(&self) -> &AgentContext<B> {
        &self.context
    }

    /// Extend the control plane without changing the dispatcher or runtime.
    /// # Errors
    /// Returns descriptor validation and transactional registration errors.
    pub fn register<P: DeserializeOwned + 'static, R: CommandOutput + 'static>(
        &mut self,
        descriptor: CommandDescriptor,
        execute: fn(&mut AgentContext<B>, P) -> R,
    ) -> Result<()> {
        self.registry.register(descriptor, execute)?;
        self.context.descriptors = self.registry.descriptors();
        Ok(())
    }

    /// Execute parsed JSON; bounded batches are sequential to preserve control order.
    #[must_use]
    pub fn handle(&mut self, value: Value) -> Option<Value> {
        if let Value::Array(batch) = value {
            if batch.is_empty() || batch.len() > MAX_BATCH_REQUESTS {
                return Some(error_response(
                    Value::Null,
                    &RpcError::new(JSONRPC_INVALID_REQUEST, "batch requires 1 to 64 requests"),
                ));
            }
            let responses: Vec<_> = batch
                .into_iter()
                .filter_map(|value| self.single(value))
                .collect();
            return (!responses.is_empty()).then_some(Value::Array(responses));
        }
        self.single(value)
    }

    fn single(&mut self, value: Value) -> Option<Value> {
        let request = match RpcRequest::parse(value) {
            Ok(request) => request,
            Err(error) => return Some(error_response(Value::Null, &error)),
        };
        let start = Instant::now();
        let outcome = if !request.params.is_object() && !request.params.is_array() {
            Err(RpcError::new(
                JSONRPC_INVALID_PARAMS,
                "params must be an object or array",
            ))
        } else {
            self.registry
                .dispatch(&mut self.context, &request.method, request.params)
                .map_or_else(
                    || Err(RpcError::new(JSONRPC_METHOD_NOT_FOUND, "method not found")),
                    |r| r.map_err(RpcError::from),
                )
        };
        self.context.engine.collect_observations();
        self.context.call_sequence = self.context.call_sequence.saturating_add(1);
        let error = outcome.as_ref().err().and_then(|error| {
            error
                .data
                .as_ref()
                .and_then(|data| data.get("code"))
                .and_then(|code| serde_json::from_value(code.clone()).ok())
        });
        self.context.record(CallRecord {
            sequence: self.context.call_sequence,
            method: request
                .method
                .chars()
                .take(MAX_RECORDED_METHOD_CHARS)
                .collect(),
            error,
            rpc_error: outcome.as_ref().err().map(|error| error.code),
            elapsed_us: u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX),
            notification: request.id.is_none(),
        });
        request.id.map(|id| response(id, outcome))
    }
}
