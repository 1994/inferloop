use crate::AgentBackend;
use infer_core::{RequestId, Result};
use infer_runtime::Engine;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize)]
pub struct SemanticNode {
    pub id: String,
    pub kind: &'static str,
    pub evidence: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct SemanticEdge {
    pub from: String,
    pub relation: &'static str,
    pub to: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SemanticGraph {
    pub nodes: Vec<SemanticNode>,
    pub edges: Vec<SemanticEdge>,
    pub dropped_events: u64,
    pub dropped_actions: u64,
    pub dropped_traces: u64,
}

#[derive(Default)]
struct Builder {
    nodes: BTreeMap<String, SemanticNode>,
    edges: BTreeSet<SemanticEdge>,
}

impl Builder {
    fn node(&mut self, kind: &'static str, raw: u64, evidence: Value) -> String {
        let id = format!("{kind}:{raw}");
        self.nodes
            .entry(id.clone())
            .or_insert_with(|| SemanticNode {
                id: id.clone(),
                kind,
                evidence,
            });
        id
    }
    fn edge(&mut self, from: &str, relation: &'static str, to: &str) {
        self.edges.insert(SemanticEdge {
            from: from.into(),
            relation,
            to: to.into(),
        });
    }
    fn program<B: AgentBackend>(&mut self, engine: &Engine<B>) -> Result<String> {
        let program = self.node(
            "program",
            engine.program().id.get(),
            json!({"backend":engine.program().backend}),
        );
        let registry = engine.backend().registry()?;
        for operation in &engine.program().operations {
            let op = self.node("op", operation.op.id.get(), json!(&operation.op));
            let kernel = self.node(
                "kernel",
                operation.kernel.get(),
                json!({"source":registry.get(operation.kernel).map(|k| &k.source)}),
            );
            if let Some(source) = registry.get(operation.kernel).map(|kernel| &kernel.source) {
                let source_id = format!(
                    "source:{}:{}:{}",
                    source.crate_name, source.file, source.function
                );
                self.nodes
                    .entry(source_id.clone())
                    .or_insert_with(|| SemanticNode {
                        id: source_id.clone(),
                        kind: "source",
                        evidence: json!(source),
                    });
                self.edge(&kernel, "implemented_by", &source_id);
            }
            self.edge(&program, "contains", &op);
            self.edge(&op, "lowered_to", &kernel);
        }
        Ok(program)
    }
}

/// Build request-to-source relationships from actual decisions and executed operations.
/// # Errors
/// Returns not-found for unknown requests or backend kernel registration errors.
pub fn query<B: AgentBackend>(
    engine: &Engine<B>,
    request: Option<RequestId>,
) -> Result<SemanticGraph> {
    if let Some(request) = request {
        engine.request(request)?;
    }
    let mut graph = Builder::default();
    let program = graph.program(engine)?;
    for record in engine
        .request_records()
        .filter(|r| request.is_none_or(|id| r.request.id == id))
    {
        let req = graph.node("request", record.request.id.get(), json!({"status":record.status,"admission":record.admission,"progress_epoch":record.progress_epoch}));
        if let Some(state) = record.state {
            let state = graph.node("state", state.get(), json!({"active":true}));
            graph.edge(&req, "owns", &state);
        }
        for decision in engine.decisions() {
            let selected = decision
                .step
                .as_ref()
                .is_some_and(|s| s.work.iter().any(|w| w.request == record.request.id));
            let deferred = decision
                .deferred
                .iter()
                .any(|w| w.request == record.request.id);
            if !selected && !deferred {
                continue;
            }
            let reason = graph.node("decision", decision.id.get(), json!(decision));
            graph.edge(&req, "scheduled_by", &reason);
            if selected && let Some(step) = &decision.step {
                let step = graph.node(
                    "step",
                    step.id.get(),
                    json!({"role":step.role,"cost":step.cost}),
                );
                graph.edge(&req, "executed_by", &step);
                graph.edge(&reason, "planned", &step);
                graph.edge(&step, "executes", &program);
            }
        }
    }
    for trace in engine
        .backend()
        .traces()
        .iter()
        .filter(|t| request.is_none_or(|id| t.request == id))
    {
        let req = graph.node(
            "request",
            trace.request.get(),
            json!({"retained_in_trace":true}),
        );
        let step = graph.node("step", trace.step.get(), json!({"retained_in_trace":true}));
        let state = graph.node(
            "state",
            trace.state.get(),
            json!({"retained_in_trace":true}),
        );
        let op = format!("op:{}", trace.op);
        graph.edge(&req, "executed_by", &step);
        graph.edge(&req, "used_state", &state);
        graph.edge(&step, "executes", &program);
        if graph.nodes.contains_key(&op) {
            graph.edge(&step, "executed_op", &op);
        }
    }
    let inspection = engine.inspect();
    Ok(SemanticGraph {
        nodes: graph.nodes.into_values().collect(),
        edges: graph.edges.into_iter().collect(),
        dropped_events: inspection.dropped_events,
        dropped_actions: inspection.dropped_actions,
        dropped_traces: engine
            .backend()
            .execution_stats()
            .map_or(0, |s| s.trace_dropped),
    })
}
