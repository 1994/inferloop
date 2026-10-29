use crate::{
    AgentBackend, AgentService, experiment::AccuracyConstraints, experiment::BenchmarkPlan,
    experiment::ExperimentConstraints, experiment::ExperimentPlan, registry::CommandDescriptor,
    registry::CommandEffect, registry::CommandRegistry, transport,
};
#[path = "../../../../engine/runtime/tests/support/mod.rs"]
mod support;

use infer_core::{Error, ModelId, RequestId, Result};
use infer_ir::{
    CanonicalRequest, ExecutionStats, LayerProbe, OpTrace, PrecisionPlan, Qos, RequestInput,
    Sampling, Workload, WorkloadOutput,
};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, collections::BTreeSet, io::Cursor};

impl AgentBackend for support::ProtocolBackend {
    fn fresh(&self) -> Result<Self> {
        Self::new(16, 8, &support::model(ModelId::ONE))
    }
    fn registry(&self) -> Result<KernelRegistry> {
        let mut registry = KernelRegistry::default();
        registry.register(&support::DeclaredKernels)?;
        Ok(registry)
    }
    fn inspection(&self) -> Value {
        json!({"kind":"declared-protocol-double"})
    }
    fn traces(&self) -> Vec<OpTrace> {
        Vec::new()
    }
    fn trace_timing_scope(&self) -> &'static str {
        "unavailable"
    }
    fn probes(&self) -> Vec<LayerProbe> {
        Vec::new()
    }
    fn profile(&self) -> Value {
        json!({"scope":"declared-protocol-double","traceEvents":[]})
    }
    fn execution_stats(&self) -> Option<ExecutionStats> {
        None
    }
}

fn service() -> Result<AgentService<support::ProtocolBackend>> {
    let model = support::model(ModelId::ONE);
    let backend = support::ProtocolBackend::new(16, 8, &model)?;
    let registry = AgentBackend::registry(&backend)?;
    AgentService::new(
        Engine::new(
            backend,
            model,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig::default(),
        )?,
        json!({"primary_target":"cuda"}),
    )
}
fn request() -> CanonicalRequest {
    CanonicalRequest {
        id: RequestId::ONE,
        model: ModelId::ONE,
        session: None,
        input: RequestInput::Sequence {
            tokens: vec![1, 2, 3].into(),
            media: vec![],
        },
        workload: Workload::Generate { max_new_tokens: 3 },
        qos: Qos::default(),
        sampling: Sampling {
            temperature: 0.0,
            seed: 7,
            ..Sampling::default()
        },
        prefix_boundary: None,
        extensions: BTreeMap::new(),
    }
}
fn response(
    service: &mut AgentService<support::ProtocolBackend>,
    method: &str,
    params: Value,
) -> Result<Value> {
    let mut message = json!({"jsonrpc":"2.0","id":1,"method":method});
    message["params"] = params;
    service
        .handle(message)
        .ok_or_else(|| Error::invariant("expected an RPC response"))
}
fn experiment_plan() -> ExperimentPlan {
    ExperimentPlan {
        baseline: RuntimeConfig::default(),
        candidate: RuntimeConfig {
            max_num_batched_tokens: 1,
            max_num_seqs: 1,
            ..RuntimeConfig::default()
        },
        workload: BenchmarkPlan {
            requests: vec![request()],
            ttft_slo_us: u64::MAX,
            tpot_slo_us: u64::MAX,
        },
        constraints: ExperimentConstraints::default(),
    }
}

#[test]
fn discovery_matches_registered_commands_and_legacy_aliases() -> Result<()> {
    let mut service = service()?;
    let result = response(&mut service, "agent.discover", json!({}))?;
    let commands = result["result"]["commands"]
        .as_array()
        .ok_or_else(|| Error::invariant("missing catalog"))?;
    assert!(commands.iter().any(|c| c["method"] == "experiment.run"));
    assert!(commands.iter().any(|c| c["method"] == "runtime.graph"));
    assert_eq!(result["result"]["protocol_version"], "1.0");
    for command in commands {
        assert!(command["params"].is_object());
        assert!(command["effect"].is_string());
    }
    let result = response(
        &mut service,
        "verify",
        json!({"reference":[1.0],"candidate":[1.0],"atol":0.0,"rtol":0.0}),
    )?;
    assert_eq!(result["result"]["passed"], true);
    Ok(())
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}
fn count(context: &mut usize, _: Empty) -> Value {
    *context += 1;
    json!(*context)
}
fn descriptor(method: &'static str, aliases: Vec<&'static str>) -> CommandDescriptor {
    CommandDescriptor {
        method,
        aliases,
        description: "increment test counter",
        effect: CommandEffect::Control,
        params: json!({"type":"object"}),
    }
}
#[test]
fn registration_conflicts_do_not_partially_install_aliases() -> Result<()> {
    let mut registry = CommandRegistry::default();
    registry.register(descriptor("counter.run", vec!["counter"]), count)?;
    assert!(
        registry
            .register(
                descriptor("new.command", vec!["new.alias", "counter"]),
                count
            )
            .is_err()
    );
    assert!(registry.descriptor("new.command").is_none());
    assert!(registry.descriptor("new.alias").is_none());
    assert!(
        registry
            .register(descriptor("rpc.reserved", vec![]), count)
            .is_err()
    );
    let mut context = 0;
    let outcome = registry
        .dispatch(&mut context, "counter", json!({}))
        .ok_or_else(|| Error::invariant("missing alias"))??;
    assert_eq!(outcome, 1);
    Ok(())
}

#[test]
fn batches_preserve_notification_semantics_and_null_ids() -> Result<()> {
    let mut service = service()?;
    let result = service
        .handle(json!([
            {"jsonrpc":"2.0","method":"runtime.tick","params":{"now_us":7}},
            {"jsonrpc":"2.0","id":null,"method":"runtime.inspect"},
            {"jsonrpc":"2.0","id":"unknown","method":"unknown"},42
        ]))
        .ok_or_else(|| Error::invariant("missing batch response"))?;
    let responses = result
        .as_array()
        .ok_or_else(|| Error::invariant("batch is not an array"))?;
    assert_eq!(responses.len(), 3);
    assert!(responses[0]["id"].is_null());
    assert_eq!(responses[1]["error"]["code"], -32601);
    assert_eq!(responses[2]["error"]["code"], -32600);
    assert_eq!(service.context().engine().now_us(), 7);
    assert!(
        service
            .handle(json!([{"jsonrpc":"2.0","method":"unknown"}]))
            .is_none()
    );
    assert_eq!(
        service
            .handle(json!([]))
            .ok_or_else(|| Error::invariant("missing empty batch error"))?["error"]["code"],
        -32600
    );
    Ok(())
}

#[test]
fn invalid_parameters_never_mutate_live_control_state() -> Result<()> {
    let mut service = service()?;
    for params in [
        json!({"now_us":8,"extra":1}),
        json!({"now_us":-1}),
        Value::Null,
        json!(true),
    ] {
        let result = response(&mut service, "runtime.tick", params)?;
        assert_eq!(result["error"]["code"], -32602);
        assert_eq!(service.context().engine().now_us(), 0);
    }
    assert_eq!(
        response(&mut service, "runtime.inspect", json!({"extra":1}))?["error"]["code"],
        -32602
    );
    assert_eq!(
        service
            .handle(
                json!({"jsonrpc":"2.0","id":true,"method":"runtime.tick","params":{"now_us":9}})
            )
            .ok_or_else(|| Error::invariant("missing invalid id error"))?["error"]["code"],
        -32600
    );
    Ok(())
}

#[test]
fn framing_discards_oversized_input_and_recovers_on_next_line() -> Result<()> {
    let mut service = service()?;
    let mut input = vec![b'x'; transport::MAX_FRAME_BYTES + 1];
    input.extend_from_slice(b"\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"runtime.inspect\"}\n");
    let mut output = Vec::new();
    transport::serve(&mut service, &mut Cursor::new(input), &mut output)?;
    let lines = String::from_utf8(output).map_err(|e| Error::invalid(e.to_string()))?;
    let responses = lines
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).map_err(|e| Error::invalid(e.to_string())))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(responses.len(), 2);
    assert_eq!(responses[0]["error"]["code"], -32000);
    assert_eq!(responses[1]["id"], 2);
    assert_eq!(responses[1]["result"]["active_requests"], 0);
    Ok(())
}

#[test]
fn graph_edges_resolve_actual_decisions_and_registered_sources() -> Result<()> {
    let mut service = service()?;
    response(&mut service, "runtime.submit", json!(request()))?;
    for now_us in 1..100 {
        response(&mut service, "runtime.tick", json!({"now_us":now_us}))?;
        if service.context().engine().is_idle() {
            break;
        }
    }
    let graph = response(&mut service, "runtime.graph", json!({"request_id":1}))?;
    let nodes = graph["result"]["nodes"]
        .as_array()
        .ok_or_else(|| Error::invariant("missing nodes"))?;
    let ids: BTreeSet<_> = nodes.iter().filter_map(|n| n["id"].as_str()).collect();
    let edges = graph["result"]["edges"]
        .as_array()
        .ok_or_else(|| Error::invariant("missing edges"))?;
    assert!(edges.iter().any(|e| e["relation"] == "scheduled_by"));
    assert!(edges.iter().any(|e| e["relation"] == "lowered_to"));
    for edge in edges {
        assert!(edge["from"].as_str().is_some_and(|id| ids.contains(id)));
        assert!(edge["to"].as_str().is_some_and(|id| ids.contains(id)));
    }
    assert!(
        nodes
            .iter()
            .any(|n| n["kind"] == "kernel" && n["evidence"]["source"]["file"].is_string())
    );
    assert_eq!(
        response(&mut service, "runtime.graph", json!({"request_id":999}))?["error"]["data"]["code"],
        "NotFound"
    );
    Ok(())
}

#[test]
fn failed_checkpoint_restore_keeps_live_engine_and_session() -> Result<()> {
    let mut service = service()?;
    response(&mut service, "runtime.submit", json!(request()))?;
    let captured = response(&mut service, "snapshot.capture", json!({}))?;
    let mut snapshot = captured["result"]["snapshot"].clone();
    snapshot["backend_identity"] = json!("different-backend");
    let before = json!(service.context().engine().inspect());
    let result = response(&mut service, "snapshot.replay", snapshot)?;
    assert!(result.get("error").is_some());
    assert_eq!(json!(service.context().engine().inspect()), before);
    assert_eq!(
        service
            .context()
            .engine()
            .request(RequestId::ONE)?
            .request
            .id,
        RequestId::ONE
    );
    Ok(())
}

#[test]
fn experiment_executes_independent_engines_and_retains_evidence() -> Result<()> {
    let mut service = service()?;
    response(&mut service, "runtime.submit", json!(request()))?;
    let before = json!(service.context().engine().inspect());
    let result = response(&mut service, "experiment.run", json!(experiment_plan()))?;
    assert!(result.get("error").is_none(), "{result}");
    assert_eq!(result["result"]["baseline"]["successful"], 1);
    assert_eq!(result["result"]["candidate"]["successful"], 1);
    assert_eq!(result["result"]["correctness"][0]["passed"], true);
    assert_eq!(
        result["result"]["baseline"]["workload_fingerprint"],
        result["result"]["candidate"]["workload_fingerprint"]
    );
    assert_eq!(json!(service.context().engine().inspect()), before);
    assert!(
        service
            .context()
            .engine()
            .request(RequestId::ONE)?
            .completed
            .is_none()
    );
    let inspection = response(
        &mut service,
        "experiment.inspect",
        json!({"experiment_id":result["result"]["id"]}),
    )?;
    assert_eq!(
        inspection["result"]["workload_fingerprint"],
        result["result"]["workload_fingerprint"]
    );
    Ok(())
}

#[test]
fn independent_reference_mismatch_and_missing_resource_evidence_reject_candidate() -> Result<()> {
    let mut service = service()?;
    let mut plan = experiment_plan();
    plan.constraints.accuracy = AccuracyConstraints {
        reference_outputs: Some(BTreeMap::from([(
            RequestId::ONE,
            WorkloadOutput::Tokens(vec![u32::MAX].into()),
        )])),
        ..AccuracyConstraints::default()
    };
    plan.constraints.max_state_bytes = Some(0);
    let result = response(&mut service, "experiment.run", json!(plan))?;
    assert_eq!(result["result"]["verdict"]["accepted"], false, "{result}");
    assert_eq!(result["result"]["correctness"][1]["passed"], false);
    assert_eq!(result["result"]["correctness"][2]["passed"], false);
    let reasons = result["result"]["verdict"]["reasons"]
        .as_array()
        .ok_or_else(|| Error::invariant("missing rejection reasons"))?;
    assert!(reasons.iter().any(|r| r == "correctness failed"));
    assert!(
        reasons
            .iter()
            .any(|r| r == "backend cannot provide state memory evidence")
    );
    Ok(())
}

#[test]
fn malformed_experiment_is_rejected_before_allocating_history() -> Result<()> {
    let mut service = service()?;
    let mut plan = experiment_plan();
    plan.workload.requests.push(request());
    assert_eq!(
        response(&mut service, "experiment.run", json!(plan))?["error"]["code"],
        -32602
    );
    assert_eq!(
        response(&mut service, "experiment.list", json!({}))?["result"],
        json!([])
    );
    Ok(())
}

#[test]
fn session_history_is_bounded_and_discloses_dropped_calls() -> Result<()> {
    let mut service = service()?;
    for _ in 0..300 {
        response(&mut service, "runtime.inspect", json!({}))?;
    }
    let inspected = response(&mut service, "agent.inspect", json!({}))?;
    assert_eq!(
        inspected["result"]["calls"].as_array().map(Vec::len),
        Some(256)
    );
    assert_eq!(inspected["result"]["dropped_calls"], 44);
    Ok(())
}

#[test]
fn semantic_observation_is_non_destructive_and_survives_legacy_event_drain() {
    let mut agent = service().unwrap();
    let submitted = agent
        .handle(json!({"jsonrpc":"2.0","id":1,"method":"runtime.submit","params":request()}))
        .unwrap();
    assert!(submitted.get("error").is_none());
    for clock in 0..20 {
        let response = agent
            .handle(
                json!({"jsonrpc":"2.0","id":2,"method":"runtime.tick","params":{"now_us":clock}}),
            )
            .unwrap();
        assert!(response.get("error").is_none());
    }
    let query =
        json!({"jsonrpc":"2.0","id":3,"method":"observability.events","params":{"request_id":1}});
    let first = agent.handle(query.clone()).unwrap();
    assert!(
        first["result"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["event"]["kind"] == "Finished")
    );
    let drained = agent
        .handle(json!({"jsonrpc":"2.0","id":4,"method":"events.drain"}))
        .unwrap();
    assert_ne!(drained["result"], json!([]));
    assert_eq!(first["result"], agent.handle(query).unwrap()["result"]);
    let summary = agent
        .handle(json!({"jsonrpc":"2.0","id":5,"method":"observability.inspect"}))
        .unwrap();
    assert_eq!(summary["result"]["metrics"]["successful_requests"], 1);
    assert_eq!(summary["result"]["trace_coverage"]["complete_requests"], 1);
    let invalid = agent
        .handle(
            json!({"jsonrpc":"2.0","id":6,"method":"observability.events","params":{"limit":4097}}),
        )
        .unwrap();
    assert_eq!(invalid["error"]["code"], -32602);
}
