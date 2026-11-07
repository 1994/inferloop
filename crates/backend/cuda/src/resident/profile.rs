//! Optional per-node CUDA event timing inside the captured width-32 prefill graphs and
//! the shared slot decode graph.
//!
//! Set `INFER_CUDA_PREFILL_PROFILE` to a writable JSONL path: at capture time a
//! timing-enabled event is recorded into the graph between consecutive nodes (with
//! `CU_EVENT_RECORD_EXTERNAL` so the records stay queryable after replay), and every
//! 32-lane prefill chunk or slot decode replay appends one JSON line of per-segment
//! device times (per-op totals plus every segment). With the variable unset no events
//! are created, no graph nodes change, and the runtime path is exactly the baseline.
//! Set `INFER_CUDA_PROFILE_GRAPH_ONLY=1` as well to record just the graph boundaries,
//! reducing instrumentation overhead when separating host time from device time.

use crate::device::{CudaDevice, device_error};
use infer_core::{Error, Result};
use infer_cuda_kernels::contract::Recording;
use infer_cuda_kernels::runtime::{CudaEvent, Device as Runtime};
use infer_cuda_kernels::runtime::{record_event, timing_event};
use infer_ir::{TensorNode, TensorOp};
use std::collections::BTreeMap;
use std::env::var;
use std::env::var_os;
use std::fs::OpenOptions;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

/// Environment variable enabling in-graph prefill profiling; the value is the JSONL output path.
const PROFILE_ENV: &str = "INFER_CUDA_PREFILL_PROFILE";

/// One recorded graph node between two boundary events.
struct Segment {
    node: usize,
    op: &'static str,
    layer: Option<usize>,
}

/// Boundary events recorded into one captured graph: one per node plus a closing event.
pub(super) struct GraphProfile {
    device: Runtime,
    segments: Vec<Segment>,
    events: Vec<Arc<CudaEvent>>,
    graph_only: bool,
    node_count: usize,
}

impl GraphProfile {
    fn new(device: Runtime) -> Self {
        Self {
            device,
            segments: Vec::new(),
            events: Vec::new(),
            graph_only: var("INFER_CUDA_PROFILE_GRAPH_ONLY").is_ok_and(|v| v == "1"),
            node_count: 0,
        }
    }

    /// Record a timing event marking the start of `node`'s kernels in the captured graph.
    pub(super) fn boundary(
        &mut self,
        scope: &Recording<'_>,
        node: usize,
        op: &TensorNode,
    ) -> Result<()> {
        self.node_count += 1;
        if self.graph_only && !self.segments.is_empty() {
            return Ok(());
        }
        self.mark(scope)?;
        self.segments.push(Segment {
            node,
            op: if self.graph_only {
                "graph"
            } else {
                category(&op.op)
            },
            layer: op.layer,
        });
        Ok(())
    }

    /// Record the closing event after the last node.
    pub(super) fn finish(&mut self, scope: &Recording<'_>) -> Result<()> {
        self.mark(scope)
    }

    fn mark(&mut self, scope: &Recording<'_>) -> Result<()> {
        let event = timing_event(&self.device)?;
        record_event(scope, &event)?;
        self.events.push(event);
        Ok(())
    }

    /// Pairwise device times for every recorded node, aggregated per op category.
    fn sample(&self, graph: &str, tokens: usize, position: usize) -> Result<serde_json::Value> {
        let mut by_op: BTreeMap<&str, f64> = BTreeMap::new();
        let mut segments = Vec::with_capacity(self.segments.len());
        let mut total = 0f64;
        for (index, segment) in self.segments.iter().enumerate() {
            let (start, end) = self
                .events
                .get(index)
                .zip(self.events.get(index + 1))
                .ok_or_else(|| Error::invariant("profile event boundary"))?;
            let ms = f64::from(start.elapsed_ms(end).map_err(device_error)?);
            total += ms;
            *by_op.entry(segment.op).or_default() += ms;
            segments.push(serde_json::json!([
                segment.node,
                segment.op,
                segment.layer,
                ms
            ]));
        }
        Ok(serde_json::json!({
            "graph": graph,
            "tokens": tokens,
            "position": position,
            "total_ms": total,
            "node_count": self.node_count,
            "by_op": by_op,
            "segments": segments,
        }))
    }
}

/// Owns the boundary events of both width-32 prefill graphs; declared after them in
/// `BatchGraph` so graph executables are destroyed before the events they record.
pub(super) struct PrefillProfile {
    path: PathBuf,
    prefill: GraphProfile,
    prefill_last: GraphProfile,
}

impl PrefillProfile {
    /// Enabled when `INFER_CUDA_PREFILL_PROFILE` names a writable output path.
    #[must_use]
    pub(super) fn from_env(device: &CudaDevice) -> Option<Self> {
        let path = var_os(PROFILE_ENV).filter(|value| !value.is_empty())?;
        let context = device.runtime();
        Some(Self {
            path: PathBuf::from(path),
            prefill: GraphProfile::new(context.clone()),
            prefill_last: GraphProfile::new(context),
        })
    }

    /// The boundary set of the graph being captured: middle chunks or the logits chunk.
    pub(super) const fn graph(&mut self, last: bool) -> &mut GraphProfile {
        if last {
            &mut self.prefill_last
        } else {
            &mut self.prefill
        }
    }

    /// Append one JSONL record for the chunk whose graph replay just completed on the
    /// stream. Best effort: a profiling failure never fails the request.
    pub(super) fn report(&self, last: bool, tokens: usize, position: usize) {
        let (graph, profile) = if last {
            ("prefill_last", &self.prefill_last)
        } else {
            ("prefill", &self.prefill)
        };
        if let Err(error) = profile
            .sample(graph, tokens, position)
            .and_then(|line| append(&self.path, &line))
        {
            eprintln!("prefill profile write failed: {error}");
        }
    }
}

/// Owns the boundary events of the shared slot decode graph; declared after the graph
/// in `SlotDecodeGraph` so the executable drops before the events it records.
pub(super) struct SlotProfile {
    path: PathBuf,
    graph: GraphProfile,
}

impl SlotProfile {
    /// Enabled when `INFER_CUDA_PREFILL_PROFILE` names a writable output path.
    pub(super) fn from_env(device: &CudaDevice) -> Option<Self> {
        let path = var_os(PROFILE_ENV).filter(|value| !value.is_empty())?;
        let context = device.runtime();
        Some(Self {
            path: PathBuf::from(path),
            graph: GraphProfile::new(context),
        })
    }

    /// The boundary set of the slot decode graph being captured.
    pub(super) const fn graph(&mut self) -> &mut GraphProfile {
        &mut self.graph
    }

    /// Append one JSONL record for the slot decode replay that just completed on the
    /// stream. Best effort: a profiling failure never fails the request.
    pub(super) fn report(&self, name: &str, lanes: usize, position: usize) {
        if let Err(error) = self
            .graph
            .sample(name, lanes, position)
            .and_then(|line| append(&self.path, &line))
        {
            eprintln!("slot profile write failed: {error}");
        }
    }
}

fn append(path: &Path, line: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| device_error(format!("prefill profile open: {error}")))?;
    writeln!(file, "{line}")
        .map_err(|error| device_error(format!("prefill profile write: {error}")))?;
    Ok(())
}

/// Op category labels, matching the provider-check op census naming.
const fn category(op: &TensorOp) -> &'static str {
    match op {
        TensorOp::Embedding => "embedding",
        TensorOp::Linear => "linear",
        TensorOp::Norm { .. } => "norm",
        TensorOp::Split { .. } => "split",
        TensorOp::Rope { .. } => "rope",
        TensorOp::Attention { .. } => "attention",
        TensorOp::Conv { .. } => "conv",
        TensorOp::Delta { .. } => "delta",
        TensorOp::GatedNorm { .. } => "gated_norm",
        TensorOp::Silu => "silu",
        TensorOp::Sigmoid => "sigmoid",
        TensorOp::Multiply | TensorOp::Scale => "multiply",
        TensorOp::MoeRoute { .. } => "moe_route",
        TensorOp::ExpertLinear { .. } => "expert_linear",
        TensorOp::MoeCombine { .. } => "moe_combine",
        TensorOp::Add => "add",
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resident_profile.rs"]
mod tests;
