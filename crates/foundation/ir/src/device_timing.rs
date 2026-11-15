//! Backend-neutral device timing of executed graphs, attributed to dataflow nodes.
//!
//! A backend measures where device time goes and reports it as [`DeviceTimingSample`]s; the
//! runtime aggregates them into bounded metrics. Attribution comes from the dataflow graph a
//! backend executes, never from a kernel name, so every model a provider lowers to this IR, and
//! every kernel a backend lowers a node to, is covered without per-model or per-kernel code:
//! [`OpClass::of`] is the one place an operator is classified.
use crate::dataflow::{TensorNode, TensorOp};
use serde::{Deserialize, Serialize};

/// How much device time a backend measures; fixed when the model loads, because measurement
/// points are part of captured graphs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTimingMode {
    /// No measurement points: execution is exactly the unmeasured path.
    Off,
    /// One measurement around each executed graph; adds no work between kernels.
    #[default]
    Graph,
    /// A measurement between consecutive dataflow nodes. Neighbouring kernels can no longer
    /// overlap, so node times run slightly long; meant for investigation, not production.
    Node,
}

impl DeviceTimingMode {
    /// Every mode, in declaration order.
    pub const ALL: [Self; 3] = [Self::Off, Self::Graph, Self::Node];

    /// Stable configuration and export name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Graph => "graph",
            Self::Node => "node",
        }
    }
}

/// Graph executions between two read-back samples under the default configuration.
pub const DEFAULT_SAMPLE_EVERY: u32 = 16;

/// Device timing selected for one deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeviceTimingConfig {
    pub mode: DeviceTimingMode,
    /// Read back one of every `sample_every` measured executions; at least one.
    pub sample_every: u32,
}

impl Default for DeviceTimingConfig {
    fn default() -> Self {
        Self {
            mode: DeviceTimingMode::default(),
            sample_every: DEFAULT_SAMPLE_EVERY,
        }
    }
}

impl DeviceTimingConfig {
    /// Whether execution number `execution` (counting from zero) is read back.
    #[must_use]
    pub fn samples(&self, execution: u64) -> bool {
        !matches!(self.mode, DeviceTimingMode::Off)
            && execution.is_multiple_of(u64::from(self.sample_every.max(1)))
    }
}

/// The kind of work an executed graph performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimedPhase {
    Prefill,
    Decode,
    /// Target verification of speculative tokens.
    Verify,
    /// Draft proposal of speculative tokens.
    Draft,
    /// Accepted-token commit or state restore after verification.
    Commit,
    Other,
}

impl TimedPhase {
    /// Every phase, in declaration order; positions index fixed aggregate tables.
    pub const ALL: [Self; 6] = [
        Self::Prefill,
        Self::Decode,
        Self::Verify,
        Self::Draft,
        Self::Commit,
        Self::Other,
    ];

    /// Stable export name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::Draft => "draft",
            Self::Commit => "commit",
            Self::Other => "other",
        }
    }

    /// Position in [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Which model's graph a segment belongs to: the served model or a speculative draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimedGraph {
    Target,
    Draft,
}

/// Operator class of a dataflow node: what kind of computation the device time paid for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpClass {
    Embedding,
    /// Dense or quantized matrix products, including expert projections.
    Projection,
    Attention,
    /// Linear-attention recurrences such as the gated delta rule.
    Recurrence,
    Convolution,
    Norm,
    Rope,
    /// Element-wise activations, products and sums.
    Elementwise,
    /// Tensor reshaping without arithmetic.
    Layout,
    /// Mixture-of-experts routing and combination.
    Routing,
    /// Work outside any dataflow node: readout, sampling, copies the backend adds.
    Runtime,
}

impl OpClass {
    /// Every class, in declaration order; positions index fixed aggregate tables.
    pub const ALL: [Self; 11] = [
        Self::Embedding,
        Self::Projection,
        Self::Attention,
        Self::Recurrence,
        Self::Convolution,
        Self::Norm,
        Self::Rope,
        Self::Elementwise,
        Self::Layout,
        Self::Routing,
        Self::Runtime,
    ];

    /// The class of `op`. Adding an operator to [`TensorOp`] fails to compile here until it is
    /// classified, so new operators cannot silently escape attribution.
    #[must_use]
    pub const fn of(op: &TensorOp) -> Self {
        match op {
            TensorOp::Embedding => Self::Embedding,
            TensorOp::Linear | TensorOp::ExpertLinear { .. } => Self::Projection,
            TensorOp::Attention { .. } => Self::Attention,
            TensorOp::Delta { .. } => Self::Recurrence,
            TensorOp::Conv { .. } => Self::Convolution,
            TensorOp::Norm { .. } | TensorOp::GatedNorm { .. } => Self::Norm,
            TensorOp::Rope { .. } => Self::Rope,
            TensorOp::Silu
            | TensorOp::Sigmoid
            | TensorOp::Multiply
            | TensorOp::Scale
            | TensorOp::Add => Self::Elementwise,
            TensorOp::Split { .. } => Self::Layout,
            TensorOp::MoeRoute { .. } | TensorOp::MoeCombine { .. } => Self::Routing,
        }
    }

    /// Stable export name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Projection => "projection",
            Self::Attention => "attention",
            Self::Recurrence => "recurrence",
            Self::Convolution => "convolution",
            Self::Norm => "norm",
            Self::Rope => "rope",
            Self::Elementwise => "elementwise",
            Self::Layout => "layout",
            Self::Routing => "routing",
            Self::Runtime => "runtime",
        }
    }

    /// Position in [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Device time of one dataflow node, or of backend work between nodes, in one execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimedSegment {
    pub graph: TimedGraph,
    /// Index of the node in its graph; `None` for work no node owns ([`OpClass::Runtime`]).
    pub node: Option<u32>,
    pub op: OpClass,
    pub layer: Option<u16>,
    /// Index into the implementation names the backend declares once (the SPI's
    /// `device_timing_variants`), naming what the node lowered to; `None` when the backend does
    /// not distinguish implementations.
    pub variant: Option<u16>,
    pub device_ns: u64,
}

impl TimedSegment {
    /// A segment for `node` at `index` of `graph`, classified by [`OpClass::of`].
    #[must_use]
    pub fn of_node(graph: TimedGraph, index: usize, node: &TensorNode, device_ns: u64) -> Self {
        Self {
            graph,
            node: u32::try_from(index).ok(),
            op: OpClass::of(&node.op),
            layer: node.layer.and_then(|layer| u16::try_from(layer).ok()),
            variant: None,
            device_ns,
        }
    }
}

/// Device time of one executed graph and, under [`DeviceTimingMode::Node`], of its nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceTimingSample {
    pub phase: TimedPhase,
    /// Sequences (or prompt rows) the graph executed for.
    pub rows: u32,
    /// Device time from the first to the last measured point of the graph.
    pub total_ns: u64,
    /// Node segments in execution order; empty under [`DeviceTimingMode::Graph`].
    pub segments: Vec<TimedSegment>,
}
