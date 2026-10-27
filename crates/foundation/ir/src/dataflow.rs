use crate::{DType, StateKind};
use infer_core::{Error, OpId, Result, TensorId};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, collections::BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorStorage {
    Weight { slot: String },
    Activation,
    State { layer: usize, kind: StateKind },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub storage: TensorStorage,
}
impl TensorSpec {
    /// # Errors
    /// Returns an invalid-input error if multiplying the dimensions overflows.
    pub fn elements(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| Error::invalid("dataflow shape overflow"))
        })
    }
}
/// One token invocation. Sequence position is supplied by the owning state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorOp {
    Embedding,
    Linear,
    Norm {
        epsilon: f32,
        offset: f32,
        head_dim: usize,
    },
    Split {
        widths: Vec<usize>,
        heads: usize,
    },
    Rope {
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
    },
    Attention {
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: Option<usize>,
    },
    Conv {
        channels: usize,
        kernel: usize,
    },
    Delta {
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    },
    GatedNorm {
        head_dim: usize,
        epsilon: f32,
    },
    /// Stable top-k softmax routing, encoded as [expert ids, probabilities].
    MoeRoute {
        experts: usize,
        active: usize,
        normalize: bool,
    },
    /// Selected expert matrices; inputs are activation, routes, then one matrix per expert.
    ExpertLinear {
        experts: usize,
        active: usize,
        input: usize,
        output: usize,
        routed_input: bool,
    },
    MoeCombine {
        active: usize,
        hidden: usize,
    },
    Silu,
    Sigmoid,
    Multiply,
    Scale,
    Add,
}
impl TensorOp {
    #[must_use]
    pub const fn operation(&self, lm_head: bool) -> crate::Operation {
        use crate::Operation as O;
        match self {
            Self::Embedding => O::TokenEmbedding,
            Self::Linear if lm_head => O::LmHead,
            Self::Linear => O::MatMul,
            Self::Norm { .. } => O::RmsNorm,
            Self::Split { .. } => O::Split,
            Self::Rope { .. } => O::Rope,
            Self::Attention { .. } => O::Attention,
            Self::Conv { .. } => O::Convolution,
            Self::Delta { .. } => O::LinearAttention,
            Self::GatedNorm { .. } => O::GatedNorm,
            Self::MoeRoute { .. } => O::MoeRoute,
            Self::ExpertLinear { .. } => O::ExpertMatMul,
            Self::MoeCombine { .. } => O::MoeCombine,
            Self::Silu => O::Silu,
            Self::Sigmoid => O::Sigmoid,
            Self::Multiply | Self::Scale => O::Multiply,
            Self::Add => O::Residual,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorNode {
    pub id: OpId,
    pub layer: Option<usize>,
    pub op: TensorOp,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub states: Vec<TensorId>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferLifetime {
    pub tensor: TensorId,
    pub first_node: usize,
    pub last_node: usize,
    pub slot: usize,
    pub elements: usize,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DataflowGraph {
    pub tensors: Vec<TensorSpec>,
    pub nodes: Vec<TensorNode>,
    pub hidden: Option<TensorId>,
    pub logits: Option<TensorId>,
    /// Per-token features a speculative draft reads from the target, when one is attached.
    #[serde(default)]
    pub features: Option<TensorId>,
    pub lifetimes: Vec<BufferLifetime>,
    pub scratch_elements: usize,
}
impl DataflowGraph {
    /// # Errors
    /// Returns an invalid-input or invariant error if dimensions, identities, ranges, or ownership are inconsistent.
    pub fn validate(&self) -> Result<()> {
        let mut specs = BTreeMap::new();
        let mut slots = BTreeSet::new();
        for tensor in &self.tensors {
            if tensor.elements()? == 0 || specs.insert(tensor.id, tensor).is_some() {
                return Err(Error::invalid("invalid/duplicate graph tensor"));
            }
            if let TensorStorage::Weight { slot } = &tensor.storage
                && (slot.is_empty() || !slots.insert(slot))
            {
                return Err(Error::invalid("invalid/duplicate graph weight slot"));
            }
        }
        let mut available: BTreeSet<_> = self
            .tensors
            .iter()
            .filter(|s| matches!(s.storage, TensorStorage::Weight { .. }))
            .map(|s| s.id)
            .collect();
        let mut node_ids = BTreeSet::new();
        for node in &self.nodes {
            if !node_ids.insert(node.id) || node.outputs.is_empty() {
                return Err(Error::invalid("invalid/duplicate graph node"));
            }
            if node.inputs.iter().any(|id| !available.contains(id)) {
                return Err(Error::invalid("graph input used before definition"));
            }
            for id in &node.states {
                if !specs
                    .get(id)
                    .is_some_and(|t| matches!(t.storage, TensorStorage::State { .. }))
                {
                    return Err(Error::invalid("invalid graph state binding"));
                }
            }
            for id in &node.outputs {
                if !specs
                    .get(id)
                    .is_some_and(|t| t.storage == TensorStorage::Activation)
                    || !available.insert(*id)
                {
                    return Err(Error::invalid("graph output has multiple writers"));
                }
            }
        }
        for output in [self.hidden, self.logits, self.features]
            .into_iter()
            .flatten()
        {
            if !available.contains(&output) {
                return Err(Error::invalid("undefined graph result"));
            }
        }
        if self.hidden.is_none() || self.logits.is_none() || self.nodes.is_empty() {
            return Err(Error::invalid("graph has no execution result"));
        }
        Ok(())
    }
    /// # Errors
    /// Returns an invalid-input error for an invalid graph or overflowing scratch-buffer requirements.
    pub fn plan_lifetimes(&mut self) -> Result<()> {
        self.validate()?;
        let mut bounds = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            for id in &node.outputs {
                bounds.insert(*id, (index, index));
            }
            for id in &node.inputs {
                if let Some((_, last)) = bounds.get_mut(id) {
                    *last = index;
                }
            }
        }
        for id in [self.hidden, self.logits, self.features]
            .into_iter()
            .flatten()
        {
            bounds
                .get_mut(&id)
                .ok_or_else(|| Error::invariant("validated result"))?
                .1 = self.nodes.len();
        }
        let specs: BTreeMap<_, _> = self.tensors.iter().map(|t| (t.id, t)).collect();
        let mut ordered: Vec<_> = bounds.into_iter().collect();
        ordered.sort_by_key(|(id, (first, _))| (*first, *id));
        let mut buffers = Vec::<(usize, usize)>::new(); // last use, capacity
        self.lifetimes.clear();
        for (id, (first, last)) in ordered {
            let elements = specs[&id].elements()?;
            let reusable_slot = buffers
                .iter()
                .enumerate()
                .filter(|(_, (used, capacity))| *used < first && *capacity >= elements)
                .min_by_key(|(_, (_, capacity))| *capacity)
                .map(|(slot, _)| slot);
            let slot = reusable_slot.unwrap_or_else(|| {
                buffers.push((last, elements));
                buffers.len() - 1
            });
            buffers[slot].0 = last;
            self.lifetimes.push(BufferLifetime {
                tensor: id,
                first_node: first,
                last_node: last,
                slot,
                elements,
            });
        }
        self.scratch_elements = buffers
            .iter()
            .try_fold(0usize, |n, (_, size)| n.checked_add(*size))
            .ok_or_else(|| Error::invalid("scratch overflow"))?;
        Ok(())
    }
}
