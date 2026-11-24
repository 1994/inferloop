//! Capture-local common subexpression elimination for adjacent shared-input projections.
use super::super::{ProgramWeights, program::ActivationQuantization};
use crate::mlp::ProjectionWeight;
use infer_core::TensorId;
use infer_ir::{DataflowGraph, TensorNode, TensorOp};
#[cfg(test)]
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    Fp4(u32),
    Fp8Token,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    first: usize,
    last: usize,
    input: TensorId,
    columns: usize,
    kind: QuantKind,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantPairRole {
    First(Pair),
    Second { pair: Pair, index: usize },
}
impl QuantPairRole {
    const fn pair(self) -> Pair {
        match self {
            Self::First(pair) | Self::Second { pair, .. } => pair,
        }
    }
    const fn index(self) -> usize {
        match self {
            Self::First(pair) => pair.first,
            Self::Second { index, .. } => index,
        }
    }
}
pub struct QuantPairPlan {
    roles: Vec<Option<QuantPairRole>>,
}
impl QuantPairPlan {
    pub fn new(graph: &DataflowGraph, weights: &ProgramWeights) -> Self {
        Self::from_contracts(
            graph,
            |id| weights.constants.contains_key(&id),
            |id| match (
                weights.projections.get(&id),
                weights.activation_quantization(id),
            ) {
                (Some(ProjectionWeight::Fp4(..)), Some(ActivationQuantization::Fp4(scale)))
                    if scale.is_finite() && scale > 0.0 =>
                {
                    Some(QuantKind::Fp4(scale.to_bits()))
                }
                (Some(ProjectionWeight::Fp8(..)), Some(ActivationQuantization::Fp8Token)) => {
                    Some(QuantKind::Fp8Token)
                }
                _ => None,
            },
        )
    }
    fn from_contracts(
        graph: &DataflowGraph,
        constant: impl Fn(TensorId) -> bool,
        contract: impl Fn(TensorId) -> Option<QuantKind>,
    ) -> Self {
        let sizes: BTreeMap<_, _> = graph
            .tensors
            .iter()
            .filter_map(|tensor| tensor.elements().ok().map(|size| (tensor.id, size)))
            .collect();
        let eligible = |node: &TensorNode| {
            if node.op != TensorOp::Linear
                || node.inputs.len() != 2
                || node.outputs.len() != 1
                || !node.states.is_empty()
                || constant(node.inputs[0])
                || node.outputs[0] == node.inputs[0]
            {
                return None;
            }
            let columns = *sizes.get(&node.inputs[0])?;
            let outputs = *sizes.get(&node.outputs[0])?;
            if columns == 0 || outputs == 0 {
                return None;
            }
            Some((node.inputs[0], columns, contract(node.inputs[1])?))
        };
        let mut roles = vec![None; graph.nodes.len()];
        let mut index = 0;
        while index + 1 < graph.nodes.len() {
            let first = &graph.nodes[index];
            let Some((input, columns, kind)) = eligible(first) else {
                index += 1;
                continue;
            };
            let mut outputs = BTreeSet::from([first.outputs[0]]);
            let mut end = index + 1;
            while let Some(node) = graph.nodes.get(end) {
                if eligible(node) != Some((input, columns, kind))
                    || !outputs.insert(node.outputs[0])
                {
                    break;
                }
                end += 1;
            }
            if end == index + 1 {
                index += 1;
                continue;
            }
            let pair = Pair {
                first: index,
                last: end - 1,
                input,
                columns,
                kind,
            };
            roles[index] = Some(QuantPairRole::First(pair));
            for (offset, role) in roles[index + 1..end].iter_mut().enumerate() {
                *role = Some(QuantPairRole::Second {
                    pair,
                    index: index + 1 + offset,
                });
            }
            index = end;
        }
        Self { roles }
    }
    pub fn role(&self, index: usize) -> Option<QuantPairRole> {
        self.roles.get(index).copied().flatten()
    }
}

/// Every field describes the actual operand or the existing quantization buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantKey {
    pub input_pointer: u64,
    pub rows: usize,
    pub columns: usize,
    pub shape: [i32; 2],
    pub strides: [i32; 2],
    pub kind: QuantKind,
    pub codes_pointer: u64,
    pub scales_pointer: u64,
}
#[derive(Clone, Copy)]
struct Primed {
    epoch: u64,
    next: usize,
    pair: Pair,
    key: QuantKey,
}
#[derive(Clone, Copy)]
pub struct QuantTicket {
    next: usize,
    pair: Pair,
    key: QuantKey,
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantReuseStats {
    pub recorded: usize,
    pub reused: usize,
}
pub struct CaptureQuantReuse {
    epoch: u64,
    active: bool,
    enabled: bool,
    context: Option<QuantPairRole>,
    primed: Option<Primed>,
    #[cfg(test)]
    stats: QuantReuseStats,
    #[cfg(test)]
    total: QuantReuseStats,
}
impl Default for CaptureQuantReuse {
    fn default() -> Self {
        Self {
            epoch: 0,
            active: false,
            enabled: enabled(),
            context: None,
            primed: None,
            #[cfg(test)]
            stats: QuantReuseStats::default(),
            #[cfg(test)]
            total: QuantReuseStats::default(),
        }
    }
}
impl CaptureQuantReuse {
    pub const fn begin(&mut self) {
        self.clear();
        self.epoch = self.epoch.wrapping_add(1);
        self.active = true;
        #[cfg(test)]
        {
            self.stats = QuantReuseStats {
                recorded: 0,
                reused: 0,
            };
        }
    }
    pub const fn end(&mut self) {
        self.clear();
        self.active = false;
    }
    pub const fn clear(&mut self) {
        self.context = None;
        self.primed = None;
    }
    pub fn enter(&mut self, index: usize, role: Option<QuantPairRole>) {
        // A permit that was not consumed cannot certify the following node.
        if self.context.take().is_some() {
            self.primed = None;
        }
        let Some(role) = role.filter(|role| self.enabled && self.active && role.index() == index)
        else {
            self.clear();
            return;
        };
        match role {
            QuantPairRole::First(_) => self.primed = None,
            QuantPairRole::Second { pair, index } => {
                if !self.primed.is_some_and(|primed| {
                    primed.epoch == self.epoch && primed.pair == pair && primed.next == index
                }) {
                    self.primed = None;
                }
            }
        }
        self.context = Some(role);
    }
    /// Consumes one permit. Missing or incompatible permits retain the original quantizer.
    pub fn prepare(&mut self, key: Option<QuantKey>) -> (bool, Option<QuantTicket>) {
        let role = self.context.take();
        let Some((role, key)) = role.zip(key) else {
            self.clear();
            #[cfg(test)]
            self.recorded();
            return (false, None);
        };
        let pair = role.pair();
        if key.columns != pair.columns || key.kind != pair.kind {
            self.clear();
            #[cfg(test)]
            self.recorded();
            return (false, None);
        }
        match role {
            QuantPairRole::First(_) => {
                self.primed = None;
                #[cfg(test)]
                self.recorded();
                (
                    false,
                    Some(QuantTicket {
                        next: pair.first + 1,
                        pair,
                        key,
                    }),
                )
            }
            QuantPairRole::Second { index, .. } => {
                let reused = self.primed.take().is_some_and(|primed| {
                    primed.epoch == self.epoch
                        && primed.pair == pair
                        && primed.next == index
                        && primed.key == key
                });
                if reused {
                    #[cfg(test)]
                    {
                        self.stats.reused += 1;
                        self.total.reused += 1;
                    }
                } else {
                    #[cfg(test)]
                    self.recorded();
                }
                let ticket = (reused && index < pair.last).then_some(QuantTicket {
                    next: index + 1,
                    pair,
                    key,
                });
                (reused, ticket)
            }
        }
    }
    pub const fn finish(&mut self, ticket: Option<QuantTicket>, success: bool) {
        if !success {
            self.clear();
        } else if let Some(ticket) = ticket {
            self.primed = Some(Primed {
                epoch: self.epoch,
                next: ticket.next,
                pair: ticket.pair,
                key: ticket.key,
            });
        }
    }
    #[cfg(test)]
    const fn recorded(&mut self) {
        #[cfg(test)]
        {
            self.stats.recorded += 1;
            self.total.recorded += 1;
        }
    }
    #[cfg(test)]
    pub const fn set_enabled(&mut self, enabled: bool) {
        self.clear();
        self.enabled = enabled;
    }
    #[cfg(test)]
    pub const fn stats(&self) -> QuantReuseStats {
        self.stats
    }
    #[cfg(test)]
    pub const fn total(&self) -> QuantReuseStats {
        self.total
    }
}

#[cfg(not(test))]
const fn enabled() -> bool {
    true
}
#[cfg(test)]
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
}
#[cfg(test)]
fn enabled() -> bool {
    ENABLED.with(Cell::get)
}
/// Applies only to workspaces constructed on this thread; unwinding restores the preference.
#[cfg(test)]
pub fn with_quant_reuse_disabled<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.with(|enabled| enabled.set(self.0));
        }
    }
    let restore = Restore(ENABLED.with(|enabled| enabled.replace(false)));
    let result = run();
    drop(restore);
    result
}

#[cfg(test)]
mod tests;
