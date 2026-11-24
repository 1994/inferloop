//! Internal experiment: retain every state effect and its full input dependency closure.
//! Captured storage, arithmetic and metadata stay identical to the production graph.
use infer_ir::DataflowGraph;
use std::{cell::Cell, collections::BTreeSet};

pub struct StateDemand {
    retained: Vec<bool>,
}
impl StateDemand {
    pub fn new(graph: &DataflowGraph) -> Self {
        let mut needed = BTreeSet::new();
        let mut retained = vec![false; graph.nodes.len()];
        for (index, node) in graph.nodes.iter().enumerate().rev() {
            if !node.states.is_empty() || node.outputs.iter().any(|id| needed.contains(id)) {
                retained[index] = true;
                needed.extend(node.inputs.iter().copied());
            }
        }
        Self { retained }
    }
    pub fn retains(&self, index: usize) -> bool {
        self.retained.get(index).copied().unwrap_or(true)
    }
}
thread_local! {
    static STATE_PREFILL: Cell<bool> = const { Cell::new(false) };
}
pub fn enabled() -> bool {
    STATE_PREFILL.with(Cell::get)
}
/// Controls only a CUDA test executable; no production CLI or environment entry exists.
pub fn with_state_prefill<T>(value: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            STATE_PREFILL.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(STATE_PREFILL.with(|cell| cell.replace(value)));
    run()
}

mod tests;
