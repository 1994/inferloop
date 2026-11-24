//! Persistent immutable metadata slots. Readers retain an Arc until their fence/output work ends.
use infer_core::{DecisionId, Error, ErrorCode, ProgramId, Result, StepId};
use infer_ir::{CostEstimate, ExecutionRole, StepPlan};
use std::mem::swap;
use std::sync::Arc;

/// Immutable step-metadata slots; a slot cannot be reused while a reader holds its `Arc`.
const STEP_POOL_SLOTS: usize = 4;

pub struct StepPool {
    slots: [Arc<StepPlan>; STEP_POOL_SLOTS],
}
impl StepPool {
    /// # Errors
    /// Reports inability to reserve fixed batch metadata slots.
    pub fn new(batch: usize, program: ProgramId) -> Result<Self> {
        fn slot(batch: usize, program: ProgramId) -> Result<Arc<StepPlan>> {
            let mut work = Vec::new();
            work.try_reserve_exact(batch)
                .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
            Ok(Arc::new(StepPlan {
                id: StepId::ONE,
                decision: DecisionId::ONE,
                program,
                role: ExecutionRole::Prefill,
                work,
                cost: CostEstimate::default(),
                graph: None,
                quantum_overrun: false,
            }))
        }
        if batch == 0 || batch > infer_gpu_api::MAX_SUBMISSION_BATCH {
            return Err(Error::invalid(
                "CPU step pool requires batch capacity 1..=64",
            ));
        }
        Ok(Self {
            slots: [
                slot(batch, program)?,
                slot(batch, program)?,
                slot(batch, program)?,
                slot(batch, program)?,
            ],
        })
    }
    /// # Errors
    /// Returns capacity while all slots have readers; publication never overwrites a live reader.
    pub fn seal(&mut self, plan: &mut StepPlan) -> Result<Arc<StepPlan>> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| Arc::strong_count(slot) == 1)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::Capacity,
                    "CPU step readers exhausted metadata pool",
                )
            })?;
        let storage =
            Arc::get_mut(slot).ok_or_else(|| Error::invariant("step slot reader appeared"))?;
        swap(storage, plan);
        plan.work.clear();
        Ok(slot.clone())
    }
}

/// Fixed cost-feedback records remain shared with replay/checkpoint readers.
pub struct QueryPool {
    slots: Vec<infer_ir::SharedOutput<infer_ir::CostQuery>>,
    cursor: usize,
}
impl QueryPool {
    pub fn new(history: usize, batch: usize) -> Result<Self> {
        let count = history
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| Error::invalid("cost query pool overflow"))?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        for _ in 0..count {
            slots.push(infer_ir::SharedOutput::with_capacity(batch)?);
        }
        Ok(Self { slots, cursor: 0 })
    }
    pub fn seal(
        &mut self,
        queries: &[infer_ir::CostQuery],
    ) -> Result<infer_ir::SharedOutput<infer_ir::CostQuery>> {
        for _ in 0..self.slots.len() {
            let index = self.cursor;
            self.cursor = (self.cursor + 1) % self.slots.len();
            if self.slots[index].replace_if_unique(queries) {
                return Ok(self.slots[index].clone());
            }
        }
        Err(Error::new(
            ErrorCode::Capacity,
            "cost feedback readers exhausted query pool",
        ))
    }
}

#[cfg(test)]
mod tests;
