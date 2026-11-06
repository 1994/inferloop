//! Test-only copies around the unchanged attention kernels; allocations precede capture.
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::{device::CudaDevice, resident::ProgramWeights, tuning::attention::Choice};
use cutile::prelude::DeviceError;
use infer_core::{Result, TensorId};
use infer_cuda_kernels::contract::{Buffer, Recording};
use infer_cuda_kernels::tile::runtime::Allocation;
use infer_ir::{DataflowGraph, TensorOp};
use std::result::Result as StdResult;
use std::{cell::Cell, collections::BTreeMap};
thread_local! {
    static FORCED: Cell<Option<Choice>> = const { Cell::new(None) };
}
pub fn forced_choice() -> Option<Choice> {
    FORCED.get()
}
pub fn with_choice<T>(choice: Choice, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<Choice>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED.set(self.0);
        }
    }
    let _restore = Restore(FORCED.replace(Some(choice)));
    run()
}
pub struct Entry {
    pub state: TensorId,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub capacity: usize,
    pub window: usize,
    pub scales: [f32; 2],
    pub query: Allocation<f32>,
    pub keys: Allocation<f32>,
    pub values: Allocation<f32>,
    pub output: Allocation<f32>,
    pub metadata: Allocation<i32>,
    pub table: Allocation<i32>,
}
#[derive(Default)]
pub struct Taps {
    pub buffers: BTreeMap<(TensorId, usize), Entry>,
}
impl Taps {
    pub fn prepare(
        &mut self,
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        lanes: usize,
        capacity: usize,
    ) -> Result<()> {
        if forced_choice().is_none() {
            return Ok(());
        }
        for node in &graph.nodes {
            let TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window,
            } = node.op
            else {
                continue;
            };
            let Some(&state) = node.states.first() else {
                continue;
            };
            let Some(&scales) = weights.kv_scales.get(&state) else {
                continue;
            };
            let allocate = |size| device.runtime().zeros(size);
            let entry = Entry {
                state,
                heads: query_heads,
                kv_heads,
                dim: head_dim,
                capacity,
                window: window.unwrap_or(0),
                scales,
                query: allocate(lanes * query_heads * head_dim)?,
                keys: allocate(lanes * kv_heads * head_dim)?,
                values: allocate(lanes * kv_heads * head_dim)?,
                output: allocate(lanes * query_heads * head_dim)?,
                metadata: device.runtime().zeros(METADATA_FIELDS)?,
                table: device.runtime().zeros(capacity / KV_BLOCK_TOKENS)?,
            };
            self.buffers.insert((node.outputs[0], lanes), entry);
        }
        Ok(())
    }
    pub fn inputs(
        &self,
        scope: &Recording<'_>,
        key: (TensorId, usize),
        [query, keys, values]: [Buffer<'_, f32>; 3],
        (metadata, table): (Buffer<'_, i32>, Buffer<'_, i32>),
    ) -> StdResult<(), DeviceError> {
        if let Some(tap) = self.buffers.get(&key) {
            let recording = scope;
            let internal = |error: infer_core::Error| DeviceError::Internal(error.to_string());
            for (out, input) in [
                (&tap.query, query),
                (&tap.keys, keys),
                (&tap.values, values),
            ] {
                recording.copy(out.operand(), input).map_err(internal)?;
            }
            recording
                .copy(tap.metadata.operand(), metadata)
                .map_err(internal)?;
            recording
                .copy(tap.table.operand(), table)
                .map_err(internal)?;
        }
        Ok(())
    }
    pub fn output(
        &self,
        scope: &Recording<'_>,
        key: (TensorId, usize),
        output: Buffer<'_, f32>,
    ) -> StdResult<(), DeviceError> {
        if let Some(tap) = self.buffers.get(&key) {
            scope
                .copy(tap.output.operand(), output)
                .map_err(|error| DeviceError::Internal(error.to_string()))?;
        }
        Ok(())
    }
}
