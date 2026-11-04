//! Test-only copies around the unchanged attention kernels; allocations precede capture.
use crate::constants::KV_BLOCK_TOKENS;
use crate::constants::METADATA_FIELDS;
use crate::device::device_error;
use crate::resident::buffers::buffer;
use crate::{device::CudaDevice, resident::ProgramWeights, tuning::attention::Choice};
use cutile::prelude::*;
use infer_core::{Result, TensorId};
use infer_cuda_kernels::contract::{Buffer, Recording};
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
    pub query: Tensor<f32>,
    pub keys: Tensor<f32>,
    pub values: Tensor<f32>,
    pub output: Tensor<f32>,
    pub metadata: Tensor<i32>,
    pub table: Tensor<i32>,
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
            let allocate = |size| {
                api::zeros::<f32>(&[size])
                    .sync_on(&device.stream)
                    .map_err(device_error)
            };
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
                metadata: api::zeros::<i32>(&[METADATA_FIELDS])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
                table: api::zeros::<i32>(&[capacity / KV_BLOCK_TOKENS])
                    .sync_on(&device.stream)
                    .map_err(device_error)?,
            };
            self.buffers.insert((node.outputs[0], lanes), entry);
        }
        Ok(())
    }
    pub fn inputs(
        &mut self,
        scope: &Scope,
        key: (TensorId, usize),
        input: [&Tensor<f32>; 3],
        (metadata, table): (Buffer<'_, i32>, Buffer<'_, i32>),
    ) -> StdResult<(), DeviceError> {
        if let Some(tap) = self.buffers.get_mut(&key) {
            for (out, input) in [
                (&mut tap.query, input[0]),
                (&mut tap.keys, input[1]),
                (&mut tap.values, input[2]),
            ] {
                scope.record(api::memcpy(out, input))?;
            }
            let recording = Recording::scope(scope);
            let copy = |tap: &Tensor<i32>, source| {
                recording
                    .copy(buffer(tap)?, source)
                    .map_err(|error| DeviceError::Internal(error.to_string()))
            };
            copy(&tap.metadata, metadata)?;
            copy(&tap.table, table)?;
        }
        Ok(())
    }
    pub fn output(
        &self,
        scope: &Scope,
        key: (TensorId, usize),
        output: Buffer<'_, f32>,
    ) -> StdResult<(), DeviceError> {
        if let Some(tap) = self.buffers.get(&key) {
            Recording::scope(scope)
                .copy(buffer(&tap.output)?, output)
                .map_err(|error| DeviceError::Internal(error.to_string()))?;
        }
        Ok(())
    }
}
