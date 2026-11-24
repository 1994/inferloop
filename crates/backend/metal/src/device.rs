//! The only raw shared-memory boundary; model/runtime code remains safe Rust.
#![allow(
    unsafe_code,
    reason = "Metal FFI and shared GPU buffers are confined to this audited device boundary"
)]
// objc 0.2's selector macro predates Cargo's checked feature names.
#![allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macros emit a legacy cargo-clippy feature check"
)]
use crate::constants::F32_BYTES;
use crate::constants::F32_BYTES_U64;
use crate::constants::INPUT_BINDING_COUNT;
use crate::constants::MAX_DECODE_THREADGROUP;
use crate::constants::NANOS_PER_SECOND;
use crate::constants::OUTPUT_BINDING;
use crate::constants::PAGE_TABLE_BINDING;
use crate::constants::PARAMS_BINDING;
use crate::constants::STATE_BINDING;
use crate::constants::TILED_PREFILL_SIMD_WIDTH;
use crate::constants::TILED_PREFILL_THREADGROUP;
use crate::constants::TILED_PREFILL_TILE_COLUMNS;
use crate::constants::TILED_PREFILL_TILE_ROWS;
use crate::constants::TOKEN_BINDING;
use infer_core::{Error, ErrorCode, Result};
use metal::{
    Buffer, BufferRef, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState,
    Device, MTLResourceOptions, MTLSize, objc,
};
use std::collections::BTreeMap;
use std::ptr::copy_nonoverlapping;
use std::ptr::write_bytes;
use std::slice::from_raw_parts;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Params {
    pub n: u32,
    pub token: u32,
    pub position: u32,
    pub capacity: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub e: u32,
    pub f: u32,
    pub g: u32,
    pub h: u32,
    pub epsilon: f32,
    pub offset: f32,
    pub theta: f32,
    pub extra: f32,
    pub rows: u32,
    pub start_row: u32,
    pub weight_formats: u32,
    pub reserved: u32,
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "Metal timestamps are seconds as f64; nonnegative nanoseconds intentionally truncate sub-nanosecond fractions and saturate at u64 limits"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "Metal timestamps are seconds as f64; nonnegative nanoseconds intentionally truncate sub-nanosecond fractions and saturate at u64 limits"
)]
pub fn command_timing(command: &CommandBufferRef) -> (u64, u64) {
    use objc::{msg_send, sel, sel_impl};
    // SAFETY: MTLCommandBuffer implements these documented timing selectors;
    // this function is called only after status is Completed.
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    // SAFETY: Same retained completed command and documented f64 return type.
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    (
        (start * NANOS_PER_SECOND).max(0.0) as u64,
        ((end - start) * NANOS_PER_SECOND).max(0.0) as u64,
    )
}
#[derive(Clone)]
pub struct MetalDevice {
    pub device: Device,
    pub queue: CommandQueue,
    pipelines: BTreeMap<&'static str, ComputePipelineState>,
}
pub struct Bindings<'a> {
    pub inputs: &'a [&'a BufferRef],
    pub state: &'a BufferRef,
    pub output: &'a BufferRef,
    pub dummy: &'a BufferRef,
    pub page_table: &'a BufferRef,
    pub tokens: &'a BufferRef,
}
impl MetalDevice {
    pub fn allocate_bytes(&self, bytes: u64) -> Result<Buffer> {
        if bytes == 0 || bytes > self.device.max_buffer_length() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "Metal allocation exceeds device limit",
            ));
        }
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        if buffer.contents().is_null() {
            return Err(Error::new(ErrorCode::Capacity, "Metal allocation failed"));
        }
        Ok(buffer)
    }

    pub fn write_bytes_idle(buffer: &BufferRef, offset: u64, input: &[u8]) -> Result<()> {
        if offset
            .checked_add(input.len() as u64)
            .is_none_or(|n| n > buffer.length())
        {
            return Err(Error::invalid("weight upload bounds"));
        }
        let offset = usize::try_from(offset)
            .map_err(|_| Error::invalid("weight offset exceeds address space"))?;
        // SAFETY: Only the loader writes this newly allocated, unpublished shared buffer.
        // The retained allocation and checked destination range cannot overlap the input slice.
        let destination = unsafe { buffer.contents().cast::<u8>().add(offset) };
        // SAFETY: The source slice and checked, retained idle allocation have disjoint ranges.
        unsafe {
            copy_nonoverlapping(input.as_ptr(), destination, input.len());
        }
        Ok(())
    }
    pub fn open() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| Error::unsupported("no Metal GPU found"))?;
        if !device.has_unified_memory() {
            return Err(Error::unsupported(
                "current Metal shared-buffer executor requires unified memory",
            ));
        }
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(include_str!("kernels.metal"), &options)
            .map_err(|e| Error::new(ErrorCode::Backend, format!("Metal shader compile: {e}")))?;
        let mut pipelines = BTreeMap::new();
        for name in [
            "embedding",
            "linear",
            "linear_prefill",
            "norm",
            "split",
            "rope",
            "kv_append",
            "attention",
            "conv",
            "delta",
            "gated_norm",
            "unary",
            "binary",
        ] {
            let function = library
                .get_function(name, None)
                .map_err(|e| Error::new(ErrorCode::Backend, e))?;
            pipelines.insert(
                name,
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| Error::new(ErrorCode::Backend, e))?,
            );
        }
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }
    pub fn upload(&self, data: &[f32]) -> Result<Buffer> {
        let bytes = data
            .len()
            .checked_mul(F32_BYTES)
            .ok_or_else(|| Error::invalid("Metal buffer size overflow"))?;
        if bytes == 0 || bytes as u64 > self.device.max_buffer_length() {
            return Err(Error::new(
                ErrorCode::Capacity,
                "Metal buffer size exceeds device limit",
            ));
        }
        Ok(self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }
    pub fn upload_indices(&self, data: &[u32]) -> Result<Buffer> {
        let bytes = data
            .len()
            .checked_mul(F32_BYTES)
            .ok_or_else(|| Error::invalid("page table overflow"))?;
        if bytes == 0 || bytes as u64 > self.device.max_buffer_length() {
            return Err(Error::invalid("page table bounds"));
        }
        Ok(self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }
    pub fn write_idle(buffer: &BufferRef, offset: usize, values: &[f32]) -> Result<()> {
        if offset
            .checked_add(values.len())
            .and_then(|n| n.checked_mul(F32_BYTES))
            .is_none_or(|n| n as u64 > buffer.length())
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(Error::invalid("device page write bounds/numerics"));
        }
        // SAFETY: The checkpoint loader holds the completion gate, the shared
        // allocation is retained and bounds/alignment are checked above.
        let destination = unsafe { buffer.contents().cast::<f32>().add(offset) };
        // SAFETY: The retained, idle shared buffer and input slice do not overlap;
        // their complete ranges were checked above.
        unsafe {
            copy_nonoverlapping(values.as_ptr(), destination, values.len());
        }
        Ok(())
    }
    pub fn write_indices_idle(buffer: &BufferRef, values: &[u32]) -> Result<()> {
        Self::write_indices_range_idle(buffer, 0, values)
    }
    pub fn write_indices_range_idle(
        buffer: &BufferRef,
        offset: usize,
        values: &[u32],
    ) -> Result<()> {
        if values
            .len()
            .checked_add(offset)
            .and_then(|n| n.checked_mul(F32_BYTES))
            .is_none_or(|n| n as u64 > buffer.length())
        {
            return Err(Error::invalid("page table write bounds"));
        }
        // SAFETY: The submission owner holds the completion gate; page tables
        // are retained, u32-aligned shared allocations and the bounds are checked.
        unsafe {
            copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents().cast::<u32>().wrapping_add(offset),
                values.len(),
            );
        }
        Ok(())
    }
    /// Caller has checked all execution and transfer fences for this private shared buffer.
    pub fn clear_idle(buffer: &BufferRef) -> Result<()> {
        let length = usize::try_from(buffer.length())
            .map_err(|_| Error::invalid("Metal clear exceeds address space"))?;
        // SAFETY: Retained shared storage has no device readers or writers under the owner's fence.
        unsafe {
            write_bytes(buffer.contents().cast::<u8>(), 0, length);
        }
        Ok(())
    }
    pub fn zeros(&self, n: usize) -> Result<Buffer> {
        if n.checked_mul(F32_BYTES)
            .is_none_or(|bytes| bytes as u64 > self.device.max_buffer_length())
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "Metal buffer exceeds device limit",
            ));
        }
        let buffer = self.allocate_bytes(n.max(1) as u64 * F32_BYTES_U64)?;
        let length = usize::try_from(buffer.length())
            .map_err(|_| Error::invalid("allocation exceeds address space"))?;
        // SAFETY: This unpublished shared allocation is idle, retained, and valid for its full byte length.
        unsafe {
            write_bytes(buffer.contents().cast::<u8>(), 0, length);
        }
        Ok(buffer)
    }
    pub fn simd_width(&self) -> u32 {
        u32::try_from(self.pipelines["linear"].thread_execution_width()).unwrap_or(u32::MAX)
    }
    /// Read into startup/admission-reserved host storage, after its GPU fence.
    pub fn read_into_idle(buffer: &BufferRef, offset: usize, output: &mut [f32]) -> Result<()> {
        if offset
            .checked_add(output.len())
            .and_then(|n| n.checked_mul(F32_BYTES))
            .is_none_or(|bytes| bytes as u64 > buffer.length())
        {
            return Err(Error::invalid("Metal readback bounds"));
        }
        // SAFETY: Shared Metal storage is retained, F32-aligned and idle under the executor's fence.
        let values = unsafe {
            from_raw_parts(
                buffer.contents().cast::<f32>().wrapping_add(offset),
                output.len(),
            )
        };
        output.copy_from_slice(values);
        Ok(())
    }
    /// Caller holds the executor's completion gate: no GPU writer may be active.
    pub fn read_idle(buffer: &BufferRef, n: usize) -> Result<Vec<f32>> {
        if n.checked_mul(F32_BYTES)
            .is_none_or(|bytes| bytes as u64 > buffer.length())
        {
            return Err(Error::invalid("Metal readback bounds"));
        }
        // SAFETY: Shared Metal allocation is retained, aligned for f32, and the
        // executor calls this only after command completion (or for immutable weights).
        let values = unsafe { from_raw_parts(buffer.contents().cast::<f32>(), n) }.to_vec();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(ErrorCode::Backend, "non-finite Metal tensor"));
        }
        Ok(values)
    }
    pub fn encode(
        &self,
        command: &CommandBufferRef,
        name: &str,
        bindings: &Bindings<'_>,
        params: Params,
        threads: usize,
    ) -> Result<()> {
        let tiled = name == "linear"
            && params.rows > 1
            && self.pipelines["linear_prefill"].thread_execution_width()
                == TILED_PREFILL_SIMD_WIDTH
            && self.pipelines["linear_prefill"].max_total_threads_per_threadgroup()
                >= TILED_PREFILL_THREADGROUP;
        let pipeline = &self.pipelines[if tiled { "linear_prefill" } else { name }];
        let threads = if name == "linear" && !tiled {
            threads
                .checked_mul(
                    usize::try_from(pipeline.thread_execution_width())
                        .map_err(|_| Error::invalid("SIMD width exceeds address space"))?,
                )
                .ok_or_else(|| Error::invalid("linear dispatch shape overflow"))?
        } else {
            threads
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for index in 0..INPUT_BINDING_COUNT {
            encoder.set_buffer(
                index as u64,
                Some(
                    bindings
                        .inputs
                        .get(index)
                        .copied()
                        .unwrap_or(bindings.dummy),
                ),
                0,
            );
        }
        encoder.set_buffer(STATE_BINDING, Some(bindings.state), 0);
        encoder.set_buffer(OUTPUT_BINDING, Some(bindings.output), 0);
        encoder.set_buffer(PAGE_TABLE_BINDING, Some(bindings.page_table), 0);
        encoder.set_buffer(TOKEN_BINDING, Some(bindings.tokens), 0);
        encoder.set_bytes(
            PARAMS_BINDING,
            size_of::<Params>() as u64,
            (&raw const params).cast(),
        );
        if tiled {
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    u64::from(params.n).div_ceil(TILED_PREFILL_TILE_COLUMNS),
                    u64::from(params.rows).div_ceil(TILED_PREFILL_TILE_ROWS),
                    1,
                ),
                MTLSize::new(TILED_PREFILL_THREADGROUP, 1, 1),
            );
        } else {
            encoder.dispatch_threads(
                MTLSize::new(threads as u64, 1, 1),
                MTLSize::new(
                    (threads as u64)
                        .min(pipeline.max_total_threads_per_threadgroup())
                        .clamp(1, MAX_DECODE_THREADGROUP),
                    1,
                    1,
                ),
            );
        }
        encoder.end_encoding();
        Ok(())
    }
    pub fn copy(
        command: &CommandBufferRef,
        source: &BufferRef,
        destination: &BufferRef,
        offset: usize,
        count: usize,
    ) -> Result<()> {
        Self::copy_range(command, source, 0, destination, offset, count)
    }
    pub fn copy_range(
        command: &CommandBufferRef,
        source: &BufferRef,
        source_offset: usize,
        destination: &BufferRef,
        offset: usize,
        count: usize,
    ) -> Result<()> {
        let bytes = count
            .checked_mul(F32_BYTES)
            .ok_or_else(|| Error::invalid("Metal copy overflow"))? as u64;
        let offset = offset
            .checked_mul(F32_BYTES)
            .ok_or_else(|| Error::invalid("Metal copy overflow"))? as u64;
        let source_offset = (source_offset as u64)
            .checked_mul(F32_BYTES_U64)
            .ok_or_else(|| Error::invalid("copy source overflow"))?;
        if source_offset
            .checked_add(bytes)
            .is_none_or(|n| n > source.length())
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > destination.length())
        {
            return Err(Error::invalid("Metal copy bounds"));
        }
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(source, source_offset, destination, offset, bytes);
        blit.end_encoding();
        Ok(())
    }
    pub fn synchronize(&self) {
        let command = self.queue.new_command_buffer();
        command.commit();
        command.wait_until_completed();
    }
}

#[cfg(test)]
mod tests;
