//! A launch of a loaded PTX kernel over contract buffers, recorded as a [`Foreign`] call.
//!
//! Hand-written kernels outside the cuTile families (the SIMT attention sources, the FA4 bridges)
//! launch through this, so their callers pass buffers rather than cuTile tensors and the graph
//! keeps their memory alive through each buffer's owner.
use crate::contract::buffer::{Buffer, Owner};
use crate::contract::recording::Foreign;
use cuda_core::{Function, launch_kernel};
use infer_core::{Error, ErrorCode, Result};
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

/// One by-value kernel parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Argument {
    /// A device address, passed as a 64-bit pointer.
    Address(u64),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl Argument {
    /// The address of `buffer`, whose constructor already guarantees the memory a launch needs.
    #[must_use]
    pub const fn buffer<T>(buffer: Buffer<'_, T>) -> Self {
        Self::Address(buffer.address())
    }
}

/// One launch of `function` on the stream it is pinned to.
pub struct KernelLaunch {
    function: Arc<Function>,
    stream: usize,
    grid: u32,
    threads: u32,
    arguments: Vec<Argument>,
}

impl KernelLaunch {
    /// A one-dimensional launch of `grid` blocks of `threads`, valid only on `stream`.
    ///
    /// # Safety
    /// `arguments` must match the kernel's parameters in number, order and type. Each
    /// [`Argument::Address`] must address memory the kernel may access as its signature implies,
    /// valid until every replay of the recording graph has finished; [`Argument::buffer`] of a
    /// buffer the graph keeps alive satisfies this.
    #[expect(
        unsafe_code,
        reason = "The constructor states the signature contract, so enqueueing can be safe"
    )]
    #[must_use]
    pub const unsafe fn new(
        function: Arc<Function>,
        stream: usize,
        grid: u32,
        threads: u32,
        arguments: Vec<Argument>,
    ) -> Self {
        Self {
            function,
            stream,
            grid,
            threads,
            arguments,
        }
    }

    /// What a graph recording this launch must hold: the module behind its function.
    #[must_use]
    pub fn owner(&self) -> Owner {
        Arc::clone(&self.function) as Owner
    }

    /// Enqueue the launch on `stream`, which must be the stream it was made for.
    /// # Errors
    /// Rejects another stream, or returns the driver's refusal.
    #[expect(
        unsafe_code,
        reason = "Each parameter pointer addresses a live argument of this call, whose signature `new` made the caller vouch for"
    )]
    pub fn enqueue_on(mut self, stream: *mut c_void) -> Result<()> {
        if stream as usize != self.stream {
            return Err(Error::invalid("kernel launch stream mismatch"));
        }
        let mut parameters: Vec<*mut c_void> = self
            .arguments
            .iter_mut()
            .map(|argument| match argument {
                Argument::Address(value) => ptr::from_mut(value).cast(),
                Argument::U32(value) => ptr::from_mut(value).cast(),
                Argument::I32(value) => ptr::from_mut(value).cast(),
                Argument::F32(value) => ptr::from_mut(value).cast(),
            })
            .collect();
        // SAFETY: `self.function` keeps the module loaded for as long as the handle is used.
        let function = unsafe { self.function.cu_function() };
        // SAFETY: `parameters` points into `self.arguments`, alive for the call, and `new`'s
        // contract covers their types and addresses.
        unsafe {
            launch_kernel(
                function,
                (self.grid, 1, 1),
                (self.threads, 1, 1),
                0,
                stream.cast(),
                &mut parameters,
            )
        }
        .map_err(|error| Error::new(ErrorCode::Backend, format!("kernel launch: {error:?}")))
    }
}

impl Foreign for KernelLaunch {
    fn enqueue(self, stream: *mut c_void) -> Result<()> {
        self.enqueue_on(stream)
    }
}
