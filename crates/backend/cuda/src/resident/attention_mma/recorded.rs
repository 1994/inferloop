//! One captured MMA attention: conversion, AOT attention and readout, in order on one stream.
use super::host;
use cuda_core::sys::CUdeviceptr;
use infer_core::Result;
use infer_cuda_kernels::contract::Foreign;
use infer_cuda_kernels::tile::runtime::KernelLaunch;
use std::ffi::c_void;
use std::sync::Arc;

pub(super) struct Recorded {
    pub prepare: KernelLaunch,
    pub finish: KernelLaunch,
    pub api: Arc<host::Api>,
    pub pointers: [CUdeviceptr; super::catalog::OPERANDS],
    pub lanes: usize,
    pub capacity: usize,
}

impl Foreign for Recorded {
    fn enqueue(self, stream: *mut c_void) -> Result<()> {
        // The bridge launch rejects any stream but the program's, before anything is enqueued.
        self.prepare.enqueue_on(stream)?;
        self.api
            .launch(self.pointers, self.lanes, self.capacity, stream.cast())?;
        self.finish.enqueue_on(stream)
    }
}
