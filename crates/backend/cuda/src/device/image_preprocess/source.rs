//! Device input operations retained through their submission fence.
use super::{nvjpeg::Lease, staging::Staged};
use infer_core::Result;
use infer_cuda_kernels::runtime::{Allocation, Operation};
use std::sync::Arc;

pub(super) enum Source {
    Rgb(Arc<Staged<u8>>),
    Jpeg {
        lease: Arc<Lease>,
        bytes: Arc<Vec<u8>>,
        width: usize,
    },
}

impl Source {
    pub fn retain(&self, operation: &Operation<'_>) -> Result<()> {
        match self {
            Self::Rgb(staged) => operation.retain(Arc::clone(staged)),
            Self::Jpeg { lease, bytes, .. } => {
                operation.retain(Arc::clone(lease))?;
                operation.retain(Arc::clone(bytes))
            }
        }
    }

    /// Write the source pixels into `output`; [`Self::retain`] must already have run.
    #[expect(
        unsafe_code,
        reason = "The staged pixels are retained by the operation and refilled only after it completes"
    )]
    pub fn enqueue(&self, operation: &Operation<'_>, output: &Allocation<u8>) -> Result<()> {
        match self {
            // SAFETY: `retain` keeps the staged buffer out of its pool, the only writer, until the
            // operation completes.
            Self::Rgb(staged) => unsafe {
                operation.upload(output, staged.buffer()?, staged.len())
            },
            Self::Jpeg {
                lease,
                bytes,
                width,
            } => lease.decode(
                Arc::clone(bytes),
                output.buffer().address(),
                *width,
                operation.stream(),
            ),
        }
    }
}
