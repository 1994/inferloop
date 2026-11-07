//! Host values staged into device memory.
//!
//! The cuTile build stages uploads through a ring of pinned slots so it never waits on the
//! stream. Uploads here complete before they return, so the ring only bounds their size.
use super::{DeviceCopy, Element, Transfers};
use crate::contract::buffer::Buffer;
use infer_core::{Error, Result};
use std::marker::PhantomData;

/// Uploads of up to `elements` values each.
pub struct Staging<T: DeviceCopy> {
    elements: usize,
    element: PhantomData<T>,
}

impl<T: DeviceCopy> Staging<T> {
    pub(super) fn new(elements: usize, depth: usize) -> Result<Self> {
        if depth == 0 {
            return Err(Error::invalid("staging needs at least one slot"));
        }
        Ok(Self {
            elements,
            element: PhantomData,
        })
    }
}

impl Transfers<'_> {
    /// Copy `values` into `dst`, after the work already on the stream.
    /// # Errors
    /// Rejects `values` longer than a slot or of another length than `dst`, and a copy the
    /// driver does not run.
    pub fn upload<T: DeviceCopy + Element>(
        &self,
        staging: &mut Staging<T>,
        dst: Buffer<'_, T::Marker>,
        values: &[T],
    ) -> Result<()> {
        if values.len() > staging.elements || values.len() != dst.len() {
            return Err(Error::invalid("staged upload extent"));
        }
        self.device.write(dst, values)
    }
}
