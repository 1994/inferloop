//! Device memory as a kernel source sees it.
//!
//! A buffer is an address, an element count and a type, with no handle to any toolkit. Family
//! methods take buffers and a plan, so the host names no cuTile type and no unsafe call. The
//! obligations a captured graph puts on its memory are discharged once, where a buffer is made.
#![expect(
    unsafe_code,
    reason = "The constructor states what every later launch relies on, so launches can be safe"
)]
use infer_core::{Error, Result};
use std::any::Any;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

/// Keeps device memory alive; a graph holds one for each allocation it recorded.
pub type Owner = Arc<dyn Any + Send + Sync>;

/// BF16 element marker, so the contract does not depend on a numeric crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16;

/// FP4 code pair element marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp4Pair;

/// FP8 element marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8;

/// Bytes one element of a buffer occupies, so a buffer can be cut without knowing its toolkit.
pub trait Width {
    /// Bytes of one element.
    const BYTES: usize;
}

impl Width for f32 {
    const BYTES: usize = 4;
}
impl Width for i32 {
    const BYTES: usize = 4;
}
impl Width for u32 {
    const BYTES: usize = 4;
}
impl Width for Bf16 {
    const BYTES: usize = 2;
}
impl Width for Fp8 {
    const BYTES: usize = 1;
}
impl Width for Fp4Pair {
    const BYTES: usize = 1;
}

/// Contiguous device memory the host owns, holding `len` elements of `T`.
#[derive(Debug)]
pub struct Buffer<'a, T> {
    address: u64,
    len: usize,
    owner: Option<&'a Owner>,
    element: PhantomData<&'a T>,
}

impl<T> Clone for Buffer<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Buffer<'_, T> {}

impl<'a, T> Buffer<'a, T> {
    /// # Safety
    /// `address` must address `len` contiguous elements of `T` on the recording's device, valid
    /// until every replay of any graph recording this buffer has finished. A kernel writes through
    /// its output buffers, so those must not alias another buffer of the same launch.
    #[must_use]
    pub const unsafe fn new(address: u64, len: usize) -> Self {
        Self {
            address,
            len,
            owner: None,
            element: PhantomData,
        }
    }

    /// A buffer whose memory a recording keeps alive by retaining `owner`.
    ///
    /// # Safety
    /// As [`Self::new`], except that `owner` must hold the memory, which discharges the lifetime
    /// obligation for any recording that retains it.
    #[must_use]
    pub const unsafe fn owned(address: u64, len: usize, owner: &'a Owner) -> Self {
        Self {
            address,
            len,
            owner: Some(owner),
            element: PhantomData,
        }
    }

    /// The same memory under another element type.
    ///
    /// # Safety
    /// `U` must have the size and layout of `T`.
    #[must_use]
    pub const unsafe fn retyped<U>(self) -> Buffer<'a, U> {
        Buffer {
            address: self.address,
            len: self.len,
            owner: self.owner,
            element: PhantomData,
        }
    }

    /// What keeps this memory alive, when the buffer knows.
    #[must_use]
    pub const fn owner(self) -> Option<&'a Owner> {
        self.owner
    }

    #[must_use]
    pub const fn address(self) -> u64 {
        self.address
    }

    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Check this buffer against the elements a plan expects of it.
    /// # Errors
    /// Names the operand whose length disagrees with the plan.
    pub fn require(self, name: &str, elements: usize) -> Result<Self> {
        if self.len == elements {
            Ok(self)
        } else {
            Err(Error::invalid(format!(
                "{name}: {} elements, plan needs {elements}",
                self.len
            )))
        }
    }
}

impl<T: Width> Buffer<'_, T> {
    /// Elements `range` of this memory, kept alive by the same owner.
    /// # Errors
    /// Rejects a range that is reversed or runs past the end.
    pub fn slice(self, range: Range<usize>) -> Result<Self> {
        let len = range
            .end
            .checked_sub(range.start)
            .filter(|_| range.end <= self.len)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "slice {range:?} of a buffer of {} elements",
                    self.len
                ))
            })?;
        let offset = range
            .start
            .checked_mul(T::BYTES)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::invalid("slice offset overflows"))?;
        Ok(Self {
            address: self.address + offset,
            len,
            owner: self.owner,
            element: PhantomData,
        })
    }
}
