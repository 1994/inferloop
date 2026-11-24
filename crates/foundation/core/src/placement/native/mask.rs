//! Dynamically sized Linux bitmaps; never truncate a large host to `cpu_set_t`'s 1024 bits.
use crate::constants::MAX_TOPOLOGY_CPUS;
use crate::{Error, Result};
use libc::c_ulong;
#[derive(Clone)]
pub(super) struct Mask(Vec<c_ulong>);
impl Mask {
    pub fn new(bits: usize) -> Result<Self> {
        if bits == 0 || bits > MAX_TOPOLOGY_CPUS {
            return Err(Error::invalid("OS bitmap size outside topology limits"));
        }
        Ok(Self(vec![0; bits.div_ceil(c_ulong::BITS as usize)]))
    }
    pub const fn bits(&self) -> usize {
        self.0.len() * c_ulong::BITS as usize
    }
    pub const fn bytes(&self) -> usize {
        self.0.len() * size_of::<c_ulong>()
    }
    pub const fn pointer(&self) -> *const c_ulong {
        self.0.as_ptr()
    }
    pub const fn pointer_mut(&mut self) -> *mut c_ulong {
        self.0.as_mut_ptr()
    }
    pub fn contains(&self, bit: usize) -> bool {
        let width = c_ulong::BITS as usize;
        self.0
            .get(bit / width)
            .is_some_and(|word| word & (1 << (bit % width)) != 0)
    }
    pub fn insert(&mut self, bit: usize) -> Result<()> {
        let width = c_ulong::BITS as usize;
        let word = self
            .0
            .get_mut(bit / width)
            .ok_or_else(|| Error::invalid("index outside OS bitmap"))?;
        *word |= 1 << (bit % width);
        Ok(())
    }
}
#[cfg(test)]
mod tests;
