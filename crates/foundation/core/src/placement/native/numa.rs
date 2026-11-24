use super::linux;
use super::mask::Mask;
use crate::placement::topology::allowed_nodes;
use crate::{
    Error, Result,
    placement::{NumaPolicy, parse_cpu_list},
};
use std::fs::read_to_string;
use std::ptr::null;
pub(super) struct Policy {
    mode: i32,
    mask: Mask,
}
impl Policy {
    pub fn current() -> Result<Self> {
        let possible = read_to_string("/sys/devices/system/node/possible")
            .map_err(|error| Error::invalid(format!("read possible NUMA nodes: {error}")))?;
        let nodes = parse_cpu_list(&possible)?;
        let bits = nodes
            .last()
            .ok_or_else(|| Error::invalid("no possible NUMA node"))?
            + 1;
        let mut mask = Mask::new(bits)?;
        let mut mode = 0i32;
        // SAFETY: mode is a writable int and mask spans bits() aligned bits; null address/zero flags queries this thread.
        if unsafe {
            libc::syscall(
                libc::SYS_get_mempolicy,
                &raw mut mode,
                mask.pointer_mut(),
                // Linux's nodemask ABI subtracts one from maxnode before copying bits.
                // Include the highest bit of an exactly word-sized bitmap without requesting an extra word.
                mask.bits() + 1,
                null::<u8>(),
                0usize,
            )
        } != 0
        {
            return Err(linux::os_error("read NUMA policy"));
        }
        Ok(Self { mode, mask })
    }
    pub fn selected(&self, policy: NumaPolicy) -> Result<Self> {
        let (mode, node) = match policy {
            NumaPolicy::Bind(node) => (2, node),
            NumaPolicy::Prefer(node) => (1, node),
        };
        if !allowed_nodes()?.contains(&node) {
            return Err(Error::invalid("NUMA node outside allowed memory set"));
        }
        let mut mask = Mask::new(self.mask.bits())?;
        mask.insert(node)?;
        Ok(Self { mode, mask })
    }
    pub fn apply(&self) -> Result<()> {
        let pointer = if self.mode == 0 {
            null()
        } else {
            self.mask.pointer()
        };
        // SAFETY: set_mempolicy reads the supplied aligned bitmap; MPOL_DEFAULT requires a null pointer.
        if unsafe {
            libc::syscall(
                libc::SYS_set_mempolicy,
                self.mode,
                pointer,
                self.mask.bits() + 1,
            )
        } == 0
        {
            Ok(())
        } else {
            Err(linux::os_error("set NUMA memory policy"))
        }
    }
}
#[cfg(test)]
mod memory_tests;
#[cfg(test)]
mod tests;
