use super::linux;
use super::mask::Mask;
use crate::constants::MAX_TOPOLOGY_CPUS;
use crate::{Error, Result};
use std::io::Error as IoError;

/// Bits in the legacy `cpu_set_t` bitmap that seeds the affinity probe.
const CPU_SETSIZE_BITS: usize = 1024;

pub(super) fn current() -> Result<Mask> {
    let mut bits = CPU_SETSIZE_BITS;
    loop {
        let mut mask = Mask::new(bits)?;
        // SAFETY: The unsigned-long storage is suitably aligned and holds exactly bytes() writable bytes.
        // Linux accepts a variable length bitmap, not only libc's fixed cpu_set_t representation.
        if unsafe { libc::sched_getaffinity(0, mask.bytes(), mask.pointer_mut().cast()) } == 0 {
            return Ok(mask);
        }
        let error = IoError::last_os_error();
        if error.raw_os_error() != Some(libc::EINVAL) || bits >= MAX_TOPOLOGY_CPUS {
            return Err(Error::invalid(format!("read CPU affinity: {error}")));
        }
        bits *= 2;
    }
}
pub(super) fn set(mask: &Mask) -> Result<()> {
    // SAFETY: pid zero is this thread; pointer is an aligned bitmap of the supplied byte size.
    if unsafe { libc::sched_setaffinity(0, mask.bytes(), mask.pointer().cast()) } == 0 {
        Ok(())
    } else {
        Err(linux::os_error("set CPU affinity"))
    }
}
pub(in crate::placement) fn allowed() -> Result<Vec<usize>> {
    let mask = current()?;
    Ok((0..mask.bits()).filter(|cpu| mask.contains(*cpu)).collect())
}
