//! Linux hardware acceptance: fresh mappings bypass the allocator's previously touched page caches.
use crate::{
    Error, Result,
    placement::{CpuTopology, NumaPolicy, ThreadPlacement},
};
use std::ptr::null;
use std::ptr::null_mut;
struct Pages {
    pointer: *mut libc::c_void,
    bytes: usize,
}
impl Pages {
    fn new(bytes: usize) -> Result<Self> {
        // SAFETY: anonymous private mapping, no file descriptor; returned address is checked before use.
        let pointer = unsafe {
            libc::mmap(
                null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(super::super::linux::os_error("map first-touch pages"));
        }
        Ok(Self { pointer, bytes })
    }
    fn touch_and_query(&self, node: usize) -> Result<()> {
        // SAFETY: sysconf has no pointer arguments and queries the OS page size.
        let page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .map_err(|_| Error::invalid("invalid OS page size"))?;
        if page_size == 0 {
            return Err(Error::invalid("zero OS page size"));
        }
        let pointers: Vec<_> = (0..self.bytes)
            .step_by(page_size)
            .map(|offset| {
                let pointer = self.pointer.cast::<u8>().wrapping_add(offset);
                // SAFETY: offsets are within this writable live mapping; volatile ensures each page is faulted.
                unsafe {
                    pointer.write_volatile(1);
                }
                pointer.cast::<libc::c_void>()
            })
            .collect();
        let mut status = vec![-1i32; pointers.len()];
        // SAFETY: move_pages queries our process with equally sized pointer/status arrays; null nodes performs no migration.
        if unsafe {
            libc::syscall(
                libc::SYS_move_pages,
                0i32,
                pointers.len(),
                pointers.as_ptr(),
                null::<i32>(),
                status.as_mut_ptr(),
                0i32,
            )
        } < 0
        {
            return Err(super::super::linux::os_error(
                "query first-touch NUMA pages",
            ));
        }
        let expected = i32::try_from(node).map_err(|_| Error::invalid("NUMA node overflow"))?;
        if status.iter().any(|actual| *actual != expected) {
            return Err(Error::invariant(format!(
                "NUMA first-touch mismatch: expected {node}, observed {status:?}"
            )));
        }
        Ok(())
    }
}
impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: this object owns exactly the live mapping returned by mmap; it is unmapped once.
        unsafe {
            libc::munmap(self.pointer, self.bytes);
        }
    }
}
#[test]
#[ignore = "requires Linux NUMA policy and move_pages query permission; make check-linux-numa"]
fn bind_first_touch_is_verified_on_physical_pages() -> Result<()> {
    let node = *CpuTopology::discover()?
        .allowed_nodes
        .first()
        .ok_or_else(|| Error::invalid("no allowed NUMA node"))?;
    ThreadPlacement {
        numa: Some(NumaPolicy::Bind(node)),
        ..ThreadPlacement::default()
    }
    .scope(|| Pages::new(2 * 1024 * 1024)?.touch_and_query(node))
}
