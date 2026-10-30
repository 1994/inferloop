//! Fixed leases keep host credits charged until the last request, worker, or output reader ends.
use crate::{Error, ErrorCode, Result};
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::process::abort;
use std::{
    sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::AtomicU32,
    sync::atomic::AtomicUsize, sync::atomic::Ordering,
};

type Wake = Arc<dyn Fn() + Send + Sync>;
struct Budget {
    limit: usize,
    used: AtomicUsize,
    wake: Mutex<Option<Wake>>,
}
struct Slot {
    budget: Arc<Budget>,
    leased: AtomicBool,
    retired: AtomicBool,
    readers: AtomicUsize,
    units: AtomicUsize,
    generation: AtomicU32,
}
pub struct CreditPool {
    budget: Arc<Budget>,
    slots: Vec<Arc<Slot>>,
}
pub struct CreditLease {
    slot: Arc<Slot>,
    generation: u32,
}
impl CreditPool {
    /// # Errors
    /// Rejects zero capacities or failure to allocate fixed lease slots.
    pub fn new(slots: usize, limit: usize) -> Result<Self> {
        if slots == 0 || limit == 0 {
            return Err(Error::invalid("invalid host credit capacity"));
        }
        let budget = Arc::new(Budget {
            limit,
            used: AtomicUsize::new(0),
            wake: Mutex::new(None),
        });
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(slots)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        for _ in 0..slots {
            storage.push(Arc::new(Slot {
                budget: budget.clone(),
                leased: AtomicBool::new(false),
                retired: AtomicBool::new(false),
                readers: AtomicUsize::new(0),
                units: AtomicUsize::new(0),
                generation: AtomicU32::new(1),
            }));
        }
        Ok(Self {
            budget,
            slots: storage,
        })
    }
    #[must_use]
    pub fn used(&self) -> usize {
        self.budget.used.load(Ordering::Acquire)
    }
    pub fn set_waker(&self, wake: Wake) {
        if let Ok(mut slot) = self.budget.wake.lock() {
            *slot = Some(wake);
        }
    }
    /// # Errors
    /// Rejects exhausted units/reader slots. It never allocates a new lease.
    pub fn reserve(&self, units: usize) -> Result<CreditLease> {
        let mut used = self.budget.used.load(Ordering::Acquire);
        loop {
            let total = used
                .checked_add(units)
                .filter(|total| *total <= self.budget.limit)
                .ok_or_else(|| Error::new(ErrorCode::Capacity, "host storage credits exhausted"))?;
            match self.budget.used.compare_exchange_weak(
                used,
                total,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => used = actual,
            }
        }
        for slot in &self.slots {
            if slot.retired.load(Ordering::Acquire)
                || slot
                    .leased
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
            {
                continue;
            }
            slot.units.store(units, Ordering::Relaxed);
            slot.readers.store(1, Ordering::Relaxed);
            return Ok(CreditLease {
                slot: slot.clone(),
                generation: slot.generation.load(Ordering::Relaxed),
            });
        }
        self.budget.used.fetch_sub(units, Ordering::AcqRel);
        Err(Error::new(
            ErrorCode::Capacity,
            "host reader lease slots exhausted",
        ))
    }
}
impl Clone for CreditLease {
    fn clone(&self) -> Self {
        let readers = self.slot.readers.fetch_add(1, Ordering::Relaxed);
        if readers > usize::MAX / 2 {
            abort();
        }
        Self {
            slot: self.slot.clone(),
            generation: self.generation,
        }
    }
}
impl Drop for CreditLease {
    fn drop(&mut self) {
        if self.slot.readers.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let units = self.slot.units.swap(0, Ordering::AcqRel);
        self.slot.budget.used.fetch_sub(units, Ordering::AcqRel);
        if let Some(next) = self.generation.checked_add(1) {
            self.slot.generation.store(next, Ordering::Release);
        } else {
            self.slot.retired.store(true, Ordering::Release);
        }
        self.slot.leased.store(false, Ordering::Release);
        // Clone the waker out and release the lock before invoking it: the callback is opaque, so
        // calling it under the shared budget lock would serialize every lease release behind it.
        let wake = self
            .slot
            .budget
            .wake
            .lock()
            .ok()
            .and_then(|wake| wake.clone());
        if let Some(wake) = wake {
            wake();
        }
    }
}
impl Debug for CreditLease {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("CreditLease")
            .field("generation", &self.generation)
            .field("units", &self.slot.units.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}
#[cfg(test)]
#[path = "../tests/unit/credits.rs"]
mod tests;
