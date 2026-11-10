//! Prime only the immediately following token-FP8 projection's row maximum.
#[cfg(test)]
use std::cell::Cell;

#[derive(Clone, Copy)]
pub struct Pending {
    pub next: usize,
    pub pointer: u64,
    pub rows: usize,
    pub columns: usize,
}
impl Pending {
    pub fn matches(self, pointer: Option<u64>, rows: usize, columns: usize) -> bool {
        pointer == Some(self.pointer) && rows == self.rows && columns == self.columns
    }
}

#[cfg(test)]
thread_local! {
    static DISABLED: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<usize> = const { Cell::new(0) };
    static CONSUMPTIONS: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
pub(super) fn separate() -> bool {
    DISABLED.get()
}
#[cfg(test)]
pub(super) fn record_capture() {
    CAPTURES.set(CAPTURES.get() + 1);
}
#[cfg(test)]
pub fn captures() -> usize {
    CAPTURES.get()
}
#[cfg(test)]
pub fn consumptions() -> usize {
    CONSUMPTIONS.get()
}
#[cfg(test)]
pub(super) fn record_consumption(ready: bool) {
    if ready {
        CONSUMPTIONS.set(CONSUMPTIONS.get() + 1);
    }
}
#[cfg(test)]
pub fn with_separate<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let _restore = Restore(DISABLED.replace(true));
    run()
}
