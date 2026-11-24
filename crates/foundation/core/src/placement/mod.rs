//! CPU placement is an owner policy, independent of the device backend.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[allow(
    unsafe_code,
    reason = "OS affinity and memory-policy FFI is confined to this audited platform boundary"
)]
mod native;
#[cfg(test)]
mod tests;
mod topology;
use crate::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::sync_channel;
use std::thread::Builder;
use std::thread::JoinHandle;
pub use topology::{CpuTopology, PlacementPair, parse_cpu_list};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadQos {
    #[default]
    Inherit,
    Utility,
    UserInitiated,
    UserInteractive,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumaPolicy {
    Bind(usize),
    Prefer(usize),
}
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThreadPlacement {
    pub cpus: Vec<usize>,
    pub numa: Option<NumaPolicy>,
    pub qos: ThreadQos,
    /// macOS cache-locality hint; does not select a physical core or a P-core.
    pub affinity_tag: Option<i32>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementReport {
    pub cpu_binding: bool,
    pub numa: Option<NumaPolicy>,
    pub qos: ThreadQos,
    pub affinity_tag: Option<i32>,
    pub allowed_cpus: Vec<usize>,
}
impl ThreadPlacement {
    /// Apply before first-touch of owner pools, then restore the caller's policy.
    /// # Errors
    /// Rejects unsupported placement, inaccessible CPUs/nodes, or failed OS policy operations.
    pub fn scope<T>(&self, initialize: impl FnOnce() -> Result<T>) -> Result<T> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let mut guard = native::Guard::enter(self)?;
            let result = initialize();
            guard.restore()?;
            result
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            self.validate_platform()?;
            initialize()
        }
    }
    /// Handshake completes only after placement succeeds. Pools initialized by `run` use this policy.
    /// # Errors
    /// Returns thread creation or placement failures before the owner can accept requests.
    pub fn spawn(
        &self,
        name: String,
        run: impl FnOnce(PlacementReport) + Send + 'static,
    ) -> Result<JoinHandle<()>> {
        let placement = self.clone();
        let (sender, receiver) = sync_channel(1);
        let handle = Builder::new()
            .name(name)
            .spawn(move || {
                let result = placement.enter_owner();
                match result {
                    Ok(report) => {
                        let _ = sender.send(Ok(()));
                        run(report);
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                    }
                }
            })
            .map_err(|error| Error::new(ErrorCode::Backend, error.to_string()))?;
        receiver
            .recv()
            .map_err(|_| Error::new(ErrorCode::Backend, "owner placement handshake failed"))??;
        Ok(handle)
    }

    fn enter_owner(&self) -> Result<PlacementReport> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            native::Guard::enter(self)?.retain()
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            self.validate_platform()?;
            self.report()
        }
    }
    /// # Errors
    /// Returns an OS error if effective CPU affinity cannot be read.
    pub fn report(&self) -> Result<PlacementReport> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let allowed_cpus = native::allowed_cpus()?;
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let allowed_cpus = CpuTopology::discover()?.allowed_cpus;
        Ok(PlacementReport {
            cpu_binding: cfg!(target_os = "linux") && !self.cpus.is_empty(),
            numa: self.numa,
            qos: self.qos,
            affinity_tag: self.affinity_tag,
            allowed_cpus,
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn validate_platform(&self) -> Result<()> {
        if self == &Self::default() {
            Ok(())
        } else {
            Err(Error::unsupported("owner placement unavailable on this OS"))
        }
    }
}
