use infer_core::{Error, ErrorCode, KernelId, Result};
use infer_ir::{DeviceCapabilities, OperationIr, PrecisionPlan};
use infer_spi::{KernelProvider, KernelRegistration};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Default)]
pub struct KernelRegistry {
    kernels: BTreeMap<KernelId, KernelRegistration>,
}
impl KernelRegistry {
    /// Registration is transactional: an invalid provider cannot partially install.
    ///
    /// # Errors
    /// Returns an invalid-input or conflict error for incompatible provider metadata or duplicate registrations.
    pub fn register(&mut self, provider: &impl KernelProvider) -> Result<()> {
        let candidates = provider.kernels();
        let mut seen = BTreeSet::new();
        for kernel in &candidates {
            if self.kernels.contains_key(&kernel.id) || !seen.insert(kernel.id) {
                return Err(Error::new(ErrorCode::Conflict, "duplicate kernel ID"));
            }
            if kernel.max_shape_elements == 0
                || kernel.estimated_ns == 0
                || kernel.source.file.is_empty()
            {
                return Err(Error::invalid("invalid kernel descriptor"));
            }
        }
        for candidate in candidates {
            self.kernels.insert(candidate.id, candidate);
        }
        Ok(())
    }
    #[must_use]
    pub fn get(&self, id: KernelId) -> Option<&KernelRegistration> {
        self.kernels.get(&id)
    }
    /// # Errors
    /// Returns an unsupported error if no registered kernel satisfies the operation, precision, backend, and workspace requirements.
    pub fn select(
        &self,
        operation: &OperationIr,
        precision: &PrecisionPlan,
        caps: &DeviceCapabilities,
        workspace: u64,
    ) -> Result<&KernelRegistration> {
        let elements = operation
            .shape
            .iter()
            .try_fold(1usize, |v, d| v.checked_mul(*d))
            .ok_or_else(|| Error::invalid("op shape overflow"))?;
        if elements == 0 {
            return Err(Error::invalid("empty op shape"));
        }
        self.kernels
            .values()
            .filter(|k| {
                k.operation == operation.operation
                    && k.backend == caps.backend_kind()
                    && k.precision == *precision
                    && caps.require(&k.requirements).is_ok()
                    && caps.require(&operation.requirements).is_ok()
                    && elements <= k.max_shape_elements
                    && k.workspace_bytes <= workspace
            })
            .min_by_key(|k| (Reverse(k.priority), k.estimated_ns, k.id))
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "no compatible kernel for {:?} with {:?}",
                    operation.operation, precision.compute
                ))
            })
    }
}

#[cfg(test)]
mod tests;
