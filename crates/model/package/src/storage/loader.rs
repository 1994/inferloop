//! Bounded, backend-independent weight loading. No device is published until all uploads succeed.
use crate::constants::F32_BYTES;
use crate::constants::F32_BYTES_U64;
use crate::storage::safetensors::convert_float_bytes;
use crate::{ModelPackage, TensorDtype, WeightBinding};
use infer_core::digest::lower_hex;
use infer_core::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightStorage {
    /// Keep each tensor's validated F32, BF16, or F16 storage format.
    Native,
    F32,
}

#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    pub storage: WeightStorage,
    pub resident_budget_bytes: u64,
    pub staging_budget_bytes: usize,
    pub chunk_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLoadPlan {
    pub slot: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightLoadPlan {
    pub tensors: Vec<TensorLoadPlan>,
    pub source_bytes: u64,
    pub resident_bytes: u64,
    /// Conservative input-page and conversion-buffer budget; excludes backend resident allocations.
    /// Mapped input pages require no heap staging allocation.
    pub staging_bytes: usize,
    pub chunk_bytes: usize,
}

/// A backend owns allocation, upload, and stream synchronization.
///
/// Dropping a partially
/// loaded tensor must release its allocation. Upload must consume the borrowed bytes
/// before returning, or copy them into separately budgeted backend-owned staging.
pub trait WeightTarget {
    type Tensor;

    /// # Errors
    /// Returns a capacity or device error when the resident allocation cannot be created.
    fn allocate(&mut self, tensor: &TensorLoadPlan) -> Result<Self::Tensor>;

    /// # Errors
    /// Returns a bounds, transfer, or device error when the chunk cannot be uploaded.
    fn upload(&mut self, tensor: &mut Self::Tensor, byte_offset: u64, bytes: &[u8]) -> Result<()>;
}

pub struct LoadedWeights<T> {
    pub tensors: BTreeMap<String, T>,
    pub plan: WeightLoadPlan,
    /// Hashes actual uploaded bytes and tensor identities, independent of chunk size.
    pub payload_fingerprint: String,
}

impl WeightLoadPlan {
    /// Inspect metadata and reject insufficient budgets before allocating any resident tensor.
    /// # Errors
    /// Returns invalid-input, unsupported-format, or capacity errors for the requested load.
    pub fn build(package: &ModelPackage, options: LoadOptions) -> Result<Self> {
        package.validate_weight_bindings()?;
        if options.chunk_bytes < F32_BYTES || options.staging_budget_bytes == 0 {
            return Err(Error::invalid(
                "weight loading needs a chunk of at least four bytes",
            ));
        }
        let chunk_bytes = options.chunk_bytes / F32_BYTES * F32_BYTES;
        let mut tensors = Vec::with_capacity(package.manifest.bindings.len());
        let mut resident_bytes = 0u64;
        let mut staging_bytes = 0usize;
        for binding in &package.manifest.bindings {
            let tensor = plan_tensor(binding, options.storage)?;
            resident_bytes = resident_bytes
                .checked_add(tensor.bytes)
                .ok_or_else(|| Error::invalid("resident weight size overflow"))?;
            let input = usize::try_from(binding.bytes.min(chunk_bytes as u64))
                .map_err(|_| Error::invalid("weight chunk exceeds address space"))?;
            let conversion = if tensor.dtype == binding.dtype {
                0
            } else {
                input
                    .checked_div(
                        usize::try_from(binding.dtype.bytes())
                            .map_err(|_| Error::invalid("weight byte width"))?,
                    )
                    .and_then(|n| n.checked_mul(F32_BYTES))
                    .ok_or_else(|| Error::invalid("conversion staging overflow"))?
            };
            staging_bytes = staging_bytes.max(
                input
                    .checked_add(conversion)
                    .ok_or_else(|| Error::invalid("weight staging overflow"))?,
            );
            tensors.push(tensor);
        }
        if resident_bytes > options.resident_budget_bytes
            || staging_bytes > options.staging_budget_bytes
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                format!(
                    "weight load needs {resident_bytes} resident bytes and {staging_bytes} staging bytes; budgets are {} and {}",
                    options.resident_budget_bytes, options.staging_budget_bytes
                ),
            ));
        }
        Ok(Self {
            tensors,
            source_bytes: package.manifest.weights_bytes,
            resident_bytes,
            staging_bytes,
            chunk_bytes,
        })
    }
}

fn plan_tensor(binding: &WeightBinding, storage: WeightStorage) -> Result<TensorLoadPlan> {
    if !binding.dtype.is_host_float() {
        return Err(Error::unsupported(
            "weight format requires a precision package provider",
        ));
    }
    let dtype = match storage {
        WeightStorage::Native => binding.dtype,
        WeightStorage::F32 => TensorDtype::F32,
    };
    let bytes = binding
        .bytes
        .checked_div(binding.dtype.bytes())
        .and_then(|n| n.checked_mul(dtype.bytes()))
        .ok_or_else(|| Error::invalid("weight storage size overflow"))?;
    Ok(TensorLoadPlan {
        slot: binding.slot.clone(),
        shape: binding.shape.clone(),
        dtype,
        bytes,
    })
}

/// Load validated shards a bounded chunk at a time into a backend-owned target.
/// # Errors
/// Returns metadata, budget, payload, I/O, allocation, or upload errors. On failure,
/// all accumulated tensor handles are dropped and no loaded model is returned.
pub fn load_weights<T: WeightTarget>(
    package: &mut ModelPackage,
    target: &mut T,
    options: LoadOptions,
) -> Result<LoadedWeights<T::Tensor>> {
    let plan = WeightLoadPlan::build(package, options)?;
    let mut digest = Sha256::new();
    let mut tensors = BTreeMap::new();
    for (binding, spec) in package.manifest.bindings.iter().zip(&plan.tensors) {
        digest.update(serde_json::to_vec(spec).map_err(|e| Error::invalid(e.to_string()))?);
        let mut tensor = target.allocate(spec)?;
        let file = package
            .shards
            .get_mut(&binding.shard)
            .ok_or_else(|| Error::invariant("validated weight shard missing"))?;
        file.visit_float_chunks(&binding.source, plan.chunk_bytes, |offset, input| {
            if spec.dtype == binding.dtype {
                digest.update(input);
                target.upload(&mut tensor, offset, input)
            } else {
                let bytes = convert_float_bytes(input, binding.dtype)?;
                digest.update(&bytes);
                target.upload(
                    &mut tensor,
                    offset / binding.dtype.bytes() * F32_BYTES_U64,
                    &bytes,
                )
            }
        })?;
        tensors.insert(binding.slot.clone(), tensor);
    }
    Ok(LoadedWeights {
        tensors,
        plan,
        payload_fingerprint: lower_hex(&digest.finalize()),
    })
}
