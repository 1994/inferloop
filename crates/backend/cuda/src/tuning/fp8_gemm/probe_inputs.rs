//! Read-only numerical probe inputs shared during one workspace preparation.
//! Every weight/configuration still executes all original full-output comparisons.
use crate::constants::F32_BYTES;
use crate::constants::MIB;
use crate::device::{CudaDevice, device_error};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Allocation;
use std::{cell::RefCell, collections::BTreeMap, marker::PhantomData, rc::Rc, sync::Arc};

/// Bound temporary sharing independently of the number of model weights.
const MAX_BYTES: usize = 128 * MIB;
type Key = (usize, usize, usize);
struct Cache {
    stream: Arc<cuda_core::Stream>,
    inputs: BTreeMap<Key, Allocation<f32>>,
    bytes: usize,
    limit: usize,
}
thread_local! {
    static INPUTS: RefCell<Option<Cache>> = const { RefCell::new(None) };
}

/// A nested preparation restores its caller's cache, including on error or panic.
/// The guard cannot move to another thread; its tensors are released before model capture.
pub struct Scope {
    previous: Option<Cache>,
    _thread: PhantomData<Rc<()>>,
}
impl Scope {
    pub(crate) fn new(device: &CudaDevice) -> Result<Self> {
        let limit = usize::try_from(device.profile()?.arena_budget_bytes())
            .map_err(device_error)?
            .min(MAX_BYTES);
        let cache = Cache {
            stream: Arc::clone(&device.stream),
            inputs: BTreeMap::new(),
            bytes: 0,
            limit,
        };
        Ok(Self {
            previous: INPUTS.with(|inputs| inputs.replace(Some(cache))),
            _thread: PhantomData,
        })
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        INPUTS.with(|inputs| *inputs.borrow_mut() = self.previous.take());
    }
}
pub(super) fn upload(
    device: &CudaDevice,
    shape: [usize; 2],
    fixture: usize,
    values: impl FnOnce() -> Result<Vec<f32>>,
) -> Result<Allocation<f32>> {
    let [rows, columns] = shape;
    let key = (rows, columns, fixture);
    let remembered = INPUTS.with(|inputs| {
        inputs
            .borrow()
            .as_ref()
            .filter(|cache| Arc::ptr_eq(&cache.stream, &device.stream))
            .and_then(|cache| cache.inputs.get(&key).cloned())
    });
    if let Some(input) = remembered {
        return Ok(input);
    }
    let bytes = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .ok_or_else(|| Error::invalid("FP8 probe input byte overflow"))?;
    let input = device.runtime().upload(values()?)?;
    INPUTS.with(|inputs| {
        if let Some(cache) = inputs.borrow_mut().as_mut()
            && Arc::ptr_eq(&cache.stream, &device.stream)
            && cache
                .bytes
                .checked_add(bytes)
                .is_some_and(|total| total <= cache.limit)
        {
            cache.bytes += bytes;
            cache.inputs.insert(key, input.clone());
        }
    });
    Ok(input)
}

#[cfg(test)]
#[path = "../../../tests/unit/fp8_probe_inputs.rs"]
mod tests;
