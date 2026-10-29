//! AOT launch code owns TMA descriptors and cluster launches; serving never imports Python.
use super::catalog::Variant;
use crate::device::device_error;
use crate::device::library_names::CUDA_RUNTIME;
use cuda_core::sys::{CUdeviceptr, CUstream};
use infer_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::env::temp_dir;
use std::env::var_os;
use std::process::id;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

const SCALE: f32 = 0.0625;
type Initialize = unsafe extern "C" fn(i32) -> i32;
type Forward = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    i32,
    f32,
    CUstream,
) -> i32;

pub struct Api {
    initialize: Initialize,
    forward: Forward,
    initialized: Mutex<BTreeSet<i32>>,
    _library: libloading::Library,
    _runtime: libloading::Library,
}
fn cache_file(name: &str, bytes: &[u8]) -> Result<PathBuf> {
    let directory = var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
        .unwrap_or_else(temp_dir)
        .join("inference.rs/fa4");
    fs::create_dir_all(&directory).map_err(device_error)?;
    let digest = format!("{:x}", Sha256::digest(bytes));
    let path = directory.join(format!("{name}-{digest}.so"));
    if !path.exists() {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = directory.join(format!(".{digest}-{}-{sequence}.tmp", id()));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(device_error)?;
            file.write_all(bytes).map_err(device_error)?;
            file.sync_all().map_err(device_error)?;
            fs::rename(&temporary, &path).map_err(device_error)
        })();
        let _ = fs::remove_file(&temporary);
        result?;
    }
    if fs::read(&path).map_err(device_error)? != bytes {
        return Err(Error::invariant(
            "MMA host artifact cache does not match embedded bytes",
        ));
    }
    Ok(path)
}
#[expect(
    unsafe_code,
    reason = "loads only embedded hash-verified AOT host code and resolves its fixed C ABI while both libraries remain owned"
)]
fn open(name: &str, bytes: &'static [u8]) -> Result<Api> {
    let runtime = CUDA_RUNTIME
        .into_iter()
        .find_map(|name| {
            // SAFETY: loading the installed CUDA runtime preserves its standard process ABI.
            unsafe { libloading::Library::new(name).ok() }
        })
        .ok_or_else(|| Error::invalid("CUDA 13 runtime unavailable for AOT attention"))?;
    let path = cache_file(name, bytes)?;
    // SAFETY: cache_file verified the complete embedded artifact before loading it.
    let library = unsafe { libloading::Library::new(path) }.map_err(device_error)?;
    // SAFETY: the generated library exports this fixed initializer ABI; its owner is retained.
    let initialize =
        *unsafe { library.get::<Initialize>(b"infer_fa4_initialize\0") }.map_err(device_error)?;
    // SAFETY: this symbol matches host.cpp.in; its owning Library stays in Api.
    let forward =
        *unsafe { library.get::<Forward>(b"infer_fa4_forward\0") }.map_err(device_error)?;
    Ok(Api {
        initialize,
        forward,
        initialized: Mutex::new(BTreeSet::new()),
        _library: library,
        _runtime: runtime,
    })
}
impl Api {
    /// Load one named, embedded AOT artifact. The name scopes the on-disk cache so an FP16
    /// artifact can never reuse a BF16 library path even if its bytes were ever identical.
    pub fn load_artifact(name: &str, bytes: &'static [u8], device: i32) -> Result<Arc<Self>> {
        static CACHE: OnceLock<Mutex<BTreeMap<String, Arc<Api>>>> = OnceLock::new();
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .map_err(|_| Error::invariant("MMA host cache poisoned"))?;
        let api = if let Some(api) = cache.get(name) {
            Arc::clone(api)
        } else {
            let api = Arc::new(open(name, bytes)?);
            cache.insert(name.to_owned(), Arc::clone(&api));
            api
        };
        drop(cache);
        api.initialize_device(device)?;
        Ok(api)
    }
    pub fn load(variant: Variant, device: i32) -> Result<Arc<Self>> {
        Self::load_artifact(variant.name(), variant.bytes(), device)
    }
    #[expect(
        unsafe_code,
        reason = "the fixed AOT initializer only binds its retained CUDA library to the requested device before graph capture"
    )]
    fn initialize_device(&self, device: i32) -> Result<()> {
        let mut initialized = self
            .initialized
            .lock()
            .map_err(|_| Error::invariant("MMA device cache poisoned"))?;
        if !initialized.contains(&device) {
            // SAFETY: initialization is serialized for this retained library and CUDA device.
            let status = unsafe { (self.initialize)(device) };
            if status != 0 {
                return Err(Error::invalid(format!(
                    "MMA device initialization failed: {status}"
                )));
            }
            initialized.insert(device);
        }
        drop(initialized);
        Ok(())
    }
    /// The graph caller retains all six buffers and checks their exact extents and stream.
    #[expect(
        unsafe_code,
        reason = "captured caller owns checked contiguous tensors and the ABI library until its final graph use; device pointers are passed opaquely"
    )]
    pub fn launch(
        &self,
        pointers: [CUdeviceptr; super::catalog::OPERANDS],
        lanes: usize,
        capacity: usize,
        stream: CUstream,
    ) -> Result<()> {
        let [q, k, v, o, used_q, used_k] = pointers.map(|value| value as *mut c_void);
        let lanes = i32::try_from(lanes).map_err(device_error)?;
        let capacity = i32::try_from(capacity).map_err(device_error)?;
        // SAFETY: Recorded retained and checked all six contiguous buffers and the exact stream.
        let status =
            unsafe { (self.forward)(q, k, v, o, used_q, used_k, lanes, capacity, SCALE, stream) };
        if status != 0 {
            return Err(Error::invalid(format!(
                "MMA attention launch failed: {status}"
            )));
        }
        Ok(())
    }
}
