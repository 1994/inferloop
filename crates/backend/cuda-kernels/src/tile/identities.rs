//! Kernel identities remembered across processes, so a warm start does not lower every candidate.
//!
//! A candidate's identity is cuTile's L2 key, a hash of the serialized Tile IR. Deriving it runs the
//! compiler frontend, which dominated warm starts: a 27B load lowered hundreds of candidates only
//! to look up tuning records it already had. The L1 key (module, function, generics, layouts,
//! options, device and compiler identity) does not cover helper modules a kernel links in, so it is
//! sound only for one build of the kernel source; entries are therefore scoped to this executable
//! and to the device identity the backend supplies, and a rebuild starts a new store.
//!
//! The store is an append-only file of fixed-width lines, so concurrent processes can share it
//! without locking: a torn or foreign line fails to parse and is ignored, and an entry that cannot
//! be written only costs the next process one lowering. A poisoned store lock disables the store
//! for this process rather than trusting state a panic interrupted.
use cutile::cuda_async::error::DeviceError;
use cutile::error::Error as TileError;
use cutile::tile_kernel::{ModuleAstFn, Specialization};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::env::{current_exe, var_os};
use std::fs::{File, OpenOptions, create_dir_all, metadata, read_dir, remove_file};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

/// Hex characters of one SHA-256 digest, the width of both fields of a line.
const DIGEST_HEX: usize = 64;
/// Stores kept per directory: each build of each binary writes its own, and older ones are stale.
const RETAINED_STORES: usize = 16;
const STORE_EXTENSION: &str = "identities";
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
/// Bits of one hex digit.
const NIBBLE_BITS: u8 = 4;
const NIBBLE_MASK: u8 = 0x0f;

struct Store {
    /// Append target; `None` after a failed write, so later misses do not retry it.
    path: Option<PathBuf>,
    known: HashMap<String, String>,
}

fn store() -> &'static Mutex<Option<Store>> {
    static STORE: OnceLock<Mutex<Option<Store>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(None))
}

/// Open this executable's store for `scope` in the default cache directory. Without a cache
/// directory or executable identity, identities are derived every time, as before.
pub(super) fn enable(scope: &[u8]) {
    let (Some(directory), Some(executable)) = (directory(), executable_identity()) else {
        return;
    };
    let mut hasher = Sha256::new();
    hasher.update(executable);
    hasher.update(scope);
    let path = directory.join(format!("{}.{STORE_EXTENSION}", hex(&hasher.finalize())));
    let known = read(&path);
    if known.is_empty() {
        prune(&directory, &path);
    }
    if let Ok(mut slot) = store().lock() {
        *slot = Some(Store {
            path: Some(path),
            known,
        });
    }
}

/// The L2 identity of `specialization`, from the store when this build already derived it.
pub(super) fn resolve(specialization: &Specialization<ModuleAstFn>) -> Result<String, DeviceError> {
    let l1 = hex(&Sha256::digest(
        format!("{:?}", specialization.l1_cache_key()).as_bytes(),
    ));
    if let Some(known) = lookup(&l1) {
        return Ok(known);
    }
    let l2 = specialization
        .l2_cache_key()
        .map_err(|error| DeviceError::from(TileError::from(error)))?;
    remember(l1, &l2);
    Ok(l2)
}

fn lookup(l1: &str) -> Option<String> {
    store().lock().ok()?.as_ref()?.known.get(l1).cloned()
}

fn remember(l1: String, l2: &str) {
    if !is_digest(l2) {
        return;
    }
    let Ok(mut guard) = store().lock() else {
        return;
    };
    let Some(store) = guard.as_mut() else {
        return;
    };
    // A failed write keeps the identity for this process; the next one derives it again.
    if let Some(path) = &store.path
        && append(path, &l1, l2).is_err()
    {
        store.path = None;
    }
    store.known.insert(l1, l2.to_owned());
}

/// One write per line keeps concurrent appends from interleaving within an entry.
fn append(path: &Path, l1: &str, l2: &str) -> io::Result<()> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(format!("{l1} {l2}\n").as_bytes())
}

fn read(path: &Path) -> HashMap<String, String> {
    let Ok(file) = File::open(path) else {
        return HashMap::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| {
            let (l1, l2) = line.split_once(' ')?;
            (is_digest(l1) && is_digest(l2)).then(|| (l1.to_owned(), l2.to_owned()))
        })
        .collect()
}

/// Remove the oldest stores beyond [`RETAINED_STORES`], keeping `current`.
fn prune(directory: &Path, current: &Path) {
    let Ok(entries) = read_dir(directory) else {
        return;
    };
    let mut stores: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == STORE_EXTENSION))
        .filter(|path| path != current)
        .filter_map(|path| Some((metadata(&path).ok()?.modified().ok()?, path)))
        .collect();
    stores.sort_unstable_by_key(|(modified, _)| Reverse(*modified));
    for (_, stale) in stores.into_iter().skip(RETAINED_STORES - 1) {
        // A store that cannot be removed now stays for a later start to prune.
        if remove_file(stale).is_err() {
            break;
        }
    }
}

fn directory() -> Option<PathBuf> {
    let root = var_os("XDG_CACHE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    let directory = root.join("infer-cuda").join("kernel-identities");
    create_dir_all(&directory).ok()?;
    Some(directory)
}

/// The running executable's path, size and modification time: a rebuild changes at least one.
fn executable_identity() -> Option<Vec<u8>> {
    let path = current_exe().ok()?;
    let file = metadata(&path).ok()?;
    Some(
        format!(
            "{}:{}:{:?}",
            path.display(),
            file.len(),
            file.modified().ok()?
        )
        .into_bytes(),
    )
}

fn is_digest(value: &str) -> bool {
    value.len() == DIGEST_HEX && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX_DIGITS[usize::from(byte >> NIBBLE_BITS)]));
        out.push(char::from(HEX_DIGITS[usize::from(byte & NIBBLE_MASK)]));
    }
    out
}

#[cfg(test)]
mod tests;
